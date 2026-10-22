"""In-process LIBERO evaluation of LingBot-VA for the sparse-acceleration case study.

One process = model (``VA_Server``, single-rank) + a CPU-rendered LIBERO env. The rollout
is ``evaluation/libero/client.py`` without the websocket: 5 zero-action init steps after
the initial state, chunks of ``frame_chunk_size`` latent frames x ``action_per_frame``
actions (the first call skips latent frame 0), one key frame per latent frame fed back to
the KV cache, vertical image flip only, 128x128 cameras. Differences, all recorded:

* step cap ``--cap`` counts policy steps after the init steps (case-study protocol: 400);
* scene seeded per episode as in the action-eval LIBERO backend (``scene_seed: episode``):
  LIBERO re-samples fixture positions from its global numpy RNG on every reset and
  ``set_init_state`` does not restore them, so the env is reseeded before each reset from
  (seed, task, init) and an episode's scene does not depend on the shard or episode order;
* per-call noise seeded by (instruction, episode without repeat, call index);
* the case-study arm (``wan_va/casestudy.py``), optional shadow arm and episode plans;
* single process without torch.distributed (bf16 weights cast once, no FSDP wrapper);
* records in the action-eval run layout (episodes/<id>/accepted.json + attempts/0001/
  result.json, policy-calls.json, actions.npy, sim_states.npy), so the shared analysis
  scripts read them unchanged.

    python evaluation/libero/casestudy_eval.py --model-path <ckpt> --run-dir <dir> \
        --suite libero_10 --arm-file arm.yaml --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wan_va"))  # the server imports `distributed` relative to wan_va/


def episode_scene_seed(seed: int, task_id: int, init_index: int) -> int:
    """Scene RNG seed of one episode (same formula as action-eval's LIBERO backend)."""
    digest = hashlib.sha256(f"{seed}|{task_id}|{init_index}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF

def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _npy(path: Path, array: np.ndarray) -> None:
    import io

    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    _atomic(path, buf.getvalue())


def _extract_obs(obs):
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", default="cs-lingbot-va")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(50)))
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="tNNN-iMMM pairs instead of the --tasks x --inits grid (case replays)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm-file", required=True)
    ap.add_argument("--shadow-arm", default=None, help="YAML/JSON casestudy dict for a shadow arm")
    ap.add_argument("--plans", default=None)
    ap.add_argument("--shadow-executed", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--master-port", type=int, default=29600)
    args = ap.parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)

    from wan_va.configs import VA_CONFIGS
    from wan_va.casestudy import (CaseStudyArm, CaseStudyController, call_seed, install,
                                  restore_caches, snapshot_caches)
    from wan_va.wan_va_server import VA_Server
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    arm_doc = yaml.safe_load(Path(args.arm_file).read_text()) or {}
    base_arm = CaseStudyArm.parse(arm_doc.get("casestudy"))
    shadow_arm = None
    if args.shadow_arm:
        raw = args.shadow_arm
        doc = yaml.safe_load(Path(raw).read_text()) if Path(raw).exists() else json.loads(raw)
        shadow_arm = CaseStudyArm.parse(doc.get("casestudy", doc) if isinstance(doc, dict) else doc)
    plans = json.loads(Path(args.plans).read_text()) if args.plans else {}

    # torch.distributed is deliberately NOT initialized: with no process group,
    # distributed.util._configure_model casts the fp32 checkpoint to bf16 once instead of
    # wrapping it in FSDP2, whose mixed-precision policy re-casts every parameter to the
    # same bf16 values on every forward (~70 forwards per call). Same weights, no re-cast.
    config = VA_CONFIGS["libero"]
    config.wan22_pretrained_model_name_or_path = args.model_path
    config.save_root = str(Path(args.run_dir) / ".lingbot_tmp")
    config.casestudy_no_dump = True
    config.rank, config.local_rank, config.world_size = 0, 0, 1
    config.prompts = [""]
    config.prompt = ""
    server = VA_Server(config)
    controller = CaseStudyController()
    install(server.transformer, controller)
    server.cs_controller = controller

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes:
        pairs = sorted({(int(e[1:4]), int(e.split("-i")[1][:3])) for e in args.episodes})
    else:
        pairs = [(t, i) for t in args.tasks for i in args.inits]
    if args.shard == 0:
        resolved = {
            "experiment": {"name": f"{args.run_name}-{arm_doc.get('name')}-{args.suite}"},
            "benchmark": {"suite": args.suite, "task_ids": args.tasks, "initial_state_indices": args.inits,
                          "episode_subset": args.episodes, "repeats": args.repeats},
            "protocol": {"id": "sparse-casestudy-lingbot-v1", "max_steps": args.cap, "wait_steps": 5,
                         "resolution": [128, 128], "replan_steps": "native (12 first call, then 16)",
                         "seed": args.seed},
            "policy": {"model": "robbyant/lingbot-va-posttrain-libero-long", "model_path": args.model_path,
                       "arm": arm_doc, "shadow": None if shadow_arm is None else shadow_arm.describe(),
                       "plans": args.plans},
        }
        _atomic(run_dir / "resolved.yaml", yaml.safe_dump(resolved, sort_keys=False).encode())
        episodes = [{"episode_id": f"t{t:03d}-i{i:03d}-r{r:02d}-s{args.seed}", "task_id": t, "init_index": i,
                     "repeat": r, "seed": args.seed}
                    for t, i in pairs for r in range(args.repeats)]
        _atomic(run_dir / "manifest.json", json.dumps({"schema_version": "lingbot-casestudy/v1",
                                                         "episodes": episodes}, indent=1).encode())

    bench = benchmark.get_benchmark_dict()[args.suite]()
    plan_list = [(t, i, r) for t, i in pairs for r in range(args.repeats)]
    mine = [x for k, x in enumerate(plan_list) if k % args.num_shards == args.shard]
    envs: dict[int, object] = {}

    def arm_for(plan: dict, call_index: int) -> tuple[CaseStudyArm, str | None]:
        calls = plan.get("calls") or {}
        if str(call_index) in calls:
            return CaseStudyArm.parse(calls[str(call_index)]), f"call:{call_index}"
        frm = plan.get("from_call")
        if frm and call_index >= int(frm["call"]):
            return CaseStudyArm.parse(frm["casestudy"]), f"from:{frm['call']}"
        return base_arm, None

    def run_call(arm: CaseStudyArm, seed: int, obs, prompt):
        controller.begin_call(arm, seed)
        server.cs_seed = seed
        server.cs_video_steps, server.cs_action_steps = arm.video_steps, arm.action_steps
        out = server.infer(dict(obs=obs, prompt=prompt))
        return np.asarray(out["action"]), dict(controller.diag)

    for task, init, rep in mine:
        eid = f"t{task:03d}-i{init:03d}-r{rep:02d}-s{args.seed}"
        ep_dir = run_dir / "episodes" / eid
        if (ep_dir / "accepted.json").exists():
            continue
        prompt = bench.get_task(task).language
        if task not in envs:
            envs[task] = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(task),
                                            camera_heights=128, camera_widths=128)
        env = envs[task]
        init_states = bench.get_task_init_states(task)
        started = time.monotonic()
        env.seed(episode_scene_seed(args.seed, task, init))
        env.reset()
        env.set_init_state(init_states[init % init_states.shape[0]])
        for _ in range(5):
            raw, _, _, _ = env.step([0.0] * 7)
        first_obs = _extract_obs(raw)
        server.infer(dict(reset=True, prompt=prompt))
        key = f"{prompt}|{re.sub(r'-r[0-9]+', '', eid)}"
        plan = plans.get(eid, {})
        executed, sim_states, calls = [], [], []
        done, first, steps, policy_s, env_s = False, True, 0, 0.0, 0.0
        while steps < args.cap and not done:
            c = len(calls)
            arm, label = arm_for(plan, c)
            seed = call_seed(key, c, arm.seed_offset)
            sim_states.append(np.asarray(env.get_sim_state(), dtype=np.float64))
            t0 = time.monotonic()
            if shadow_arm is not None:
                before = snapshot_caches(server.transformer)
                init_latent = server.init_latent
            action, diag = run_call(arm, seed, first_obs, prompt)
            if shadow_arm is not None:
                after = snapshot_caches(server.transformer)
                restore_caches(server.transformer, before)
                shadow, _ = run_call(shadow_arm, seed, first_obs, prompt)
                restore_caches(server.transformer, after)
                del before, after
                a = action[:7].reshape(7, -1).T
                b = shadow[:7].reshape(7, -1).T
                start = action.shape[2] if first else 0
                a, b = a[start:start + args.shadow_executed], b[start:start + args.shadow_executed]
                d = a - b
                diag["shadow"] = {
                    "rms_executed": float(np.sqrt(np.mean(d[:, :6] ** 2))),
                    "gripper_disagree_executed": int(np.sum(np.sign(a[:, -1]) != np.sign(b[:, -1]))),
                    "translation_diff_executed": float(np.linalg.norm(a[:, :3].sum(0) - b[:, :3].sum(0))),
                }
                _ = init_latent
            policy_s += time.monotonic() - t0
            diag.update({"call_index": c, "seed": seed, "arm": arm.describe(),
                         "model_seconds": round(time.monotonic() - t0, 6)})
            if label:
                diag["plan_override"] = {"label": label}
            calls.append({"call_index": c + 1, "step_index": steps, "metadata": {"diagnostics": diag}})
            if os.environ.get("CS_VERBOSE"):
                print(f"  {eid} call {c} step {steps} model {diag['model_seconds']:.1f}s", flush=True)
            key_frames = []
            t1 = time.monotonic()
            start_idx = 1 if first else 0
            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4  # as in client.py
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    ee = action[:, i, j]
                    raw, _, done, _ = env.step(ee)
                    executed.append(np.asarray(ee[:7], dtype=np.float32))
                    steps += 1
                    if done or steps >= args.cap:
                        break
                    if (j + 1) % action_per_frame == 0:
                        key_frames.append(_extract_obs(raw))
                if done or steps >= args.cap:
                    break
            env_s += time.monotonic() - t1
            first = False
            if not done and steps < args.cap:
                t2 = time.monotonic()
                server.infer(dict(obs=key_frames, compute_kv_cache=True, imagine=False, state=action))
                policy_s += time.monotonic() - t2
        status = "succeeded" if done else "failed"
        result = {"schema_version": "lingbot-casestudy/episode/v1", "episode_id": eid, "attempt": 1,
                  "status": status, "success": bool(done),
                  "termination_reason": "success" if done else "max_steps", "task_id": task, "init_index": init,
                  "repeat": rep, "seed": args.seed, "instruction": prompt, "executed_steps": steps,
                  "policy_calls": len(calls), "waited_steps": 5,
                  "wall_seconds": round(time.monotonic() - started, 3), "policy_seconds": round(policy_s, 3),
                  "env_seconds": round(env_s, 3)}
        att = ep_dir / "attempts" / "0001"
        _npy(att / "actions.npy", np.asarray(executed, dtype=np.float32).reshape(-1, 7))
        _npy(att / "sim_states.npy", np.stack(sim_states))
        _npy(att / "final_sim_state.npy", np.asarray(env.get_sim_state(), dtype=np.float64))
        _atomic(att / "policy-calls.json", json.dumps(calls, indent=1).encode())
        _atomic(att / "result.json", json.dumps(result, indent=1).encode())
        _atomic(ep_dir / "accepted.json", json.dumps({"attempt": 1, "status": status}).encode())
        print(f"[{'OK  ' if done else 'FAIL'}] {eid} steps={steps} calls={len(calls)} "
              f"wall={result['wall_seconds']}s", flush=True)
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    main()
