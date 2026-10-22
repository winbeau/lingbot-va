"""Sparse-acceleration case study (CVPR'27 Sparse-rethinking) hooks for LingBot-VA.

A ``CaseStudyController`` is attached to every self-attention module (``attn1``). During
the action denoising loop of a call it restricts what action queries see of the current
chunk's predicted video (cache entries with ``token_type == 0`` and ``is_pred``); real
observation history, text and action keys stay dense:

``dense``   no change.
``av``      keep ``keep_ratio`` of the current chunk's predicted video keys, ranked by
            action-query attention mass at ``select_layer`` on the first action
            denoising forward of the call (conditional branch); reused for the rest of
            the call. The A->V token pruning arm.
``random``  same budget, uniformly random subset seeded per call (random-drop control).

Noise for each call is drawn from generators seeded by (episode, call index), shared by
all arms. ``snapshot_caches``/``restore_caches`` copy the transformer KV cache so a
shadow arm can be evaluated on exactly the same state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import torch

MODES = ("dense", "av", "random")


def call_seed(episode_key: str, call_index: int, offset: int = 0) -> int:
    """Same per-call seed function as the FastWAM and DreamZero case-study adapters."""
    digest = hashlib.sha256(f"{episode_key}|{int(call_index)}|{int(offset)}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


@dataclass
class CaseStudyArm:
    mode: str = "dense"
    keep_ratio: float = 0.5
    select_layer: int = 15
    video_steps: Optional[int] = None  # override job_config.num_inference_steps
    action_steps: Optional[int] = None  # override job_config.action_num_inference_steps
    seed_offset: int = 0

    @staticmethod
    def parse(raw: Optional[dict]) -> "CaseStudyArm":
        raw = dict(raw or {})
        unknown = sorted(set(raw) - set(CaseStudyArm.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown casestudy keys: {unknown}")
        arm = CaseStudyArm(**raw)
        if arm.mode not in MODES:
            raise ValueError(f"casestudy.mode must be one of {MODES}")
        if not 0.0 < arm.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        return arm

    def describe(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CaseStudyController:
    arm: CaseStudyArm = field(default_factory=CaseStudyArm)
    active: bool = False  # True only during the action loop
    seed: int = 0
    selected: Optional[torch.Tensor] = None  # kept cache-slot indices for this call
    diag: dict = field(default_factory=dict)

    def begin_call(self, arm: CaseStudyArm, seed: int) -> None:
        self.arm, self.seed, self.selected, self.active = arm, int(seed), None, False
        self.diag = {}

    def restrict(self, layer_index: int, query: torch.Tensor, key: torch.Tensor, valid: torch.Tensor,
                 token_types: torch.Tensor, is_pred: torch.Tensor):
        """Return a boolean mask over ``valid`` keys to keep (or None for all)."""
        if not self.active or self.arm.mode == "dense":
            return None
        current = (token_types == 0) & is_pred
        n_cur = int(current.sum())
        if n_cur == 0:
            return None
        k = max(1, int(round(self.arm.keep_ratio * n_cur)))
        if self.selected is None:
            if self.arm.mode == "random":
                g = torch.Generator(device="cpu").manual_seed((self.seed + 1) & 0x7FFFFFFF)
                pick = torch.randperm(n_cur, generator=g)[:k].to(valid.device)
            elif layer_index == self.arm.select_layer:
                # Mass of the conditional branch's action queries on each current video key,
                # in the softmax over all visible keys (heads averaged).
                q = query[:1].float().transpose(1, 2)  # [1, H, Sq, D]
                kk = key[:1].float().transpose(1, 2)  # [1, H, Sk, D]
                probs = torch.softmax((q @ kk.transpose(-1, -2)) * (q.shape[-1] ** -0.5), dim=-1)
                mass = probs.mean(dim=1)[0].sum(dim=0)  # [Sk]
                cur_mass = mass[current]
                pick = torch.topk(cur_mass, k).indices
                self.diag["kept_mass_fraction"] = float(cur_mass[pick].sum() / cur_mass.sum().clamp_min(1e-12))
                self.diag["select_layer"] = layer_index
            else:
                return None  # layers before the selection layer stay dense on the first forward
            cur_slots = valid[current]
            self.selected = cur_slots[pick]
            self.diag["kept_video_keys"] = int(k)
            self.diag["current_video_keys"] = n_cur
        keep = ~current | torch.isin(valid, self.selected)
        return keep


def install(transformer, controller: CaseStudyController) -> None:
    for index, block in enumerate(transformer.blocks):
        block.attn1.casestudy = controller
        block.attn1.casestudy_layer_index = index


def snapshot_caches(transformer, cache_name: str = "pos") -> list[dict]:
    snap = []
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get(cache_name) if block.attn1.attn_caches else None
        snap.append(None if cache is None else {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cache.items()})
    return snap


def restore_caches(transformer, snap: list[dict], cache_name: str = "pos") -> None:
    for block, saved in zip(transformer.blocks, snap):
        if saved is None:
            continue
        cache = block.attn1.attn_caches[cache_name]
        for k, v in saved.items():
            if torch.is_tensor(v):
                cache[k].copy_(v)
            else:
                cache[k] = v
