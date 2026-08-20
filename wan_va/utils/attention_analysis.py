# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# The streaming softmax extraction follows Pyramid-Forcing's
# wan/modules/attention/capture.py (Apache-2.0).
import json
import re
import shutil
from pathlib import Path

import torch

from .attention_sparsity import TOP_P, TOP_P_TOLERANCE


def _distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _normalize_head_indices(head_indices, num_heads, device):
    if head_indices is None:
        return None
    indices = [int(index) for index in head_indices]
    if not indices or indices != sorted(set(indices)):
        raise ValueError("head_indices must be a non-empty sorted list of unique indices")
    if any(index < 0 or index >= num_heads for index in indices):
        raise ValueError(
            f"head_indices {indices} are outside the available head range "
            f"[0, {num_heads - 1}]"
        )
    return torch.tensor(indices, dtype=torch.long, device=device)


def _prepare_attention_tensors(query, key, query_mask, head_indices=None):
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError(f"query/key must be [L,H,D], got {query.shape}/{key.shape}")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError(f"query/key head shapes differ: {query.shape}/{key.shape}")
    if query_mask.ndim != 1 or query.shape[0] != query_mask.numel():
        raise ValueError("query_mask must be one-dimensional and match query tokens")
    query_mask = query_mask.to(query.device, dtype=torch.bool)
    selected_query = query[query_mask].float().transpose(0, 1)
    all_key = key.float().transpose(0, 1)
    if selected_query.shape[1] == 0 or all_key.shape[1] == 0:
        raise ValueError("The requested query/key slice is empty")
    indices = _normalize_head_indices(
        head_indices, selected_query.shape[0], selected_query.device
    )
    if indices is not None:
        selected_query = selected_query.index_select(0, indices)
        all_key = all_key.index_select(0, indices.to(all_key.device))
    return selected_query, all_key


def _select_key_tensor(key, key_mask, head_indices=None, allow_empty=False):
    if key_mask.ndim != 1 or key.shape[0] != key_mask.numel():
        raise ValueError("key mask must be one-dimensional and match key tokens")
    key_mask = key_mask.to(key.device, dtype=torch.bool)
    selected_key = key[key_mask].float().transpose(0, 1)
    if selected_key.shape[1] == 0 and not allow_empty:
        raise ValueError("The requested key slice is empty")
    if head_indices is not None:
        indices = _normalize_head_indices(
            head_indices, selected_key.shape[0], selected_key.device
        )
        selected_key = selected_key.index_select(0, indices)
    return selected_key


@torch.no_grad()
def compute_streaming_softmax_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    query_mask: torch.Tensor,
    softmax_scale: float | None = None,
    chunk_tokens: int = 512,
    head_indices=None,
):
    """Return selected queries and the exact joint visible-key softmax stats.

    The returned ``row_max`` and ``row_sumexp`` cover every visible key, not
    just a requested region. ``head_indices`` is applied before score
    computation so dual-region exports never materialize discarded heads.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    selected_query, all_key = _prepare_attention_tensors(
        query, key, query_mask, head_indices=head_indices
    )
    row_max = torch.full(
        selected_query.shape[:2],
        float("-inf"),
        dtype=torch.float32,
        device=selected_query.device,
    )
    row_sumexp = torch.zeros_like(row_max)

    for start in range(0, all_key.shape[1], chunk_tokens):
        key_chunk = all_key[:, start : start + chunk_tokens]
        scores = torch.matmul(selected_query, key_chunk.transpose(-2, -1))
        scores.mul_(softmax_scale)
        chunk_max = scores.max(dim=-1).values
        new_max = torch.maximum(row_max, chunk_max)
        row_sumexp = (
            torch.exp(row_max - new_max) * row_sumexp
            + torch.exp(scores - new_max.unsqueeze(-1)).sum(dim=-1)
        )
        row_max = new_max
    return selected_query, row_max, row_sumexp


@torch.no_grad()
def materialize_probability_slice(
    query: torch.Tensor,
    key: torch.Tensor,
    query_mask: torch.Tensor,
    key_slice_mask: torch.Tensor,
    softmax_scale: float | None = None,
    chunk_tokens: int = 512,
    head_indices=None,
    stats=None,
    allow_empty=False,
) -> torch.Tensor:
    """Materialize one raw joint-softmax region slice from shared stats."""
    if stats is None:
        stats = compute_streaming_softmax_stats(
            query=query,
            key=key,
            query_mask=query_mask,
            softmax_scale=softmax_scale,
            chunk_tokens=chunk_tokens,
            head_indices=head_indices,
        )
    selected_query, row_max, row_sumexp = stats
    selected_key = _select_key_tensor(
        key,
        key_slice_mask,
        head_indices=head_indices,
        allow_empty=allow_empty,
    )
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    selected_scores = torch.matmul(
        selected_query, selected_key.transpose(-2, -1)
    )
    selected_scores.mul_(softmax_scale)
    return torch.exp(
        selected_scores - row_max.unsqueeze(-1)
    ) / row_sumexp.clamp_min(1e-12).unsqueeze(-1)


@torch.no_grad()
def compute_token_probability_slice(
    query: torch.Tensor,
    key: torch.Tensor,
    query_mask: torch.Tensor,
    key_slice_mask: torch.Tensor,
    softmax_scale: float | None = None,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    """Compute an exact post-softmax Q-to-selected-K probability block.

    This legacy wrapper retains the old ``[H, selected Q, selected K]``
    payload and its joint denominator over all visible video/action keys.
    """
    return materialize_probability_slice(
        query=query,
        key=key,
        query_mask=query_mask,
        key_slice_mask=key_slice_mask,
        softmax_scale=softmax_scale,
        chunk_tokens=chunk_tokens,
    )


@torch.no_grad()
def compute_token_sparsity_slice(
    query: torch.Tensor,
    key: torch.Tensor,
    query_mask: torch.Tensor,
    key_slice_mask: torch.Tensor,
    softmax_scale: float | None = None,
    head_indices=None,
) -> torch.Tensor:
    """Compute legacy mean top-p VV sparsity directly from VV logits."""
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    selected_query, _ = _prepare_attention_tensors(
        query, key, query_mask, head_indices=head_indices
    )
    selected_key = _select_key_tensor(
        key, key_slice_mask, head_indices=head_indices
    )

    scores = torch.matmul(selected_query, selected_key.transpose(-2, -1))
    scores.mul_(softmax_scale)
    sorted_scores = scores.sort(dim=-1, descending=True).values.double()
    row_max = sorted_scores[..., :1].clone()
    sorted_scores.sub_(row_max)
    weights = sorted_scores.exp_()
    weights.div_(weights.sum(dim=-1, keepdim=True))
    cumulative = weights.cumsum(dim=-1)
    reached = cumulative >= TOP_P - TOP_P_TOLERANCE
    tokens_needed = torch.argmax(reached.to(torch.int32), dim=-1) + 1
    sparsity = 1.0 - tokens_needed.double() / selected_key.shape[1]
    return sparsity.mean(dim=-1)


@torch.no_grad()
def compute_joint_region_metrics(
    query: torch.Tensor,
    key: torch.Tensor,
    query_mask: torch.Tensor,
    key_region_mask: torch.Tensor,
    softmax_scale: float | None = None,
    chunk_tokens: int = 512,
    head_indices=None,
    stats=None,
):
    """Compute conditional top-p sparsity and raw joint mass for one region.

    ``row_mass`` is returned only as a short-lived validation tensor so callers
    can verify that disjoint region masses sum to one without storing matrices.
    """
    if stats is None:
        stats = compute_streaming_softmax_stats(
            query=query,
            key=key,
            query_mask=query_mask,
            softmax_scale=softmax_scale,
            chunk_tokens=chunk_tokens,
            head_indices=head_indices,
        )
    selected_query, row_max, row_sumexp = stats
    key_region_mask = key_region_mask.to(key.device, dtype=torch.bool)
    if key_region_mask.ndim != 1 or key.shape[0] != key_region_mask.numel():
        raise ValueError("key region mask must match key tokens")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    region_count = int(key_region_mask.sum().item())
    if region_count == 0:
        row_mass = torch.zeros(
            selected_query.shape[:2], dtype=torch.float64, device=query.device
        )
        nan = torch.full(
            (selected_query.shape[0],), float("nan"), dtype=torch.float64,
            device=query.device,
        )
        zero = torch.zeros_like(nan)
        return {"sparsity": nan, "mass": zero, "row_mass": row_mass}

    selected_key = _select_key_tensor(
        key, key_region_mask, head_indices=head_indices
    )
    scores = torch.matmul(selected_query, selected_key.transpose(-2, -1))
    scores.mul_(softmax_scale)
    raw_probability = torch.exp(
        scores.double() - row_max.double().unsqueeze(-1)
    ) / row_sumexp.double().clamp_min(1e-12).unsqueeze(-1)
    row_mass = raw_probability.sum(dim=-1)
    mean_mass = row_mass.mean(dim=-1)

    sorted_scores = scores.sort(
        dim=-1, descending=True, stable=True
    ).values.double()
    region_row_max = sorted_scores[..., :1].clone()
    weights = (sorted_scores - region_row_max).exp_()
    weights.div_(weights.sum(dim=-1, keepdim=True))
    cumulative = weights.cumsum(dim=-1)
    reached = cumulative >= TOP_P - TOP_P_TOLERANCE
    tokens_needed = torch.argmax(reached.to(torch.int32), dim=-1) + 1
    sparsity = (
        1.0 - tokens_needed.double() / float(region_count)
    ).mean(dim=-1)
    return {
        "sparsity": sparsity,
        "mass": mean_mass,
        "row_mass": row_mass,
    }


class JointAttentionRecorder:
    """Capture legacy VV or dual-region VA/VV attention exports."""

    def __init__(
        self,
        output_dir,
        layers,
        experiment_name,
        batch_index=0,
        chunk_tokens=512,
        worker_id=0,
        summary_layers=None,
        write_summary=True,
        step_indices=None,
        vv_chunk_sparsity=False,
        expected_chunks=None,
        export_mode=None,
        regions=("VV",),
        head_indices=None,
        overwrite=False,
    ):
        self.output_root = Path(output_dir)
        self.layers = set(layers)
        self.summary_layers = sorted(
            set(self.layers if summary_layers is None else summary_layers)
        )
        self.batch_index = batch_index
        self.chunk_tokens = int(chunk_tokens)
        self.worker_id = int(worker_id)
        self.write_summary = bool(write_summary)
        self.vv_chunk_sparsity = bool(vv_chunk_sparsity)
        if export_mode is None:
            self.export_mode = (
                "vv-chunk-sparsity" if self.vv_chunk_sparsity else "legacy-matrix"
            )
        else:
            self.export_mode = str(export_mode).strip().lower().replace("_", "-")
        if self.export_mode not in {
            "legacy-matrix",
            "vv-chunk-sparsity",
            "matrix",
            "chunk-sparsity",
        }:
            raise ValueError(f"Unknown attention export mode: {export_mode}")
        self.vv_chunk_sparsity = self.export_mode == "vv-chunk-sparsity"
        self.dual_region = self.export_mode in {"matrix", "chunk-sparsity"}
        self.is_compact = self.export_mode in {
            "vv-chunk-sparsity", "chunk-sparsity"
        }
        self.regions = tuple(str(region).upper() for region in regions)
        if self.dual_region:
            if self.regions != ("VV", "VA"):
                raise ValueError(
                    "dual-region attention export requires regions=('VV', 'VA')"
                )
        elif self.regions != ("VV",):
            raise ValueError("legacy attention export supports only the VV region")
        self.head_indices = None if head_indices is None else tuple(
            int(index) for index in head_indices
        )
        self.overwrite = bool(overwrite)
        if self.head_indices is not None and (
            not self.head_indices
            or list(self.head_indices) != sorted(set(self.head_indices))
        ):
            raise ValueError("head_indices must be a non-empty sorted list")
        if self.head_indices is not None and not self.dual_region:
            raise ValueError(
                "head_indices are supported only by the dual-region export modes"
            )
        self.expected_chunks = (
            None if expected_chunks is None else int(expected_chunks)
        )
        if self.expected_chunks is not None and self.expected_chunks <= 0:
            raise ValueError("Expected attention chunk count must be positive")
        if self.chunk_tokens <= 0:
            raise ValueError("Attention chunk size must be positive")
        self.step_indices = (
            None
            if step_indices is None
            else sorted(set(int(step) for step in step_indices))
        )
        if self.step_indices is not None and any(
            step < 0 for step in self.step_indices
        ):
            raise ValueError("Attention step indices must be non-negative")
        self._saved = True
        self._last_summary = None
        self.actual_head_indices = None
        self._available_head_count = None
        self.start_experiment(experiment_name)

    def start_experiment(self, experiment_name):
        """Reset recorder state for another prompt after the prior save."""
        if not self._saved:
            raise RuntimeError(
                "Cannot start a new attention experiment before saving the current one"
            )
        self.output_dir = self.output_root / _slugify(experiment_name)
        success_path = self.output_dir / "_SUCCESS"
        if success_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Attention output is already complete: {self.output_dir}; "
                "pass --overwrite to replace it"
            )
        if success_path.exists() and self.overwrite:
            if self.worker_id == 0:
                shutil.rmtree(self.output_dir)
            _distributed_barrier()
        self.blocks_root = self.output_dir / ".attention-blocks"
        self.blocks_dir = self.blocks_root / f"worker_{self.worker_id:02d}"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        if self.step_indices is None:
            self.block_files = {layer: [] for layer in self.layers}
            self.chunk_metadata = []
        else:
            self.block_files = {
                step: {layer: [] for layer in self.layers}
                for step in self.step_indices
            }
            self.chunk_metadata = {step: [] for step in self.step_indices}
        self._saved = False

    @torch.no_grad()
    def capture(
        self,
        layer_index,
        query,
        key,
        query_token_types,
        key_token_types,
        query_frame_ids,
        key_frame_ids,
        update_cache,
        cache_name,
        attention_step_index=None,
        attention_timestep=None,
    ):
        if layer_index not in self.layers:
            return
        step_index = (
            None if attention_step_index is None else int(attention_step_index)
        )
        if self.step_indices is None:
            if update_cache != 1:
                return
            capture_step = None
            block_files = self.block_files
            chunk_metadata = self.chunk_metadata
        else:
            if step_index not in self.step_indices:
                return
            capture_step = step_index
            block_files = self.block_files[capture_step]
            chunk_metadata = self.chunk_metadata[capture_step]

        query_token_types = query_token_types.to(query.device)
        key_token_types = key_token_types.to(query.device)
        query_frame_ids = query_frame_ids.to(query.device)
        key_frame_ids = key_frame_ids.to(query.device)
        current_video_query = query_token_types == 0
        if not torch.any(current_video_query):
            return
        visible_video_history = key_token_types == 0
        visible_action_history = key_token_types == 1
        if not torch.any(visible_video_history):
            raise ValueError("Video query has no visible video-history key")
        if self.dual_region:
            if int(visible_video_history.sum().item()) + int(
                visible_action_history.sum().item()
            ) != key.shape[1]:
                raise ValueError(
                    "Dual-region attention requires every visible key to be "
                    "video or action"
                )

        batch_index = min(self.batch_index, query.shape[0] - 1)
        chunk_index = len(block_files[layer_index])
        query_batch = query[batch_index]
        key_batch = key[batch_index]
        if self.dual_region:
            available_heads = int(query_batch.shape[1])
            if self._available_head_count is None:
                self._available_head_count = available_heads
                if self.head_indices is None:
                    self.actual_head_indices = tuple(range(available_heads))
                else:
                    _normalize_head_indices(
                        self.head_indices, available_heads, query_batch.device
                    )
                    self.actual_head_indices = self.head_indices
            elif available_heads != self._available_head_count:
                raise ValueError("Attention layers expose inconsistent head counts")
            head_indices = self.head_indices
            stats = compute_streaming_softmax_stats(
                query=query_batch,
                key=key_batch,
                query_mask=current_video_query,
                chunk_tokens=self.chunk_tokens,
                head_indices=head_indices,
            )
            if self.export_mode == "chunk-sparsity":
                vv_stats = compute_joint_region_metrics(
                    query=query_batch,
                    key=key_batch,
                    query_mask=current_video_query,
                    key_region_mask=visible_video_history,
                    chunk_tokens=self.chunk_tokens,
                    head_indices=head_indices,
                    stats=stats,
                )
                va_stats = compute_joint_region_metrics(
                    query=query_batch,
                    key=key_batch,
                    query_mask=current_video_query,
                    key_region_mask=visible_action_history,
                    chunk_tokens=self.chunk_tokens,
                    head_indices=head_indices,
                    stats=stats,
                )
                row_mass_error = (
                    vv_stats["row_mass"] + va_stats["row_mass"] - 1.0
                ).abs().max()
                block_files[layer_index].append({
                    "sparsity": {
                        "VV": vv_stats["sparsity"].cpu(),
                        "VA": va_stats["sparsity"].cpu(),
                    },
                    "mass": {
                        "VV": vv_stats["mass"].cpu(),
                        "VA": va_stats["mass"].cpu(),
                    },
                })
            else:
                vv_probability = materialize_probability_slice(
                    query=query_batch,
                    key=key_batch,
                    query_mask=current_video_query,
                    key_slice_mask=visible_video_history,
                    chunk_tokens=self.chunk_tokens,
                    head_indices=head_indices,
                    stats=stats,
                ).cpu()
                va_probability = materialize_probability_slice(
                    query=query_batch,
                    key=key_batch,
                    query_mask=current_video_query,
                    key_slice_mask=visible_action_history,
                    chunk_tokens=self.chunk_tokens,
                    head_indices=head_indices,
                    stats=stats,
                    allow_empty=True,
                ).cpu()
                row_mass_error = (
                    vv_probability.sum(dim=-1)
                    + va_probability.sum(dim=-1)
                    - 1.0
                ).abs().max()
                step_dir = (
                    self.blocks_dir
                    if capture_step is None
                    else self.blocks_dir / f"step_{capture_step:03d}"
                )
                layer_dir = step_dir / f"layer_{layer_index:02d}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                block_path = layer_dir / f"chunk_{chunk_index:03d}.pt"
                torch.save({
                    "attention": {"VV": vv_probability, "VA": va_probability},
                    "head_indices": list(self.actual_head_indices or ()),
                }, block_path)
                block_files[layer_index].append(block_path)
            row_mass_error_value = float(row_mass_error.item())
            if not torch.isfinite(row_mass_error).item() or row_mass_error_value > 5e-4:
                raise RuntimeError(
                    "Joint VA/VV row masses do not sum to one: "
                    f"max error={row_mass_error_value:.6g}"
                )
            del stats
        elif self.vv_chunk_sparsity:
            head_sparsity = compute_token_sparsity_slice(
                query=query_batch,
                key=key_batch,
                query_mask=current_video_query,
                key_slice_mask=visible_video_history,
            )
            block_files[layer_index].append(head_sparsity.cpu())
        else:
            probabilities = compute_token_probability_slice(
                query=query_batch,
                key=key_batch,
                query_mask=current_video_query,
                key_slice_mask=visible_video_history,
                chunk_tokens=self.chunk_tokens,
            ).cpu()
            step_dir = (
                self.blocks_dir
                if capture_step is None
                else self.blocks_dir / f"step_{capture_step:03d}"
            )
            layer_dir = step_dir / f"layer_{layer_index:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            block_path = layer_dir / f"chunk_{chunk_index:03d}.pt"
            torch.save({"attention": probabilities}, block_path)
            block_files[layer_index].append(block_path)

        query_frames = query_frame_ids[current_video_query]
        query_tokens = int(query_frames.numel())
        visible_video_tokens = int(visible_video_history.sum().item())
        visible_action_tokens = int(visible_action_history.sum().item())
        timestep_value = None
        if attention_timestep is not None:
            if isinstance(attention_timestep, torch.Tensor):
                timestep_value = float(attention_timestep.detach().cpu().item())
            else:
                timestep_value = float(attention_timestep)
        metadata = {
            "chunk_index": chunk_index,
            "step_index": step_index,
            "scheduler_timestep": timestep_value,
            "query_tokens": query_tokens,
            "visible_video_history_tokens": visible_video_tokens,
            "visible_action_history_tokens": visible_action_tokens,
            "all_key_tokens_in_softmax": int(key.shape[1]),
            "query_frame_start": int(query_frames.min().item()),
            "query_frame_end": int(query_frames.max().item()) + 1,
            "cache_name": cache_name,
        }
        if self.dual_region:
            metadata["query_token_ids"] = _region_token_ids(query_frames)
            metadata["region_token_ids"] = {
                "VV": _region_token_ids(key_frame_ids[visible_video_history]),
                "VA": _region_token_ids(key_frame_ids[visible_action_history]),
            }
            metadata["joint_mass_max_abs_error"] = float(row_mass_error.item())
        elif not self.vv_chunk_sparsity:
            metadata["query_token_ids"] = _video_token_ids(query_frames)
            metadata["key_token_ids"] = _video_token_ids(
                key_frame_ids[visible_video_history]
            )
        if chunk_index == len(chunk_metadata):
            chunk_metadata.append(metadata)
        else:
            expected = chunk_metadata[chunk_index]
            fields = [
                "query_tokens",
                "visible_video_history_tokens",
                "visible_action_history_tokens",
                "all_key_tokens_in_softmax",
                "query_frame_start",
                "query_frame_end",
            ]
            if self.dual_region:
                fields.extend(["query_token_ids", "region_token_ids"])
            elif not self.vv_chunk_sparsity:
                fields.extend(["query_token_ids", "key_token_ids"])
            for field in fields:
                if metadata[field] != expected[field]:
                    raise ValueError(
                        f"Layer {layer_index} chunk {chunk_index} has inconsistent {field}"
                    )

    def save(self, publish_summary=None):
        if not self.layers:
            # Rank with no assigned layers (layers < world_size): nothing
            # to export, but keep the per-worker cleanup symmetric.
            if self.blocks_dir.exists():
                shutil.rmtree(self.blocks_dir)
            try:
                self.blocks_root.rmdir()
            except OSError:
                pass
            _distributed_barrier()
            _distributed_barrier()
            self._last_summary = None
            self._saved = True
            return None
        if self.step_indices is None:
            groups = [
                (None, self.block_files, self.chunk_metadata, self.output_dir)
            ]
        else:
            missing_steps = [
                step
                for step in self.step_indices
                if not any(self.block_files[step].values())
            ]
            if missing_steps:
                raise RuntimeError(
                    "No current-video attention blocks were captured for step "
                    f"indices {missing_steps}"
                )
            groups = [
                (
                    step,
                    self.block_files[step],
                    self.chunk_metadata[step],
                    self.output_dir / f"step_{step:03d}",
                )
                for step in self.step_indices
            ]

        step_summaries = [
            self._save_group(
                requested_step=step,
                block_files=block_files,
                chunk_metadata=chunk_metadata,
                output_dir=output_dir,
            )
            for step, block_files, chunk_metadata, output_dir in groups
        ]
        if self.dual_region:
            common_summary = {
                "semantics": (
                    "video-query attention uses one joint softmax denominator "
                    "over all visible video and action-history keys"
                ),
                "attention_query_region": "V",
                "attention_key_regions": list(self.regions),
                "softmax_scope": "all visible video and action keys",
                "source_phase": "i2va video denoising forward",
                "joint_stage_validation": (
                    "every chunk after chunk 0 contains action-history keys "
                    "in the same second-stage video forward"
                ),
                "step_index_semantics": (
                    "zero-based index in the padded video denoising loop; "
                    "scheduler_timestep stores the actual scheduler value"
                ),
                "csv_header": False,
                "future_columns": "empty CSV fields",
                "layer_indices": self.summary_layers,
                "head_indices": list(self.actual_head_indices or ()),
                "num_heads": len(self.actual_head_indices or ()),
                "chunk_tokens": self.chunk_tokens,
                "joint_softmax": True,
            }
            if self.export_mode == "chunk-sparsity":
                common_summary.update({
                    "sparsity_scope": "region-conditional",
                    "sparsity_top_p": TOP_P,
                    "sparsity_formula": "1 - k / N_region",
                    "sparsity_reduction": (
                        "mean over current-video query tokens within each chunk"
                    ),
                    "mass_formula": (
                        "mean over queries of raw joint-softmax probability mass"
                    ),
                    "empty_region_policy": {
                        "VA_chunk_0_sparsity": "nan",
                        "VA_chunk_0_mass": 0.0,
                    },
                    "csv_patterns": {
                        "sparsity": "step_XXX/sparsity/{region}/lXX.csv",
                        "mass": "step_XXX/mass/{region}/lXX.csv",
                    },
                })
                csv_pattern = "step_XXX/{metric}/{region}/lXX.csv"
                step_csv_pattern = csv_pattern
            else:
                common_summary.update({
                    "matrix_probability": "raw joint-softmax probability",
                    "csv_patterns": {
                        "VV": "step_XXX/vv/lXXhXX.csv",
                        "VA": "step_XXX/va/lXXhXX.csv",
                    },
                    "regions": ["VV", "VA"],
                })
                csv_pattern = "step_XXX/{region}/lXXhXX.csv"
                step_csv_pattern = csv_pattern
        elif self.vv_chunk_sparsity:
            common_summary = {
                "semantics": (
                    "each layer CSV stores second-stage i2va video-to-video "
                    "top-p sparsity; rows are heads and columns are chunks; "
                    "the video-only softmax is mathematically equivalent to "
                    "renormalizing the VV slice of the full multimodal softmax"
                ),
                "attention_region": "VV",
                "source_phase": "i2va video denoising forward",
                "joint_stage_validation": (
                    "every chunk after chunk 0 contains action-history keys "
                    "in the full multimodal softmax"
                ),
                "step_index_semantics": (
                    "zero-based index in the padded video denoising loop; "
                    "scheduler_timestep stores the actual scheduler value"
                ),
                "csv_header": False,
                "csv_rows": "attention heads in ascending index order",
                "csv_columns": "generated chunks in ascending index order",
                "sparsity_top_p": TOP_P,
                "sparsity_formula": "1 - k / N",
                "sparsity_reduction": (
                    "mean over current-video query tokens within each chunk"
                ),
                "layer_indices": self.summary_layers,
                "compute_path": "direct GPU VV logits with float64 accumulation",
            }
            csv_pattern = "lXX.csv"
            step_csv_pattern = "step_XXX/lXX.csv"
        else:
            common_summary = {
                "semantics": (
                    "rows concatenate each chunk's current-video Q; columns "
                    "are chronological video-history K; softmax denominators "
                    "include all KV modalities visible at that chunk"
                ),
                "step_index_semantics": (
                    "zero-based index in the padded video denoising loop; "
                    "scheduler_timestep stores the actual scheduler value"
                ),
                "csv_header": False,
                "future_columns": "empty CSV fields",
                "layer_indices": self.summary_layers,
                "chunk_tokens": self.chunk_tokens,
            }
            csv_pattern = "lXXhXX.csv"
            step_csv_pattern = "step_XXX/lXXhXX.csv"
        if self.step_indices is None:
            summary = {
                **common_summary,
                "csv_pattern": csv_pattern,
                **step_summaries[0],
            }
        else:
            summary = {
                **common_summary,
                "csv_pattern": step_csv_pattern,
                "requested_step_indices": self.step_indices,
                "steps": step_summaries,
            }

        self._last_summary = summary
        summary_path = self.output_dir / "summary.json"
        if publish_summary is None:
            publish_summary = self.write_summary
        if self.blocks_dir.exists():
            shutil.rmtree(self.blocks_dir)
        try:
            self.blocks_root.rmdir()
        except OSError:
            pass
        _distributed_barrier()
        if publish_summary:
            self.publish_summary()
        _distributed_barrier()
        self._saved = True
        return summary_path

    def publish_summary(self):
        """Atomically publish the last summary and complete marker."""
        if self._last_summary is None:
            raise RuntimeError("No saved attention summary is available")
        summary_path = self.output_dir / "summary.json"
        temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(self._last_summary, indent=2))
        temporary_path.replace(summary_path)
        if self.dual_region:
            success_path = self.output_dir / "_SUCCESS"
            success_tmp = success_path.with_suffix(success_path.suffix + ".tmp")
            success_tmp.touch()
            success_tmp.replace(success_path)
        return summary_path

    def _save_group(
        self,
        requested_step,
        block_files,
        chunk_metadata,
        output_dir,
    ):
        layer_indices = sorted(
            layer for layer, paths in block_files.items() if paths
        )
        if not layer_indices:
            raise RuntimeError("No current-video attention blocks were captured")
        chunk_count = len(chunk_metadata)
        if any(len(block_files[layer]) != chunk_count for layer in layer_indices):
            raise RuntimeError("Attention layers captured different numbers of chunks")
        if self.expected_chunks is not None and chunk_count != self.expected_chunks:
            raise RuntimeError(
                f"Captured {chunk_count} attention chunks; expected "
                f"{self.expected_chunks}"
            )
        chunk_indices = [int(metadata["chunk_index"]) for metadata in chunk_metadata]
        if chunk_indices != list(range(chunk_count)):
            raise RuntimeError(
                f"Attention chunks are not contiguous: {chunk_indices}"
            )

        step_indices = {metadata["step_index"] for metadata in chunk_metadata}
        scheduler_timesteps = {
            metadata["scheduler_timestep"] for metadata in chunk_metadata
        }
        if len(step_indices) != 1 or len(scheduler_timesteps) != 1:
            raise RuntimeError("Attention chunks have inconsistent timestep metadata")

        step_index = next(iter(step_indices))
        scheduler_timestep = next(iter(scheduler_timesteps))
        num_heads = self._num_heads(block_files, layer_indices[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.dual_region:
            public_chunks = [dict(metadata) for metadata in chunk_metadata]
        else:
            public_chunks = [
                {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"query_token_ids", "key_token_ids"}
                }
                for metadata in chunk_metadata
            ]

        if self.is_compact:
            if self.dual_region:
                self._validate_joint_regions(chunk_metadata)
                for layer_index in layer_indices:
                    self._save_layer_dual_sparsity_csv(
                        output_dir=output_dir,
                        block_files=block_files,
                        layer_index=layer_index,
                        num_heads=num_heads,
                        chunk_count=chunk_count,
                    )
                max_mass_error = max(
                    float(metadata["joint_mass_max_abs_error"])
                    for metadata in chunk_metadata
                )
                return {
                    "requested_step_index": requested_step,
                    "step_index": step_index,
                    "scheduler_timestep": scheduler_timestep,
                    "csv_directory": str(output_dir.relative_to(self.output_dir) or "."),
                    "csv_shape_per_layer": [num_heads, chunk_count],
                    "num_heads": num_heads,
                    "num_chunks": chunk_count,
                    "head_indices": list(self.actual_head_indices or ()),
                    "chunk_indices": chunk_indices,
                    "joint_stage_verified": True,
                    "mass_max_abs_row_error": max_mass_error,
                    "regions": {
                        "VV": {
                            "sparsity_csv_directory": str(
                                (output_dir / "sparsity" / "vv").relative_to(self.output_dir)
                            ),
                            "mass_csv_directory": str(
                                (output_dir / "mass" / "vv").relative_to(self.output_dir)
                            ),
                        },
                        "VA": {
                            "sparsity_csv_directory": str(
                                (output_dir / "sparsity" / "va").relative_to(self.output_dir)
                            ),
                            "mass_csv_directory": str(
                                (output_dir / "mass" / "va").relative_to(self.output_dir)
                            ),
                        },
                    },
                    "chunks": public_chunks,
                }
            self._validate_joint_vv_chunks(chunk_metadata)
            for layer_index in layer_indices:
                self._save_layer_sparsity_csv(
                    output_dir=output_dir,
                    block_files=block_files,
                    layer_index=layer_index,
                    num_heads=num_heads,
                    chunk_count=chunk_count,
                )
            return {
                "requested_step_index": requested_step,
                "step_index": step_index,
                "scheduler_timestep": scheduler_timestep,
                "csv_directory": str(output_dir.relative_to(self.output_dir) or "."),
                "csv_shape_per_layer": [num_heads, chunk_count],
                "num_heads": num_heads,
                "num_chunks": chunk_count,
                "chunk_indices": chunk_indices,
                "joint_stage_verified": True,
                "chunks": public_chunks,
            }

        if self.dual_region:
            self._validate_joint_regions(chunk_metadata)
            row_token_ids = [
                token
                for metadata in chunk_metadata
                for token in metadata["query_token_ids"]
            ]
            history_token_ids = {
                region: _ordered_union(
                    metadata["region_token_ids"][region]
                    for metadata in chunk_metadata
                )
                for region in self.regions
            }
            for layer_index in layer_indices:
                self._save_layer_dual_csvs(
                    output_dir=output_dir,
                    block_files=block_files,
                    chunk_metadata=chunk_metadata,
                    layer_index=layer_index,
                    num_heads=num_heads,
                    history_token_ids=history_token_ids,
                )
            return {
                "requested_step_index": requested_step,
                "step_index": step_index,
                "scheduler_timestep": scheduler_timestep,
                "csv_directory": str(output_dir.relative_to(self.output_dir) or "."),
                "logical_shape_per_csv": {
                    region: [len(row_token_ids), len(history_token_ids[region])]
                    for region in self.regions
                },
                "num_heads": num_heads,
                "num_chunks": chunk_count,
                "head_indices": list(self.actual_head_indices or ()),
                "row_token_ids": row_token_ids,
                "row_chunk_boundaries": _cumulative_boundaries(
                    metadata["query_tokens"] for metadata in chunk_metadata
                ),
                "regions": {
                    region: {
                        "history_token_ids": history_token_ids[region],
                        "history_chunk_boundaries": _history_boundaries(
                            chunk_metadata,
                            history_token_ids[region],
                            token_field=region,
                        ),
                        "csv_directory": str(
                            (output_dir / region.lower()).relative_to(self.output_dir)
                        ),
                        "csv_pattern": "lXXhXX.csv",
                    }
                    for region in self.regions
                },
                "joint_stage_verified": True,
                "mass_max_abs_row_error": max(
                    float(metadata["joint_mass_max_abs_error"])
                    for metadata in chunk_metadata
                ),
                "chunks": public_chunks,
            }

        row_token_ids = [
            token
            for metadata in chunk_metadata
            for token in metadata["query_token_ids"]
        ]
        history_token_ids = _ordered_union(
            metadata["key_token_ids"] for metadata in chunk_metadata
        )
        logical_shape = [len(row_token_ids), len(history_token_ids)]
        for layer_index in layer_indices:
            self._save_layer_csvs(
                output_dir=output_dir,
                block_files=block_files,
                chunk_metadata=chunk_metadata,
                layer_index=layer_index,
                num_heads=num_heads,
                history_token_ids=history_token_ids,
            )

        return {
            "requested_step_index": requested_step,
            "step_index": step_index,
            "scheduler_timestep": scheduler_timestep,
            "csv_directory": str(output_dir.relative_to(self.output_dir) or "."),
            "logical_shape_per_csv": logical_shape,
            "num_heads": num_heads,
            "num_chunks": chunk_count,
            "row_token_ids": row_token_ids,
            "history_token_ids": history_token_ids,
            "row_chunk_boundaries": _cumulative_boundaries(
                metadata["query_tokens"] for metadata in chunk_metadata
            ),
            "history_chunk_boundaries": _history_boundaries(
                chunk_metadata, history_token_ids
            ),
            "chunks": public_chunks,
        }

    def _validate_joint_vv_chunks(self, chunk_metadata):
        if len(chunk_metadata) < 2:
            raise RuntimeError(
                "At least two chunks are required to verify second-stage VV "
                "attention"
            )
        missing_action_history = [
            int(metadata["chunk_index"])
            for metadata in chunk_metadata[1:]
            if int(metadata["visible_action_history_tokens"]) <= 0
        ]
        if missing_action_history:
            raise RuntimeError(
                "VV chunk sparsity requires the second-stage i2va cache; "
                "no action-history keys were visible in chunks "
                f"{missing_action_history}"
            )

    def _validate_joint_regions(self, chunk_metadata):
        if len(chunk_metadata) < 2:
            raise RuntimeError(
                "At least two chunks are required to verify second-stage "
                "joint VA/VV attention"
            )
        if int(chunk_metadata[0]["visible_action_history_tokens"]) != 0:
            raise RuntimeError("Chunk 0 unexpectedly contains action-history keys")
        missing_action_history = [
            int(metadata["chunk_index"])
            for metadata in chunk_metadata[1:]
            if int(metadata["visible_action_history_tokens"]) <= 0
        ]
        if missing_action_history:
            raise RuntimeError(
                "Joint VA/VV export requires second-stage i2va action-history "
                "keys in chunks "
                f"{missing_action_history}; refusing pure-video export"
            )
        for metadata in chunk_metadata:
            visible_video = int(metadata["visible_video_history_tokens"])
            visible_action = int(metadata["visible_action_history_tokens"])
            all_keys = int(metadata["all_key_tokens_in_softmax"])
            if visible_video + visible_action != all_keys:
                raise RuntimeError(
                    f"Chunk {metadata['chunk_index']} has inconsistent key counts: "
                    f"VV={visible_video}, VA={visible_action}, all={all_keys}"
                )

    def _num_heads(self, block_files, layer_index):
        first_block = block_files[layer_index][0]
        if self.dual_region:
            if self.is_compact:
                values = first_block.get("sparsity", {}).get("VV")
                if values is None:
                    raise RuntimeError("Compact dual block is missing VV sparsity")
                return int(values.shape[0])
            payload = torch.load(first_block, map_location="cpu", weights_only=True)
            attention = payload.get("attention")
            if not isinstance(attention, dict) or "VV" not in attention:
                raise RuntimeError("Dual attention block is missing VV payload")
            return int(attention["VV"].shape[0])
        if self.vv_chunk_sparsity:
            return int(first_block.shape[0])
        payload = torch.load(first_block, map_location="cpu", weights_only=True)
        if "attention" not in payload:
            raise RuntimeError("Attention block is missing required key 'attention'")
        return int(payload["attention"].shape[0])

    def _save_layer_dual_sparsity_csv(
        self,
        output_dir,
        block_files,
        layer_index,
        num_heads,
        chunk_count,
    ):
        for metric in ("sparsity", "mass"):
            for region in self.regions:
                chunk_values = []
                for payload in block_files[layer_index]:
                    values = payload[metric][region].double()
                    if values.shape != (num_heads,):
                        raise RuntimeError(
                            f"Layer {layer_index} {metric}/{region} has invalid "
                            f"shape {tuple(values.shape)}; expected {(num_heads,)}"
                        )
                    chunk_values.append(values)
                matrix = torch.stack(chunk_values, dim=1).cpu()
                if matrix.shape != (num_heads, chunk_count):
                    raise RuntimeError(
                        f"Layer {layer_index} {metric}/{region} has invalid "
                        f"matrix shape {tuple(matrix.shape)}"
                    )
                csv_dir = output_dir / metric / region.lower()
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_path = csv_dir / f"l{layer_index:02d}.csv"
                temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
                with temporary_path.open("w") as handle:
                    for row in matrix.tolist():
                        handle.write(",".join(
                            f"{float(value):.9g}" for value in row
                        ))
                        handle.write("\n")
                temporary_path.replace(csv_path)

    def _save_layer_dual_csvs(
        self,
        output_dir,
        block_files,
        chunk_metadata,
        layer_index,
        num_heads,
        history_token_ids,
    ):
        from contextlib import ExitStack

        selected_heads = list(self.actual_head_indices or range(num_heads))
        handles = {}
        temporary_paths = {}
        for region in self.regions:
            region_dir = output_dir / region.lower()
            region_dir.mkdir(parents=True, exist_ok=True)
            for head_index in selected_heads:
                csv_path = region_dir / (
                    f"l{layer_index:02d}h{head_index:02d}.csv"
                )
                temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
                temporary_paths[(region, head_index)] = (csv_path, temporary_path)

        with ExitStack() as stack:
            for key, (_, temporary_path) in temporary_paths.items():
                handles[key] = stack.enter_context(
                    temporary_path.open("w", buffering=1024 * 1024)
                )
            lookups = {
                region: {
                    tuple(token): index
                    for index, token in enumerate(history_token_ids[region])
                }
                for region in self.regions
            }
            for metadata, block_path in zip(
                chunk_metadata, block_files[layer_index]
            ):
                payload = torch.load(
                    block_path, map_location="cpu", weights_only=True
                )
                attention = payload.get("attention")
                if not isinstance(attention, dict):
                    raise RuntimeError(
                        "Dual matrix payload must contain attention regions"
                    )
                payload_heads = payload.get("head_indices")
                if payload_heads != selected_heads:
                    raise RuntimeError(
                        f"Layer {layer_index} payload head ids {payload_heads} "
                        f"do not match selected heads {selected_heads}"
                    )
                for region in self.regions:
                    block = attention.get(region)
                    if block is None:
                        raise RuntimeError(
                            f"Dual matrix payload is missing region {region}"
                        )
                    if block.shape[0] != num_heads or block.shape[1] != metadata["query_tokens"]:
                        raise RuntimeError(
                            f"Layer {layer_index} {region} block has invalid "
                            f"shape {tuple(block.shape)}"
                        )
                    region_tokens = metadata["region_token_ids"][region]
                    if block.shape[2] != len(region_tokens):
                        raise RuntimeError(
                            f"Layer {layer_index} {region} key shape does not "
                            "match token metadata"
                        )
                    columns = [
                        lookups[region][tuple(token)] for token in region_tokens
                    ]
                    block_array = block.float().numpy()
                    for position, head_index in enumerate(selected_heads):
                        _write_csv_block(
                            handle=handles[(region, head_index)],
                            block=block_array[position],
                            columns=columns,
                            total_columns=len(history_token_ids[region]),
                        )
                del payload
        for csv_path, temporary_path in temporary_paths.values():
            temporary_path.replace(csv_path)

    def _save_layer_sparsity_csv(
        self,
        output_dir,
        block_files,
        layer_index,
        num_heads,
        chunk_count,
    ):
        chunk_values = []
        for values in block_files[layer_index]:
            values = values.double()
            if values.shape != (num_heads,):
                raise RuntimeError(
                    f"Layer {layer_index} has invalid chunk sparsity shape "
                    f"{tuple(values.shape)}; expected {(num_heads,)}"
                )
            chunk_values.append(values)
        matrix = torch.stack(chunk_values, dim=1).cpu()
        if matrix.shape != (num_heads, chunk_count):
            raise RuntimeError(
                f"Layer {layer_index} has invalid sparsity matrix shape "
                f"{tuple(matrix.shape)}"
            )

        csv_path = output_dir / f"l{layer_index:02d}.csv"
        temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with temporary_path.open("w") as handle:
            for row in matrix.tolist():
                handle.write(",".join(f"{float(value):.9g}" for value in row))
                handle.write("\n")
        temporary_path.replace(csv_path)

    def _save_layer_csvs(
        self,
        output_dir,
        block_files,
        chunk_metadata,
        layer_index,
        num_heads,
        history_token_ids,
    ):
        payloads = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in block_files[layer_index]
        ]
        history_lookup = {
            tuple(token): index for index, token in enumerate(history_token_ids)
        }
        for head_index in range(num_heads):
            csv_path = output_dir / f"l{layer_index:02d}h{head_index:02d}.csv"
            with csv_path.open("w", buffering=1024 * 1024) as handle:
                for metadata, payload in zip(chunk_metadata, payloads):
                    block = payload["attention"][head_index].float().numpy()
                    columns = [
                        history_lookup[tuple(token)]
                        for token in metadata["key_token_ids"]
                    ]
                    _write_csv_block(
                        handle=handle,
                        block=block,
                        columns=columns,
                        total_columns=len(history_token_ids),
                    )
        del payloads


def _write_csv_block(handle, block, columns, total_columns):
    import numpy as np

    if block.ndim != 2:
        raise ValueError(f"attention block must be [Q,K], got {block.shape}")
    if len(columns) != block.shape[1]:
        raise ValueError(
            f"column metadata has {len(columns)} entries but block has "
            f"{block.shape[1]} columns"
        )
    if total_columns < len(columns):
        raise ValueError(
            f"logical width {total_columns} is smaller than visible width "
            f"{len(columns)}"
        )
    if any(column < 0 or column >= total_columns for column in columns):
        raise ValueError("attention column metadata is outside logical width")

    # The default non-evicting cache yields a chronological prefix. This path
    # avoids constructing a dense all-history matrix and writes future columns
    # as empty CSV fields.
    if columns == list(range(len(columns))):
        formatted = np.char.mod("%.9g", block)
        if not columns:
            suffix = "," * max(total_columns - 1, 0)
            for _ in range(formatted.shape[0]):
                handle.write(suffix)
                handle.write("\n")
            return
        trailing_empty_columns = total_columns - len(columns)
        suffix = "," * trailing_empty_columns
        for row in formatted:
            handle.write(",".join(row.tolist()))
            handle.write(suffix)
            handle.write("\n")
        return

    # General fallback for a cache that has evicted/reordered historical slots.
    for row in block:
        fields = [""] * total_columns
        for column, value in zip(columns, row):
            fields[column] = f"{float(value):.9g}"
        handle.write(",".join(fields))
        handle.write("\n")


def _slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:120].rstrip("-") or "prompt-action"


def _video_token_ids(frame_ids):
    counts = {}
    token_ids = []
    for frame_id in frame_ids.cpu().tolist():
        frame_id = int(frame_id)
        spatial_index = counts.get(frame_id, 0)
        token_ids.append([frame_id, spatial_index])
        counts[frame_id] = spatial_index + 1
    return token_ids


def _region_token_ids(frame_ids):
    """Return stable ``[frame_id, slot_within_frame]`` token identities."""
    return _video_token_ids(frame_ids)


def _ordered_union(sequences):
    seen = set()
    output = []
    for sequence in sequences:
        for token in sequence:
            token_tuple = tuple(token)
            if token_tuple not in seen:
                seen.add(token_tuple)
                output.append(token)
    return output


def _cumulative_boundaries(lengths):
    boundaries = []
    total = 0
    for length in lengths:
        total += int(length)
        boundaries.append(total)
    return boundaries[:-1]


def _history_boundaries(chunk_metadata, history_token_ids, token_field="key_token_ids"):
    lookup = {tuple(token): index for index, token in enumerate(history_token_ids)}
    boundaries = []
    for metadata in chunk_metadata[:-1]:
        if token_field in {"VV", "VA"}:
            visible = metadata["region_token_ids"][token_field]
        else:
            visible = metadata[token_field]
        if visible:
            boundaries.append(max(lookup[tuple(token)] for token in visible) + 1)
    return boundaries
