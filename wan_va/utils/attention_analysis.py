# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# The streaming softmax extraction follows Pyramid-Forcing's
# wan/modules/attention/capture.py (Apache-2.0).
import json
import re
import shutil
from pathlib import Path

import torch


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

    Query and key use [L, H, D]. The stable softmax denominator includes every
    visible key token, while the returned [H, selected Q, selected K] tensor
    materializes only the requested video-to-video slice.
    """
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError(f"query/key must be [L,H,D], got {query.shape}/{key.shape}")
    if query_mask.ndim != 1 or key_slice_mask.ndim != 1:
        raise ValueError("query_mask and key_slice_mask must be one-dimensional")
    if query.shape[0] != query_mask.numel() or key.shape[0] != key_slice_mask.numel():
        raise ValueError("Token and mask lengths do not match")
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    query_mask = query_mask.to(query.device, dtype=torch.bool)
    key_slice_mask = key_slice_mask.to(key.device, dtype=torch.bool)
    selected_query = query[query_mask].float().transpose(0, 1)
    all_key = key.float().transpose(0, 1)
    selected_key = key[key_slice_mask].float().transpose(0, 1)
    if selected_query.shape[1] == 0 or selected_key.shape[1] == 0:
        raise ValueError("The requested query/key slice is empty")

    row_max = torch.full(
        selected_query.shape[:2],
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    row_sumexp = torch.zeros_like(row_max)

    # Pass 1: exact denominator over all visible video and action KV tokens.
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

    # Pass 2: materialize current-video Q x visible-video-history K only.
    selected_scores = torch.matmul(selected_query, selected_key.transpose(-2, -1))
    selected_scores.mul_(softmax_scale)
    return torch.exp(selected_scores - row_max.unsqueeze(-1)) / row_sumexp.clamp_min(
        1e-12
    ).unsqueeze(-1)


class JointAttentionRecorder:
    """Write one concatenated current-Q/history-K CSV per layer and head."""

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
    ):
        self.output_root = Path(output_dir)
        self.output_dir = self.output_root / _slugify(experiment_name)
        self.layers = set(layers)
        self.summary_layers = sorted(
            set(self.layers if summary_layers is None else summary_layers)
        )
        self.batch_index = batch_index
        self.chunk_tokens = chunk_tokens
        self.blocks_root = self.output_dir / ".attention-blocks"
        self.blocks_dir = self.blocks_root / f"worker_{worker_id:02d}"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.write_summary = write_summary
        self.step_indices = (
            None if step_indices is None else sorted(set(int(step) for step in step_indices))
        )
        if self.step_indices is not None and any(step < 0 for step in self.step_indices):
            raise ValueError("Attention step indices must be non-negative")
        if self.step_indices is None:
            self.block_files = {layer: [] for layer in self.layers}
            self.chunk_metadata = []
        else:
            self.block_files = {
                step: {layer: [] for layer in self.layers}
                for step in self.step_indices
            }
            self.chunk_metadata = {step: [] for step in self.step_indices}

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
        if not torch.any(visible_video_history):
            raise ValueError("Video query has no visible video-history key")

        batch_index = min(self.batch_index, query.shape[0] - 1)
        probabilities = compute_token_probability_slice(
            query=query[batch_index],
            key=key[batch_index],
            query_mask=current_video_query,
            key_slice_mask=visible_video_history,
            chunk_tokens=self.chunk_tokens,
        ).cpu()

        chunk_index = len(block_files[layer_index])
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

        query_ids = _video_token_ids(query_frame_ids[current_video_query])
        key_ids = _video_token_ids(key_frame_ids[visible_video_history])
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
            "query_token_ids": query_ids,
            "key_token_ids": key_ids,
            "query_tokens": len(query_ids),
            "visible_video_history_tokens": len(key_ids),
            "all_key_tokens_in_softmax": int(key.shape[1]),
            "query_frame_start": min(token[0] for token in query_ids),
            "query_frame_end": max(token[0] for token in query_ids) + 1,
            "cache_name": cache_name,
        }
        if chunk_index == len(chunk_metadata):
            chunk_metadata.append(metadata)
        else:
            expected = chunk_metadata[chunk_index]
            for field in (
                "query_token_ids",
                "key_token_ids",
                "all_key_tokens_in_softmax",
            ):
                if metadata[field] != expected[field]:
                    raise ValueError(
                        f"Layer {layer_index} chunk {chunk_index} has inconsistent {field}"
                    )

    def save(self):
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
        common_summary = {
            "semantics": (
                "rows concatenate each chunk's current-video Q; columns are "
                "chronological video-history K; softmax denominators include "
                "all KV modalities visible at that chunk"
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
        if self.step_indices is None:
            summary = {
                **common_summary,
                "csv_pattern": "lXXhXX.csv",
                **step_summaries[0],
            }
        else:
            summary = {
                **common_summary,
                "csv_pattern": "step_XXX/lXXhXX.csv",
                "requested_step_indices": self.step_indices,
                "steps": step_summaries,
            }

        summary_path = self.output_dir / "summary.json"
        if self.write_summary:
            summary_path.write_text(json.dumps(summary, indent=2))
        shutil.rmtree(self.blocks_dir)
        try:
            self.blocks_root.rmdir()
        except OSError:
            pass
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

        step_indices = {metadata["step_index"] for metadata in chunk_metadata}
        scheduler_timesteps = {
            metadata["scheduler_timestep"] for metadata in chunk_metadata
        }
        if len(step_indices) != 1 or len(scheduler_timesteps) != 1:
            raise RuntimeError("Attention chunks have inconsistent timestep metadata")

        row_token_ids = [
            token
            for metadata in chunk_metadata
            for token in metadata["query_token_ids"]
        ]
        history_token_ids = _ordered_union(
            metadata["key_token_ids"] for metadata in chunk_metadata
        )
        logical_shape = [len(row_token_ids), len(history_token_ids)]
        num_heads = self._num_heads(block_files, layer_indices[0])
        output_dir.mkdir(parents=True, exist_ok=True)

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
            "step_index": next(iter(step_indices)),
            "scheduler_timestep": next(iter(scheduler_timesteps)),
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
            "chunks": [
                {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"query_token_ids", "key_token_ids"}
                }
                for metadata in chunk_metadata
            ],
        }

    def _num_heads(self, block_files, layer_index):
        payload = torch.load(
            block_files[layer_index][0], map_location="cpu", weights_only=True
        )
        return int(payload["attention"].shape[0])

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

    # The default non-evicting cache yields a chronological prefix. This path
    # avoids constructing a dense all-history matrix and writes future columns
    # as empty CSV fields.
    if columns == list(range(len(columns))):
        formatted = np.char.mod("%.9g", block)
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


def _history_boundaries(chunk_metadata, history_token_ids):
    lookup = {tuple(token): index for index, token in enumerate(history_token_ids)}
    boundaries = []
    for metadata in chunk_metadata[:-1]:
        visible = metadata["key_token_ids"]
        if visible:
            boundaries.append(max(lookup[tuple(token)] for token in visible) + 1)
    return boundaries
