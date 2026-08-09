# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# The chunked softmax aggregation is adapted from Pyramid-Forcing's
# wan/modules/attention/capture.py (Apache-2.0).
import json
from collections import defaultdict
from pathlib import Path

import torch


_MODALITY_NAMES = {0: "video", 1: "action"}
_MODALITY_SHORT_NAMES = {0: "v", 1: "a"}


def compute_frame_attention_metrics_single_sequence(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    q_frame_ids: torch.Tensor,
    k_frame_ids: torch.Tensor,
    softmax_scale: float | None = None,
    chunk_tokens: int = 4096,
    capture_mode: str = "prob_mass",
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Compute grouped attention statistics without materializing full QK."""
    if q_seq.ndim != 2 or k_seq.ndim != 2:
        raise ValueError(f"q_seq/k_seq must be 2D, got {q_seq.shape}/{k_seq.shape}")
    if q_frame_ids.ndim != 1 or k_frame_ids.ndim != 1:
        raise ValueError(
            "q_frame_ids/k_frame_ids must be 1D, "
            f"got {q_frame_ids.shape}/{k_frame_ids.shape}"
        )
    if (
        q_seq.shape[0] != q_frame_ids.shape[0]
        or k_seq.shape[0] != k_frame_ids.shape[0]
    ):
        raise ValueError("Token length and frame-id length mismatch")
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")

    if softmax_scale is None:
        softmax_scale = q_seq.shape[-1] ** -0.5
    if capture_mode not in {"logits_mean", "prob_mass", "both"}:
        raise ValueError(f"Unsupported capture_mode={capture_mode}")

    need_logits = capture_mode in {"logits_mean", "both"}
    need_prob = capture_mode in {"prob_mass", "both"}
    q_unique = torch.unique(q_frame_ids.to(torch.long), sorted=True)
    k_unique = torch.unique(k_frame_ids.to(torch.long), sorted=True)
    nqf = int(q_unique.numel())
    nkf = int(k_unique.numel())

    device = q_seq.device
    q_seq_f = q_seq.float()
    k_seq_f = k_seq.float()
    q_ids = q_frame_ids.to(device=device, dtype=torch.long)
    k_ids = k_frame_ids.to(device=device, dtype=torch.long)
    q_masks = [(q_ids == int(qf.item())) for qf in q_unique]

    logits_mean = None
    prob_mass = None
    if need_logits:
        logits_sum = torch.zeros(nqf, nkf, dtype=torch.float32, device=device)
        logits_count = torch.zeros_like(logits_sum)
        for k_start in range(0, k_seq_f.shape[0], chunk_tokens):
            k_end = min(k_start + chunk_tokens, k_seq_f.shape[0])
            k_chunk = k_seq_f[k_start:k_end]
            k_ids_chunk = k_ids[k_start:k_end]
            scores = torch.matmul(q_seq_f, k_chunk.transpose(0, 1)) * softmax_scale
            for q_index, q_mask in enumerate(q_masks):
                if not torch.any(q_mask):
                    continue
                q_scores = scores[q_mask]
                for k_index, key_group in enumerate(k_unique):
                    k_mask = k_ids_chunk == int(key_group.item())
                    if not torch.any(k_mask):
                        continue
                    values = q_scores[:, k_mask]
                    logits_sum[q_index, k_index] += values.sum()
                    logits_count[q_index, k_index] += float(values.numel())
        logits_mean = torch.where(
            logits_count > 0,
            logits_sum / logits_count.clamp_min(1.0),
            torch.zeros_like(logits_sum),
        )

    if need_prob:
        query_length = q_seq_f.shape[0]
        row_max = torch.full(
            (query_length,), float("-inf"), dtype=torch.float32, device=device
        )
        row_sumexp = torch.zeros(query_length, dtype=torch.float32, device=device)

        # Pass 1: stable streaming softmax denominator over K chunks.
        for k_start in range(0, k_seq_f.shape[0], chunk_tokens):
            k_end = min(k_start + chunk_tokens, k_seq_f.shape[0])
            k_chunk = k_seq_f[k_start:k_end]
            scores = torch.matmul(q_seq_f, k_chunk.transpose(0, 1)) * softmax_scale
            chunk_max = scores.max(dim=-1).values
            new_max = torch.maximum(row_max, chunk_max)
            row_sumexp = (
                torch.exp(row_max - new_max) * row_sumexp
                + torch.exp(scores - new_max.unsqueeze(-1)).sum(dim=-1)
            )
            row_max = new_max

        row_valid = row_sumexp > 0
        denominator = row_sumexp.clamp_min(1e-12)
        probability_sum = torch.zeros(
            nqf, nkf, dtype=torch.float32, device=device
        )
        valid_query_count = torch.zeros(nqf, dtype=torch.float32, device=device)
        for q_index, q_mask in enumerate(q_masks):
            valid_query_count[q_index] = float((q_mask & row_valid).sum().item())

        # Pass 2: recompute one Q x K_chunk block and aggregate immediately.
        for k_start in range(0, k_seq_f.shape[0], chunk_tokens):
            k_end = min(k_start + chunk_tokens, k_seq_f.shape[0])
            k_chunk = k_seq_f[k_start:k_end]
            k_ids_chunk = k_ids[k_start:k_end]
            scores = torch.matmul(q_seq_f, k_chunk.transpose(0, 1)) * softmax_scale
            probabilities = (
                torch.exp(scores - row_max.unsqueeze(-1))
                / denominator.unsqueeze(-1)
            )
            probabilities = torch.where(
                row_valid.unsqueeze(-1),
                probabilities,
                torch.zeros_like(probabilities),
            )
            for q_index, q_mask in enumerate(q_masks):
                q_mask_valid = q_mask & row_valid
                if not torch.any(q_mask_valid):
                    continue
                q_probabilities = probabilities[q_mask_valid]
                for k_index, key_group in enumerate(k_unique):
                    k_mask = k_ids_chunk == int(key_group.item())
                    if not torch.any(k_mask):
                        continue
                    probability_sum[q_index, k_index] += (
                        q_probabilities[:, k_mask].sum(dim=-1).sum()
                    )

        probability_denominator = valid_query_count.clamp_min(1.0).unsqueeze(1)
        prob_mass = torch.where(
            valid_query_count.unsqueeze(1) > 0,
            probability_sum / probability_denominator,
            torch.zeros_like(probability_sum),
        )

    return logits_mean, prob_mass, q_unique, k_unique


def _build_group_ids(token_types, frame_ids):
    pairs = torch.stack((token_types.to(torch.long), frame_ids.to(torch.long)), dim=1)
    unique_pairs, inverse = torch.unique(
        pairs, sorted=True, return_inverse=True, dim=0
    )
    return inverse, unique_pairs


def _group_labels(unique_pairs):
    return [
        f"{_MODALITY_SHORT_NAMES[int(modality)]}:f{int(frame_id):03d}"
        for modality, frame_id in unique_pairs.tolist()
    ]


class JointAttentionRecorder:
    """Capture selected self-attention calls as tiny modality/frame summaries."""

    def __init__(self, output_dir, layers, batch_index=0, chunk_tokens=512):
        self.output_dir = Path(output_dir)
        self.layers = set(layers)
        self.batch_index = batch_index
        self.chunk_tokens = chunk_tokens
        self.records = []
        self._layer_capture_counts = defaultdict(int)

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
    ):
        if update_cache != 1 or layer_index not in self.layers:
            return

        batch_index = min(self.batch_index, query.shape[0] - 1)
        query = query[batch_index]
        key = key[batch_index]
        device = query.device
        query_token_types = query_token_types.to(device)
        key_token_types = key_token_types.to(device)
        query_frame_ids = query_frame_ids.to(device)
        key_frame_ids = key_frame_ids.to(device)
        query_group_ids, query_groups = _build_group_ids(
            query_token_types, query_frame_ids
        )
        key_group_ids, key_groups = _build_group_ids(key_token_types, key_frame_ids)

        head_probability_mass = []
        for head_index in range(query.shape[1]):
            _, probability_mass, _, _ = compute_frame_attention_metrics_single_sequence(
                q_seq=query[:, head_index],
                k_seq=key[:, head_index],
                q_frame_ids=query_group_ids,
                k_frame_ids=key_group_ids,
                chunk_tokens=self.chunk_tokens,
                capture_mode="prob_mass",
            )
            head_probability_mass.append(probability_mass.cpu())
        head_probability_mass = torch.stack(head_probability_mass)
        mean_probability_mass = head_probability_mass.mean(dim=0)

        query_group_counts = torch.bincount(
            query_group_ids.cpu(), minlength=query_groups.shape[0]
        ).float()
        key_group_counts = torch.bincount(
            key_group_ids.cpu(), minlength=key_groups.shape[0]
        ).float()
        query_weights = query_group_counts / query_group_counts.sum().clamp_min(1.0)
        mean_key_group_mass = (mean_probability_mass * query_weights[:, None]).sum(
            dim=0
        )
        per_head_key_group_mass = (
            head_probability_mass * query_weights[None, :, None]
        ).sum(dim=1)

        modality_mass = {}
        modality_density = {}
        per_head_modality_mass = {}
        key_token_counts = {}
        for modality_id, modality_name in _MODALITY_NAMES.items():
            key_group_mask = key_groups[:, 0].cpu() == modality_id
            key_count = int(key_group_counts[key_group_mask].sum().item())
            key_token_counts[modality_name] = key_count
            if key_count == 0:
                continue
            modality_mass[modality_name] = float(
                mean_key_group_mass[key_group_mask].sum().item()
            )
            modality_density[modality_name] = (
                modality_mass[modality_name] / key_count
            )
            per_head_modality_mass[modality_name] = (
                per_head_key_group_mass[:, key_group_mask].sum(dim=1).tolist()
            )

        query_modalities = torch.unique(query_token_types)
        if query_modalities.numel() != 1:
            raise ValueError(
                "Inference capture expects one query modality per call, got "
                f"{query_modalities.tolist()}"
            )
        query_modality = _MODALITY_NAMES[int(query_modalities.item())]
        capture_index = self._layer_capture_counts[layer_index]
        self._layer_capture_counts[layer_index] += 1
        self.records.append(
            {
                "layer": layer_index,
                "capture_index": capture_index,
                "cache_name": cache_name,
                "query_modality": query_modality,
                "query_tokens": int(query.shape[0]),
                "key_tokens": int(key.shape[0]),
                "num_heads": int(query.shape[1]),
                "key_token_counts": key_token_counts,
                "attention_mass_by_key_modality": modality_mass,
                "attention_density_by_key_modality": modality_density,
                "attention_mass_by_head_and_key_modality": per_head_modality_mass,
                "query_frame_labels": _group_labels(query_groups),
                "key_frame_labels": _group_labels(key_groups),
                "query_group_token_counts": query_group_counts.tolist(),
                "key_group_token_counts": key_group_counts.tolist(),
                "frame_attention_mass": mean_probability_mass.tolist(),
                "frame_attention_mass_by_head": head_probability_mass.tolist(),
            }
        )

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "source_method": (
                        "Pyramid-Forcing two-pass chunked softmax probability mass"
                    ),
                    "layers": sorted(self.layers),
                    "chunk_tokens": self.chunk_tokens,
                    "capture_rule": "self-attention calls with update_cache=1",
                    "records": self.records,
                },
                indent=2,
            )
        )
        self._save_plots()
        return summary_path

    def _save_plots(self):
        for record in self.records:
            _save_heatmap(
                record["frame_attention_mass"],
                record["query_frame_labels"],
                record["key_frame_labels"],
                self.output_dir
                / (
                    f"layer_{record['layer']:02d}_{record['query_modality']}_"
                    f"capture_{record['capture_index']:02d}.png"
                ),
                f"Layer {record['layer']} {record['query_modality']} attention mass",
            )

        for metric_key, file_name, title in (
            (
                "attention_mass_by_key_modality",
                "modality_attention_mass.png",
                "Joint attention mass",
            ),
            (
                "attention_density_by_key_modality",
                "modality_attention_density.png",
                "Joint attention density per key token",
            ),
        ):
            matrix = [[None, None], [None, None]]
            for query_id, query_name in _MODALITY_NAMES.items():
                for key_id, key_name in _MODALITY_NAMES.items():
                    values = [
                        record[metric_key][key_name]
                        for record in self.records
                        if record["query_modality"] == query_name
                        and key_name in record[metric_key]
                    ]
                    if values:
                        matrix[query_id][key_id] = sum(values) / len(values)
            _save_heatmap(
                matrix,
                [_MODALITY_NAMES[index] for index in range(2)],
                [_MODALITY_NAMES[index] for index in range(2)],
                self.output_dir / file_name,
                title,
            )


def _save_heatmap(matrix, row_labels, column_labels, path, title):
    from PIL import Image, ImageDraw

    valid_values = [
        float(value)
        for row in matrix
        for value in row
        if value is not None
    ]
    if not valid_values:
        return
    maximum = max(valid_values)
    cell_width = 92
    cell_height = 54
    left = max(130, max((len(label) for label in row_labels), default=0) * 8 + 18)
    top = 76
    width = left + cell_width * len(column_labels) + 24
    height = top + cell_height * len(row_labels) + 32
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), title, fill="black")

    for column, label in enumerate(column_labels):
        draw.text((left + column * cell_width + 6, 46), label, fill="black")
    for row, label in enumerate(row_labels):
        draw.text((12, top + row * cell_height + 18), label, fill="black")
        for column, value in enumerate(matrix[row]):
            x0 = left + column * cell_width
            y0 = top + row * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            if value is None:
                color = (235, 235, 235)
                text = "n/a"
            else:
                normalized = 0.0 if maximum <= 0 else float(value) / maximum
                color = (
                    int(247 - 210 * normalized),
                    int(250 - 105 * normalized),
                    int(255 - 35 * normalized),
                )
                text = f"{float(value):.4g}"
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(80, 80, 80))
            draw.text((x0 + 8, y0 + 18), text, fill="black")
    image.save(path)
