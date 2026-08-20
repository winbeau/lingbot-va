# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Shared top-p sparsity calculation for attention analysis."""

from __future__ import annotations

import numpy as np

TOP_P = 0.9
TOP_P_TOLERANCE = 1e-12


def top_p_sparsity_rows(
    attention: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    top_p: float = TOP_P,
) -> np.ndarray:
    """Return ``1 - k/N`` for every attention row.

    The last dimension contains key tokens. Each row is renormalized over its
    valid keys before finding the minimum ``k`` whose cumulative mass reaches
    ``top_p``. Leading dimensions are preserved in the returned array.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(
            f"attention must have at least two dimensions, got {values.shape}"
        )

    if valid_mask is None:
        valid = np.ones(values.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError(
                f"valid mask shape {valid.shape} does not match attention "
                f"shape {values.shape}"
            )

    valid_tokens = valid.sum(axis=-1)
    if np.any(valid_tokens == 0):
        index = tuple(int(value) for value in np.argwhere(valid_tokens == 0)[0])
        raise ValueError(f"attention row {index} has no available key tokens")

    present = values[valid]
    if np.any(~np.isfinite(present)):
        raise ValueError("attention contains non-finite values")
    if np.any(present < 0):
        raise ValueError("attention contains negative values")

    values = np.where(valid, values, 0.0)
    row_mass = values.sum(axis=-1)
    if np.any(row_mass <= 0):
        index = tuple(int(value) for value in np.argwhere(row_mass <= 0)[0])
        raise ValueError(f"attention row {index} has zero attention mass")

    probabilities = values / row_mass[..., None]
    probabilities.sort(axis=-1)
    cumulative = np.cumsum(probabilities[..., ::-1], axis=-1)
    tokens_needed = np.argmax(
        cumulative >= top_p - TOP_P_TOLERANCE,
        axis=-1,
    ) + 1
    return 1.0 - tokens_needed / valid_tokens
