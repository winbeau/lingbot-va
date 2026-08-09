#!/usr/bin/env python3
"""Compute per-layer top-p sparsity from video-to-video attention CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TOP_P = 0.9
ATTENTION_CSV = re.compile(r"^l(?P<layer>\d+)h(?P<head>\d+)\.csv$")


def _default_processes() -> int:
    return max(1, (os.cpu_count() or 1) * 3 // 4)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute layer sparsity as 1 - k/N, where k is the fewest available "
            "key tokens whose normalized attention reaches top-p=0.9."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing lXXhYY.csv attention files for one step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for <input-directory-name>.csv.",
    )
    parser.add_argument(
        "--processes",
        type=_positive_int,
        default=_default_processes(),
        help="CPU worker processes (default: three quarters of available CPUs).",
    )
    return parser.parse_args()


def _discover_layers(input_dir: Path) -> list[tuple[int, list[Path]]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    layer_files: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = ATTENTION_CSV.fullmatch(path.name)
        if match is None:
            continue
        layer_files[int(match.group("layer"))].append(
            (int(match.group("head")), path)
        )

    if not layer_files:
        raise FileNotFoundError(
            f"No lXXhYY.csv attention files found directly under {input_dir}"
        )

    return [
        (layer, [path for _, path in sorted(head_files)])
        for layer, head_files in sorted(layer_files.items())
    ]


def _head_density_sum(path: Path) -> tuple[float, int]:
    try:
        attention = np.genfromtxt(
            path,
            delimiter=",",
            dtype=np.float64,
            filling_values=np.nan,
            ndmin=2,
        )
    except Exception as error:
        raise ValueError(f"Failed to read {path}: {error}") from error

    valid = ~np.isnan(attention)
    valid_tokens = valid.sum(axis=1)
    if np.any(valid_tokens == 0):
        row = int(np.flatnonzero(valid_tokens == 0)[0])
        raise ValueError(f"{path}: query row {row} has no available key tokens")

    present = attention[valid]
    if np.any(~np.isfinite(present)):
        raise ValueError(f"{path}: attention contains non-finite values")
    if np.any(present < 0):
        raise ValueError(f"{path}: attention contains negative values")

    values = np.nan_to_num(attention, nan=0.0)
    row_mass = values.sum(axis=1)
    if np.any(row_mass <= 0):
        row = int(np.flatnonzero(row_mass <= 0)[0])
        raise ValueError(f"{path}: query row {row} has zero attention mass")

    # The recorder keeps the full multimodal softmax denominator while writing
    # only video keys. Renormalize each available video-to-video row so top-p is
    # measured over the video tokens represented by this CSV.
    probabilities = values / row_mass[:, None]
    probabilities.sort(axis=1)
    cumulative = np.cumsum(probabilities[:, ::-1], axis=1)
    tokens_needed = np.argmax(cumulative >= TOP_P - 1e-12, axis=1) + 1
    density = 1.0 - tokens_needed / valid_tokens
    return float(density.sum()), int(density.size)


def _layer_density(task: tuple[int, list[Path]]) -> tuple[int, float]:
    layer, paths = task
    density_sum = 0.0
    query_count = 0
    for path in paths:
        head_sum, head_queries = _head_density_sum(path)
        density_sum += head_sum
        query_count += head_queries
    return layer, density_sum / query_count


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    layer_tasks = _discover_layers(input_dir)
    worker_count = min(args.processes, len(layer_tasks))

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = sorted(executor.map(_layer_density, layer_tasks))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{input_dir.name}.csv"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "density"])
        for layer, density in results:
            writer.writerow([layer, f"{density:.9g}"])
    temporary_path.replace(output_path)
    print(f"Wrote {len(results)} layer densities to {output_path}")


if __name__ == "__main__":
    main()
