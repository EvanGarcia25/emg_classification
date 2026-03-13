# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize CER vs. electrode channel count from a Hydra multirun directory."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


@dataclass
class SweepRow:
    run_dir: Path
    electrode_channels: int
    val_cer: float | None
    test_cer: float | None


def _extract_results_dict(log_text: str) -> dict[str, Any] | None:
    """Extract the final printed results dict from train.py logs."""
    anchor = "{'val_metrics':"
    start_idx = log_text.rfind(anchor)
    if start_idx == -1:
        return None

    depth = 0
    end_idx: int | None = None
    for idx, char in enumerate(log_text[start_idx:], start=start_idx):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx + 1
                break

    if end_idx is None:
        return None

    snippet = log_text[start_idx:end_idx]
    try:
        parsed = ast.literal_eval(snippet)
    except (ValueError, SyntaxError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_cer(metrics_list: Any, key: str) -> float | None:
    if not isinstance(metrics_list, list) or len(metrics_list) == 0:
        return None
    metrics = metrics_list[0]
    if not isinstance(metrics, dict) or key not in metrics:
        return None
    return float(metrics[key])


def _load_electrode_channels(run_dir: Path) -> int | None:
    overrides_path = run_dir / "hydra_configs" / "overrides.yaml"
    if overrides_path.exists():
        for line in overrides_path.read_text().splitlines():
            override = line.strip().lstrip("-").strip()
            if override.startswith("electrode_channels="):
                return int(override.split("=", maxsplit=1)[1])

    config_path = run_dir / "hydra_configs" / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        value = cfg.get("electrode_channels")
        if value is not None:
            return int(value)

    return None


def _parse_run(run_dir: Path) -> SweepRow | None:
    electrode_channels = _load_electrode_channels(run_dir)
    if electrode_channels is None:
        return None

    log_files = sorted(run_dir.glob("*.log"))
    if not log_files:
        return SweepRow(run_dir, electrode_channels, None, None)

    log_text = log_files[0].read_text()
    results = _extract_results_dict(log_text)
    if results is None:
        return SweepRow(run_dir, electrode_channels, None, None)

    val_cer = _extract_cer(results.get("val_metrics"), "val/CER")
    test_cer = _extract_cer(results.get("test_metrics"), "test/CER")
    return SweepRow(run_dir, electrode_channels, val_cer, test_cer)


def _first_below_threshold(
    rows: list[SweepRow],
    metric: str,
    threshold: float,
) -> int | None:
    valid_rows = [
        row
        for row in rows
        if getattr(row, metric) is not None
        and getattr(row, metric) < threshold
    ]
    if not valid_rows:
        return None
    return min(row.electrode_channels for row in valid_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Path to Hydra multirun directory containing job*/ subdirectories.",
    )
    parser.add_argument(
        "--target-cer",
        type=float,
        default=20.0,
        help="CER threshold used to report the minimum required channels.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. "
            "Defaults to <sweep_dir>/channel_sweep_summary.csv"
        ),
    )
    args = parser.parse_args()

    run_dirs = sorted(
        [
            path
            for path in args.sweep_dir.iterdir()
            if path.is_dir() and path.name.startswith("job")
        ]
    )

    rows = [
        row
        for row in (_parse_run(run_dir) for run_dir in run_dirs)
        if row is not None
    ]
    rows.sort(key=lambda row: row.electrode_channels)

    if not rows:
        raise RuntimeError(
            "No sweep jobs with electrode_channels were found under "
            f"{args.sweep_dir}"
        )

    output_csv = args.output_csv or (args.sweep_dir / "channel_sweep_summary.csv")
    with output_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["electrode_channels", "val_cer", "test_cer", "run_dir"])
        for row in rows:
            writer.writerow(
                [
                    row.electrode_channels,
                    row.val_cer,
                    row.test_cer,
                    str(row.run_dir),
                ]
            )

    print("Channel sweep summary:")
    print("electrode_channels,val_cer,test_cer")
    for row in rows:
        print(f"{row.electrode_channels},{row.val_cer},{row.test_cer}")

    min_val_channels = _first_below_threshold(rows, "val_cer", args.target_cer)
    min_test_channels = _first_below_threshold(rows, "test_cer", args.target_cer)

    if min_val_channels is None:
        print(f"No runs achieved val/CER < {args.target_cer:.2f}")
    else:
        print(
            "Minimum channels for val/CER < "
            f"{args.target_cer:.2f}: {min_val_channels}"
        )

    if min_test_channels is None:
        print(f"No runs achieved test/CER < {args.target_cer:.2f}")
    else:
        print(
            "Minimum channels for test/CER < "
            f"{args.target_cer:.2f}: {min_test_channels}"
        )

    channels = np.array([row.electrode_channels for row in rows], dtype=float)
    test_cer = np.array(
        [
            row.test_cer
            for row in rows
            if row.test_cer is not None
        ],
        dtype=float,
    )
    test_channels = np.array(
        [
            row.electrode_channels
            for row in rows
            if row.test_cer is not None
        ],
        dtype=float,
    )
    if len(test_cer) >= 2:
        corr = np.corrcoef(test_channels, test_cer)[0, 1]
        print(f"Pearson corr(channels, test/CER): {corr:.4f}")

    if len(channels) >= 2:
        val_cer = np.array(
            [
                row.val_cer
                for row in rows
                if row.val_cer is not None
            ],
            dtype=float,
        )
        val_channels = np.array(
            [
                row.electrode_channels
                for row in rows
                if row.val_cer is not None
            ],
            dtype=float,
        )
        if len(val_cer) >= 2:
            corr = np.corrcoef(val_channels, val_cer)[0, 1]
            print(f"Pearson corr(channels, val/CER): {corr:.4f}")

    print(f"Wrote CSV summary to: {output_csv}")


if __name__ == "__main__":
    main()
