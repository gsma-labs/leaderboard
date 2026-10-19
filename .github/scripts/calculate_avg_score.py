#!/usr/bin/env python3
"""Compute derived leaderboard columns locally; standalone publication is disabled.

Computes `average`, `benchmarks_completed`, and `rank` from benchmark scores.
Imported by the single guarded publisher before its atomic commit.
"""

from __future__ import annotations

import ast

import pandas as pd

from registry import get_benchmark_columns

DATASET_REPO = "GSMA/leaderboard"
BENCHMARK_COLUMNS = get_benchmark_columns()


def extract_score(value: object) -> float | None:
    """Extract the primary score from a string-encoded or list score value.

    Handles both new format (string "[score, stderr]") and legacy
    format (list/array [score, stderr, ...]).
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) >= 1:
                return float(parsed[0])
        except (ValueError, SyntaxError):
            return None
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError):
        return None


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average, benchmarks_completed, and rank for all rows.

    - average: arithmetic mean of available benchmark scores (0-1 scale)
    - benchmarks_completed: count of non-null benchmark columns
    - rank: 1-based rank by average (descending), ties get same rank
    """
    averages = []
    completed_counts = []

    for _, row in df.iterrows():
        scores = []
        n_completed = 0
        for col in BENCHMARK_COLUMNS:
            score = extract_score(row.get(col))
            if score is not None:
                scores.append(score)
                n_completed += 1

        model = row.get("model", "unknown")
        missing = [col for col in BENCHMARK_COLUMNS if extract_score(row.get(col)) is None]
        if missing:
            print(f"Info: {model} missing benchmarks: {', '.join(missing)} "
                  f"(averaging {n_completed}/{len(BENCHMARK_COLUMNS)})")

        if scores:
            averages.append(sum(scores) / len(scores))
        else:
            averages.append(None)
        completed_counts.append(n_completed)

    df["average"] = averages
    df["benchmarks_completed"] = completed_counts

    # Rank by average descending (None values get no rank)
    df["rank"] = (
        df["average"]
        .rank(ascending=False, method="min", na_option="bottom")
        .where(df["average"].notna())
        .astype("Int64")
    )

    return df


def main() -> None:
    """The atomic publisher owns all writes; this former second writer is disabled."""
    raise SystemExit('Deprecated: use sync_to_huggingface.py --manifest ... (or --dry-run).')


if __name__ == "__main__":
    main()
