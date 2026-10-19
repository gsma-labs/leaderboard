#!/usr/bin/env python3
"""Exact local seven-benchmark contract for cards, validation and publication."""
from __future__ import annotations

# Task name -> authoritative GSMA/leaderboard column name.
BENCHMARK_HF_COLUMNS: dict[str, str] = {
    "teleqna": "teleqna",
    "oranbench": "oranbench",
    "srsranbench": "srsranbench",
    "telelogs": "telelogs",
    "telemath": "telemath",
    "three_gpp": "three_gpp",
    "teletables": "teletables",
}

# Column names emitted by older submission tooling -> current column name.
COLUMN_ALIASES: dict[str, str] = {
    "3gpp_tsg": "three_gpp",
}

# Task name -> config name in GSMA/ot-full and GSMA/ot-lite. This is a
# DIFFERENT namespace from the leaderboard columns above; they coincide for
# every benchmark except three_gpp, which is why the two were once conflated.
# Only listed where it differs from the task name.
BENCHMARK_DATASET_CONFIGS: dict[str, str] = {
    "three_gpp": "3gpp_tsg",
}

def get_benchmark_to_hf_map() -> dict[str, str]:
    """Return the exact seven tasks supported by this leaderboard.

    Validation must not change when an unrelated upstream registry changes.
    """
    return dict(BENCHMARK_HF_COLUMNS)


def get_required_columns() -> list[str]:
    """Return the full list of required parquet/leaderboard columns.

    Returns:
        ``["model", <hf_columns...>, "date"]``
    """
    hf_map = get_benchmark_to_hf_map()
    return ["model", *sorted(hf_map.values()), "date"]


def get_benchmark_columns() -> list[str]:
    """Return just the benchmark HF column names (no 'model'/'date')."""
    hf_map = get_benchmark_to_hf_map()
    return sorted(hf_map.values())


def get_dataset_config(benchmark: str) -> str:
    """Return the Full/Lite dataset config name for a benchmark task."""
    return BENCHMARK_DATASET_CONFIGS.get(benchmark, benchmark)


def rename_aliased_columns(df):
    """Rename legacy submission column names to their current equivalents.

    Submission parquets built by older tooling use ``3gpp_tsg`` where the
    leaderboard uses ``three_gpp``.  Returns a new DataFrame; the original is
    left untouched.
    """
    present = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns}
    return df.rename(columns=present) if present else df
