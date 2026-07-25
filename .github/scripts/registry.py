#!/usr/bin/env python3
"""Shared benchmark registry fetched from gsma-labs/evals.

Single source of truth for benchmark IDs and their HuggingFace column names.
All leaderboard scripts import from here instead of hardcoding benchmark lists.

The registry dynamically fetches ``__all__`` from the evals repo's
``_registry.py`` and validates it against a local mapping that translates
task names to HF column names (e.g. ``three_gpp`` -> ``3gpp_tsg``).
"""

from __future__ import annotations

import ast
import time
import urllib.request

REGISTRY_URL = (
    "https://raw.githubusercontent.com/gsma-labs/evals/main/src/evals/_registry.py"
)

# Task name (from evals __all__) -> HuggingFace column name.
# These MUST match the live column names in the GSMA/leaderboard dataset.
# A benchmark in the evals registry with no entry here is simply not on the
# leaderboard yet; it is ignored rather than treated as an error, so adding an
# eval upstream can never break submissions.
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

# Task name -> config name in the GSMA/open_telco source dataset. This is a
# DIFFERENT namespace from the leaderboard columns above; they coincide for
# every benchmark except three_gpp, which is why the two were once conflated.
# Only listed where it differs from the task name.
BENCHMARK_DATASET_CONFIGS: dict[str, str] = {
    "three_gpp": "3gpp_tsg",
}

_FETCH_TIMEOUT_SECONDS = 10

_cached_benchmarks: list[str] | None = None


def _parse_all_from_source(source: str) -> list[str]:
    """Extract ``__all__`` list from Python source code via AST parsing."""
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != "__all__":
                continue
            result = ast.literal_eval(node.value)
            if not isinstance(result, list) or not result:
                raise ValueError("__all__ must be a non-empty list")
            if not all(isinstance(item, str) for item in result):
                raise ValueError("__all__ must contain only strings")
            return result
    raise ValueError("No __all__ found in registry source")


def fetch_registry_benchmarks() -> list[str]:
    """Fetch benchmark IDs from gsma-labs/evals ``_registry.py``.

    Returns:
        Sorted list of benchmark task names from the evals ``__all__``.

    Falls back to local ``BENCHMARK_HF_COLUMNS`` keys when the fetch fails.
    Results are cached for the lifetime of the process.
    """
    global _cached_benchmarks
    if _cached_benchmarks is not None:
        return list(_cached_benchmarks)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(REGISTRY_URL)
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
                source = resp.read().decode()
            benchmarks = _parse_all_from_source(source)
            print(f"Registry: fetched {len(benchmarks)} benchmarks from gsma-labs/evals")
            _cached_benchmarks = sorted(benchmarks)
            return list(_cached_benchmarks)
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(
                f"Warning: Failed to fetch registry after {max_retries} attempts, "
                f"using local fallback: {type(exc).__name__}"
            )
            _cached_benchmarks = sorted(BENCHMARK_HF_COLUMNS.keys())
            return list(_cached_benchmarks)

    # Unreachable, but satisfies type checker
    _cached_benchmarks = sorted(BENCHMARK_HF_COLUMNS.keys())
    return list(_cached_benchmarks)


def get_benchmark_to_hf_map() -> dict[str, str]:
    """Return task_name -> HF column mapping for benchmarks on the leaderboard.

    Benchmarks present in the evals registry but absent from
    ``BENCHMARK_HF_COLUMNS`` are not on the leaderboard yet and are skipped
    with a warning.  Raising here instead would break every submission the
    moment a new eval lands upstream.
    """
    registry = fetch_registry_benchmarks()
    unmapped = [b for b in registry if b not in BENCHMARK_HF_COLUMNS]
    if unmapped:
        print(
            f"Note: registry benchmarks not yet on the leaderboard: {unmapped}. "
            f"Add them to BENCHMARK_HF_COLUMNS in registry.py to include them."
        )
    return {b: BENCHMARK_HF_COLUMNS[b] for b in registry if b in BENCHMARK_HF_COLUMNS}


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
    """Return the GSMA/open_telco config name for a benchmark task name."""
    return BENCHMARK_DATASET_CONFIGS.get(benchmark, benchmark)


def rename_aliased_columns(df):
    """Rename legacy submission column names to their current equivalents.

    Submission parquets built by older tooling use ``3gpp_tsg`` where the
    leaderboard uses ``three_gpp``.  Returns a new DataFrame; the original is
    left untouched.
    """
    present = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns}
    return df.rename(columns=present) if present else df
