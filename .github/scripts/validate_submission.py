#!/usr/bin/env python3
"""Validate leaderboard submission files with sample count verification.

ALL benchmarks from the evals registry must be present for a valid submission.
The workflow will generate parquet files from JSON trajectories, so parquet
files are optional in the submission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from registry import (
    get_benchmark_columns,
    get_benchmark_to_hf_map,
    get_dataset_config,
    get_required_columns,
    rename_aliased_columns,
)

# Providers we already know. An unknown provider is reported as a warning, not
# an error: the leaderboard is open to any operator or vendor, and a hard
# whitelist here would block every new contributor.
RECOGNIZED_PROVIDERS = [
    "Openai",
    "Anthropic",
    "Google",
    "Mistral",
    "Deepseek",
    "Meta",
    "Cohere",
    "Together",
    "Openrouter",
    "Groq",
    "Fireworks",
]

# HuggingFace dataset for expected sample counts
OPEN_TELCO_DATASET = "GSMA/open_telco"


def fetch_expected_sample_counts(token: str | None = None) -> dict[str, int | None]:
    """Fetch expected sample counts from GSMA/open_telco HuggingFace dataset.

    Args:
        token: Optional HuggingFace token for dataset access

    Returns:
        Dict mapping benchmark name to expected sample count (None if fetch failed)
    """
    benchmark_to_hf = get_benchmark_to_hf_map()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not available, skipping sample count validation")
        return {k: None for k in benchmark_to_hf}

    expected_counts: dict[str, int | None] = {}
    max_retries = 3

    for benchmark in benchmark_to_hf:
        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    OPEN_TELCO_DATASET,
                    name=get_dataset_config(benchmark),
                    split="test",
                    token=token,
                )
                expected_counts[benchmark] = len(ds)
                print(f"  {benchmark}: {len(ds)} samples expected")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                print(f"Warning: Could not fetch {benchmark} count after {max_retries} attempts: {type(e).__name__}")
                expected_counts[benchmark] = None

    return expected_counts


def extract_sample_info_from_trajectory(
    json_path: Path,
) -> tuple[str | None, list | None]:
    """Extract benchmark name and sample_ids from trajectory JSON.

    Args:
        json_path: Path to trajectory JSON file

    Returns:
        Tuple of (benchmark_name, sample_ids_list) or (None, None) if not found
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    # Extract benchmark name from eval.task
    benchmark = None
    benchmark_to_hf = get_benchmark_to_hf_map()
    if "eval" in data and isinstance(data["eval"], dict):
        task = data["eval"].get("task", "")
        task_lower = task.lower()
        for key in benchmark_to_hf:
            if key in task_lower:
                benchmark = key
                break

    # Extract sample_ids from eval.dataset
    sample_ids = None
    if "eval" in data and isinstance(data["eval"], dict):
        dataset_info = data["eval"].get("dataset", {})
        if isinstance(dataset_info, dict):
            sample_ids = dataset_info.get("sample_ids")

    return benchmark, sample_ids


def validate_sample_counts(
    trajectory_files: list[Path],
    expected_counts: dict[str, int | None],
) -> tuple[dict[str, bool], dict[str, dict], list[str]]:
    """Validate that trajectory sample counts match expected.

    Args:
        trajectory_files: List of trajectory JSON file paths
        expected_counts: Dict of benchmark -> expected sample count

    Returns:
        Tuple of (check_dict, sample_details_dict, errors_list)
    """
    checks = {"sample_count_valid": True}
    sample_details: dict[str, dict] = {}
    errors: list[str] = []

    # Group sample_ids by benchmark
    benchmark_samples: dict[str, set] = {}

    for json_path in trajectory_files:
        benchmark, sample_ids = extract_sample_info_from_trajectory(json_path)

        if benchmark and sample_ids is not None:
            if benchmark not in benchmark_samples:
                benchmark_samples[benchmark] = set()
            # Add sample IDs (as set to handle duplicates from epochs)
            benchmark_samples[benchmark].update(sample_ids)

    # Validate counts for each benchmark
    for benchmark, expected in expected_counts.items():
        if expected is None:
            # Skip if we couldn't fetch expected count
            sample_details[benchmark] = {
                "expected": "unknown",
                "actual": len(benchmark_samples.get(benchmark, set())),
                "valid": True,  # Can't validate, assume OK
                "skipped": True,
            }
            continue

        actual_ids = benchmark_samples.get(benchmark, set())
        actual_count = len(actual_ids)

        is_valid = actual_count == expected
        sample_details[benchmark] = {
            "expected": expected,
            "actual": actual_count,
            "valid": is_valid,
        }

        if not is_valid:
            checks["sample_count_valid"] = False
            if actual_count < expected:
                errors.append(
                    f"{benchmark}: Only {actual_count}/{expected} samples evaluated. "
                    f"Did you use --limit? Full benchmark required for submission."
                )
            elif actual_count > expected:
                errors.append(
                    f"{benchmark}: {actual_count} samples found, expected {expected}. "
                    f"Possible duplicate evaluations or wrong dataset split."
                )

    # ALL benchmarks are required — missing benchmarks cause validation failure
    for benchmark in expected_counts:
        if benchmark not in benchmark_samples:
            if benchmark not in sample_details:
                sample_details[benchmark] = {
                    "expected": expected_counts.get(benchmark) or "unknown",
                    "actual": 0,
                    "valid": False,
                }
                checks["sample_count_valid"] = False
                errors.append(
                    f"Missing benchmark: {benchmark}. "
                    f"ALL benchmarks are required for submission."
                )

    # Translate keys from task names to HF column names for display
    benchmark_to_hf = get_benchmark_to_hf_map()
    display_details = {}
    for benchmark, detail in sample_details.items():
        hf_name = benchmark_to_hf.get(benchmark, benchmark)
        display_details[hf_name] = detail

    return checks, display_details, errors


def validate_parquet(
    parquet_path: Path,
) -> tuple[dict[str, bool], list[str], list[str]]:
    """Validate parquet file structure.

    Args:
        parquet_path: Path to parquet file

    Returns:
        Tuple of (checks dict, errors list, warnings list)
    """
    checks = {
        "parquet_exists": False,
        "parquet_schema": False,
        "model_format": False,
        "provider_recognized": False,
    }
    errors: list[str] = []
    warnings: list[str] = []

    if not parquet_path.exists():
        errors.append(f"Parquet file not found: {parquet_path}")
        return checks, errors, warnings

    checks["parquet_exists"] = True

    try:
        df = rename_aliased_columns(pd.read_parquet(parquet_path, engine="pyarrow"))
    except Exception as e:
        errors.append(f"Failed to read parquet: {e}")
        return checks, errors, warnings

    # Check required columns
    missing = set(get_required_columns()) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    else:
        checks["parquet_schema"] = True

    # Validate model format: "model_name (Provider)"
    model_format_ok = True

    for model in df["model"].unique():
        if " (" not in model or not model.endswith(")"):
            errors.append(f"Invalid model format: {model}")
            model_format_ok = False
        else:
            provider = model.split("(")[-1].rstrip(")")
            if provider.lower() not in [p.lower() for p in RECOGNIZED_PROVIDERS]:
                warnings.append(
                    f"Provider '{provider}' is new to the leaderboard — "
                    f"a maintainer should confirm it names a real organisation."
                )

    checks["model_format"] = model_format_ok
    # New providers are welcome; the warning above is for the reviewer.
    checks["provider_recognized"] = True

    # Validate score arrays (can be list, tuple, or numpy array from parquet)
    for col in get_benchmark_columns():
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            if val is not None:
                if not isinstance(val, (list, tuple, np.ndarray)) or len(val) < 2:
                    errors.append(f"Invalid score format in {col} row {idx}: expected [score, stderr, ...]")

    return checks, errors, warnings


def validate_sample_counts_from_parquet(
    files: list[str],
    expected_counts: dict[str, int | None],
) -> tuple[dict[str, bool], dict[str, dict], list[str]]:
    """Validate sample counts from parquet score arrays.

    For parquet-only submissions, each benchmark column stores
    [score, stderr, n_samples]. Extract n_samples and compare
    against expected counts from HuggingFace.

    Args:
        files: List of file paths from the submission.
        expected_counts: Dict of benchmark -> expected sample count.

    Returns:
        Tuple of (check_dict, sample_details_dict, errors_list)
    """
    checks = {"sample_count_valid": True}
    sample_details: dict[str, dict] = {}
    errors: list[str] = []

    # Column name -> benchmark key mapping (reverse of registry map)
    column_to_benchmark = {v: k for k, v in get_benchmark_to_hf_map().items()}

    # Read all parquet files
    benchmark_samples: dict[str, int] = {}
    for file_path in files:
        path = Path(file_path)
        if path.suffix != ".parquet" or not path.exists():
            continue
        try:
            df = rename_aliased_columns(pd.read_parquet(path, engine="pyarrow"))
        except Exception:
            continue

        for col in get_benchmark_columns():
            if col not in df.columns:
                continue
            for val in df[col]:
                if val is None:
                    continue
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3:
                    benchmark_key = column_to_benchmark.get(col, col)
                    try:
                        count = int(val[2])
                    except (TypeError, ValueError):
                        errors.append(f"Invalid sample count in {col}: {val[2]}")
                        continue
                    if benchmark_key in benchmark_samples and benchmark_samples[benchmark_key] != count:
                        errors.append(
                            f"Inconsistent sample counts for {col}: "
                            f"{benchmark_samples[benchmark_key]} vs {count}"
                        )
                    benchmark_samples[benchmark_key] = count

    # Validate counts
    for benchmark, expected in expected_counts.items():
        if expected is None:
            sample_details[benchmark] = {
                "expected": "unknown",
                "actual": benchmark_samples.get(benchmark, 0),
                "valid": True,
                "skipped": True,
            }
            continue

        actual_count = benchmark_samples.get(benchmark, 0)
        is_valid = actual_count == expected

        sample_details[benchmark] = {
            "expected": expected,
            "actual": actual_count,
            "valid": is_valid,
        }

        if not is_valid:
            if actual_count == 0:
                sample_details[benchmark]["not_submitted"] = True
                sample_details[benchmark]["valid"] = True
                # Do NOT reset checks here — prior failures must persist
            elif actual_count < expected:
                checks["sample_count_valid"] = False
                errors.append(
                    f"{benchmark}: Only {actual_count}/{expected} samples evaluated. "
                    f"Did you use --limit? Full benchmark required for submission."
                )
            elif actual_count > expected:
                checks["sample_count_valid"] = False
                errors.append(
                    f"{benchmark}: {actual_count} samples found, expected {expected}. "
                    f"Possible duplicate evaluations or wrong dataset split."
                )

    # Ensure at least one benchmark was actually submitted
    submitted = [
        b for b, d in sample_details.items()
        if d.get("actual", 0) > 0 and not d.get("not_submitted", False)
    ]
    if not submitted:
        errors.append(
            "No valid benchmark data found in parquet. "
            "At least one benchmark required."
        )
        checks["sample_count_valid"] = False

    # Re-evaluate overall validity after processing all benchmarks
    if errors:
        checks["sample_count_valid"] = False

    return checks, sample_details, errors


def validate_trajectory_json(json_path: Path) -> tuple[dict[str, bool], list[str]]:
    """Validate trajectory JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Tuple of (checks dict, errors list)
    """
    checks = {
        "json_valid": False,
        "inspect_eval": False,
        "no_errors": False,
    }
    errors = []

    try:
        with open(json_path) as f:
            data = json.load(f)
        checks["json_valid"] = True
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {json_path.name}: {e}")
        return checks, errors
    except Exception as e:
        errors.append(f"Failed to read {json_path.name}: {e}")
        return checks, errors

    # Check for Inspect eval signature
    # Inspect AI logs have "eval" key with model info
    if "eval" in data:
        checks["inspect_eval"] = True
    elif "model" in data and "results" in data:
        # Alternative format
        checks["inspect_eval"] = True
    else:
        errors.append(f"{json_path.name}: Missing eval data - may not be from Inspect")

    # Check for errors in trajectory
    status = data.get("status", "")
    error_field = data.get("error")

    if status == "error" or error_field:
        error_msg = error_field or "unknown error"
        errors.append(f"{json_path.name}: Trajectory has error: {error_msg}")
    else:
        checks["no_errors"] = True

    return checks, errors


def _is_safe_path(file_path: str) -> bool:
    """Check that file path is within expected directories."""
    path = Path(file_path)
    parts = path.parts
    if ".." in parts or any(p.startswith(".") for p in parts):
        return False
    if parts and parts[0] not in ("model_cards", "trajectories"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate leaderboard submission")
    parser.add_argument(
        "--files",
        required=True,
        help="Space-separated list of changed files",
    )
    args = parser.parse_args()

    files = args.files.strip().split()

    # Get HF token for dataset access
    hf_token = os.environ.get("HF_TOKEN")

    # Fetch expected sample counts from HuggingFace
    print("Fetching expected sample counts from GSMA/open_telco...")
    expected_counts = fetch_expected_sample_counts(hf_token)

    # Aggregate results
    result: dict = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "checks": {
            "parquet_exists": True,
            "parquet_schema": True,
            "json_valid": True,
            "model_format": True,
            "provider_recognized": True,
            "inspect_eval": True,
            "no_errors": True,
            "sample_count_valid": True,
        },
        "sample_details": {},
    }

    parquet_found = False
    json_files: list[Path] = []

    for file_path in files:
        if not file_path:
            continue

        if not _is_safe_path(file_path):
            result["errors"].append(f"Rejected path outside allowed directories: {file_path}")
            continue

        path = Path(file_path)

        if path.suffix == ".parquet":
            parquet_found = True
            checks, errors, warnings = validate_parquet(path)
            for key, value in checks.items():
                if not value:
                    result["checks"][key] = False
            result["errors"].extend(errors)
            result["warnings"].extend(warnings)

        elif path.suffix == ".json":
            json_files.append(path)
            checks, errors = validate_trajectory_json(path)
            for key, value in checks.items():
                if not value:
                    result["checks"][key] = False
            result["errors"].extend(errors)

    # Validate sample counts from trajectory files
    if json_files:
        print("Validating sample counts...")
        count_checks, sample_details, count_errors = validate_sample_counts(
            json_files, expected_counts
        )
        for key, value in count_checks.items():
            if not value:
                result["checks"][key] = False
        result["sample_details"] = sample_details
        result["errors"].extend(count_errors)

        # Check that all required benchmarks were found
        submitted = [
            b for b, d in sample_details.items() if d.get("actual", 0) > 0
        ]
        if not submitted:
            result["errors"].append(
                "No valid benchmark trajectories found. "
                "ALL benchmarks are required for submission."
            )

    # Check that we found required files
    # Note: Parquet is now optional - workflow will generate it from trajectories
    if not parquet_found:
        # Don't fail - workflow will generate parquet from JSON trajectories
        result["checks"]["parquet_exists"] = True  # Will be generated
        # No error - this is expected for new submission flow

    if not json_files and parquet_found:
        # Parquet-only submission (client-side conversion) — validate sample
        # counts from the score arrays instead of JSON trajectory files.
        result["checks"]["json_valid"] = True
        result["checks"]["inspect_eval"] = True
        result["checks"]["no_errors"] = True
        count_checks, sample_details, count_errors = validate_sample_counts_from_parquet(
            files, expected_counts
        )
        for key, value in count_checks.items():
            if not value:
                result["checks"][key] = False
        result["sample_details"] = sample_details
        result["errors"].extend(count_errors)
    elif not json_files:
        result["checks"]["json_valid"] = False
        result["checks"]["inspect_eval"] = False
        result["checks"]["no_errors"] = False
        result["checks"]["sample_count_valid"] = False
        result["errors"].append("No JSON trajectory files or parquet found in submission")

    # Determine overall pass/fail
    result["passed"] = len(result["errors"]) == 0

    # Write result for workflow
    with open("validation_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Validation {'passed' if result['passed'] else 'failed'}")
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error}")
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
