#!/usr/bin/env python3
"""Validate leaderboard submission files with sample count verification.

Supports partial submissions where not all 4 benchmarks need to be present.
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

REQUIRED_COLUMNS = ["model", "teleqna", "telelogs", "telemath", "3gpp_tsg", "teletables", "date"]

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

# Map benchmark task names to HuggingFace dataset config names
BENCHMARK_TO_HF_CONFIG = {
    "teleqna": "teleqna",
    "telelogs": "telelogs",
    "telemath": "telemath",
    "three_gpp": "3gpp_tsg",
    "teletables": "teletables",
}


def fetch_expected_sample_counts(token: str | None = None) -> dict[str, int | None]:
    """Fetch expected sample counts from GSMA/open_telco HuggingFace dataset.

    Args:
        token: Optional HuggingFace token for dataset access

    Returns:
        Dict mapping benchmark name to expected sample count (None if fetch failed)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not available, skipping sample count validation")
        return {k: None for k in BENCHMARK_TO_HF_CONFIG}

    expected_counts: dict[str, int | None] = {}
    max_retries = 3

    for benchmark, hf_config in BENCHMARK_TO_HF_CONFIG.items():
        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    OPEN_TELCO_DATASET,
                    name=hf_config,
                    split="test",
                    token=token,
                )
                expected_counts[benchmark] = len(ds)
                print(f"  {benchmark}: {len(ds)} samples expected")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    print(f"Warning: Could not fetch {benchmark} count after {max_retries} attempts: {e}")
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
    if "eval" in data and isinstance(data["eval"], dict):
        task = data["eval"].get("task", "")
        # Normalize benchmark name
        task_lower = task.lower()
        for key in BENCHMARK_TO_HF_CONFIG:
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

    # Note: We no longer fail for missing benchmarks - partial submissions are allowed
    # Only record missing benchmarks for informational purposes
    for benchmark in expected_counts:
        if benchmark not in benchmark_samples and expected_counts[benchmark] is not None:
            if benchmark not in sample_details:
                sample_details[benchmark] = {
                    "expected": expected_counts[benchmark],
                    "actual": 0,
                    "valid": True,  # Not a failure - just not submitted
                    "not_submitted": True,
                }
                # Don't mark as invalid - partial submissions are OK

    return checks, sample_details, errors


def validate_parquet(parquet_path: Path) -> tuple[dict[str, bool], list[str]]:
    """Validate parquet file structure.

    Args:
        parquet_path: Path to parquet file

    Returns:
        Tuple of (checks dict, errors list)
    """
    checks = {
        "parquet_exists": False,
        "parquet_schema": False,
        "model_format": False,
        "provider_recognized": False,
    }
    errors = []

    if not parquet_path.exists():
        errors.append(f"Parquet file not found: {parquet_path}")
        return checks, errors

    checks["parquet_exists"] = True

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        errors.append(f"Failed to read parquet: {e}")
        return checks, errors

    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    else:
        checks["parquet_schema"] = True

    # Validate model format: "model_name (Provider)"
    model_format_ok = True
    provider_recognized = True

    for model in df["model"].unique():
        if " (" not in model or not model.endswith(")"):
            errors.append(f"Invalid model format: {model}")
            model_format_ok = False
        else:
            provider = model.split("(")[-1].rstrip(")")
            if provider.lower() not in [p.lower() for p in RECOGNIZED_PROVIDERS]:
                errors.append(f"Unrecognized provider: {provider}")
                provider_recognized = False

    checks["model_format"] = model_format_ok
    checks["provider_recognized"] = provider_recognized

    # Validate score arrays (can be list, tuple, or numpy array from parquet)
    for col in ["teleqna", "telelogs", "telemath", "3gpp_tsg", "teletables"]:
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            if val is not None:
                if not isinstance(val, (list, tuple, np.ndarray)) or len(val) < 2:
                    errors.append(f"Invalid score format in {col} row {idx}: expected [score, stderr, ...]")

    return checks, errors


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

        path = Path(file_path)

        if path.suffix == ".parquet":
            parquet_found = True
            checks, errors = validate_parquet(path)
            for key, value in checks.items():
                if not value:
                    result["checks"][key] = False
            result["errors"].extend(errors)

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

        # Check that at least one valid benchmark was found
        submitted_benchmarks = [
            b for b, d in sample_details.items()
            if d.get("actual", 0) > 0 and not d.get("not_submitted", False)
        ]
        if not submitted_benchmarks:
            result["errors"].append(
                "No valid benchmark trajectories found. "
                "At least one benchmark (teleqna, telelogs, telemath, 3gpp_tsg, teletables) required."
            )

    # Check that we found required files
    # Note: Parquet is now optional - workflow will generate it from trajectories
    if not parquet_found:
        # Don't fail - workflow will generate parquet from JSON trajectories
        result["checks"]["parquet_exists"] = True  # Will be generated
        # No error - this is expected for new submission flow

    if not json_files:
        result["checks"]["json_valid"] = False
        result["checks"]["inspect_eval"] = False
        result["checks"]["no_errors"] = False
        result["checks"]["sample_count_valid"] = False
        result["errors"].append("No JSON trajectory files found in submission")

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

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
