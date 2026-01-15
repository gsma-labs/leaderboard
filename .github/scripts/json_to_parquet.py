#!/usr/bin/env python3
"""Convert JSON trajectories to parquet format during validation.

This script parses Inspect AI trajectory JSON files and generates a parquet file
with the standardized leaderboard schema. It supports partial submissions where
not all benchmarks need to be present.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Benchmark task name to column mapping
TASK_TO_COLUMN = {
    "teleqna": "teleqna",
    "telelogs": "telelogs",
    "telemath": "telemath",
    "three_gpp": "3gpp_tsg",
    "teletables": "teletables",
}

REQUIRED_COLUMNS = ["model", "teleqna", "telelogs", "telemath", "3gpp_tsg", "teletables", "date"]


def extract_benchmark_from_task(task_name: str) -> str | None:
    """Map Inspect AI task name to benchmark column.

    Args:
        task_name: Task name from trajectory (e.g., 'three_gpp', 'teleqna')

    Returns:
        Column name or None if not recognized
    """
    task_lower = task_name.lower()
    for task_key, column in TASK_TO_COLUMN.items():
        if task_key in task_lower:
            return column
    return None


def extract_model_info(model_string: str) -> tuple[str, str]:
    """Extract model name and provider from model string.

    Handles formats like:
    - "openrouter/openai/gpt-5-mini" -> ("gpt-5-mini", "Openai")
    - "openai/gpt-4o" -> ("gpt-4o", "Openai")
    - "anthropic/claude-haiku-4.5" -> ("claude-haiku-4.5", "Anthropic")

    Args:
        model_string: Model identifier from trajectory

    Returns:
        Tuple of (model_name, provider)
    """
    parts = model_string.split("/")

    if len(parts) >= 3:
        # Format: router/provider/model (e.g., openrouter/openai/gpt-5-mini)
        provider = parts[1].capitalize()
        model_name = parts[-1]
    elif len(parts) == 2:
        # Format: provider/model
        provider = parts[0].capitalize()
        model_name = parts[1]
    else:
        # Just model name
        provider = "Unknown"
        model_name = parts[0]

    return model_name, provider


def parse_trajectory_json(json_path: Path) -> dict | None:
    """Parse a trajectory JSON file and extract score data.

    Args:
        json_path: Path to trajectory JSON file

    Returns:
        Dict with keys: benchmark, score, stderr, n_samples, model_name, provider
        or None if parsing fails
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to parse {json_path.name}: {e}")
        return None

    # Extract task/benchmark
    eval_data = data.get("eval", {})
    task_name = eval_data.get("task", "")
    benchmark = extract_benchmark_from_task(task_name)

    if not benchmark:
        print(f"Warning: Unrecognized task '{task_name}' in {json_path.name}")
        return None

    # Extract model info
    model_string = eval_data.get("model", "")
    if not model_string:
        print(f"Warning: No model found in {json_path.name}")
        return None

    model_name, provider = extract_model_info(model_string)

    # Extract score from results.scores[0].metrics
    results = data.get("results", {})
    scores_list = results.get("scores", [])

    if not scores_list:
        print(f"Warning: No scores found in {json_path.name}")
        return None

    metrics = scores_list[0].get("metrics", {})
    accuracy_data = metrics.get("accuracy", {})
    stderr_data = metrics.get("stderr", {})

    accuracy = accuracy_data.get("value")
    stderr = stderr_data.get("value")

    if accuracy is None:
        print(f"Warning: No accuracy value in {json_path.name}")
        return None

    # Convert to percentage (scores are stored as 0-1 in JSON, 0-100 in parquet)
    score = accuracy * 100
    stderr_val = (stderr * 100) if stderr else 0.0

    # Get n_samples
    n_samples = results.get("total_samples", 0)

    return {
        "benchmark": benchmark,
        "score": round(score, 2),
        "stderr": round(stderr_val, 6),
        "n_samples": float(n_samples),
        "model_name": model_name,
        "provider": provider,
    }


def generate_parquet(
    trajectory_files: list[Path],
    output_path: Path,
) -> dict:
    """Generate parquet file from trajectory JSON files.

    Args:
        trajectory_files: List of trajectory JSON file paths
        output_path: Path for output parquet file

    Returns:
        Dict with model info and benchmarks found

    Raises:
        ValueError: If no valid trajectory data found
    """
    # Parse all trajectories
    parsed_data: list[dict] = []
    for json_path in trajectory_files:
        result = parse_trajectory_json(json_path)
        if result:
            parsed_data.append(result)

    if not parsed_data:
        raise ValueError("No valid trajectory data found in any JSON files")

    # Get model info (should be consistent across all trajectories)
    model_name = parsed_data[0]["model_name"]
    provider = parsed_data[0]["provider"]

    # Verify all trajectories are from the same model
    for data in parsed_data[1:]:
        if data["model_name"] != model_name or data["provider"] != provider:
            print(
                f"Warning: Mixed models in trajectories: "
                f"{model_name} ({provider}) vs {data['model_name']} ({data['provider']})"
            )

    # Build row with score arrays [score, stderr, n_samples]
    row: dict = {
        "model": f"{model_name} ({provider})",
        "teleqna": None,
        "telelogs": None,
        "telemath": None,
        "3gpp_tsg": None,
        "teletables": None,
        "date": date.today().isoformat(),
    }

    benchmarks_found = []
    for data in parsed_data:
        benchmark = data["benchmark"]
        score_array = [data["score"], data["stderr"], data["n_samples"]]
        row[benchmark] = score_array
        benchmarks_found.append(benchmark)

    # Create DataFrame and save
    df = pd.DataFrame([row])
    df.to_parquet(output_path)

    return {
        "model_name": model_name,
        "provider": provider,
        "display_name": f"{model_name} ({provider})",
        "benchmarks_found": benchmarks_found,
        "benchmarks_missing": [b for b in TASK_TO_COLUMN.values() if b not in benchmarks_found],
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSON trajectories to parquet format"
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output parquet file",
    )
    args = parser.parse_args()

    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)

    if not trajectory_dir.exists():
        print(f"Error: Trajectory directory not found: {trajectory_dir}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(trajectory_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {trajectory_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON trajectory files")

    # Determine output filename from directory name
    dir_name = trajectory_dir.name  # e.g., "openai_gpt-5-mini"
    output_path = output_dir / f"{dir_name}.parquet"

    try:
        result = generate_parquet(json_files, output_path)
        print(f"Generated parquet: {result['output_path']}")
        print(f"Model: {result['display_name']}")
        print(f"Benchmarks found: {', '.join(result['benchmarks_found'])}")
        if result["benchmarks_missing"]:
            print(f"Benchmarks missing: {', '.join(result['benchmarks_missing'])}")

        # Write result for workflow
        with open("conversion_result.json", "w") as f:
            json.dump(result, f, indent=2)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
