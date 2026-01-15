#!/usr/bin/env python3
"""Sync validated submissions to HuggingFace dataset with intelligent merge.

Supports partial submissions by:
- Only updating benchmark columns that are submitted
- Preserving existing scores for non-submitted benchmarks
- Generating a sync report showing what was updated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset


DATASET_REPO = "GSMA/leaderboard"
BENCHMARK_COLUMNS = ["teleqna", "telelogs", "telemath", "3gpp_tsg", "teletables"]


def load_existing_dataset(token: str) -> pd.DataFrame:
    """Load existing leaderboard dataset from HuggingFace.

    Args:
        token: HuggingFace API token

    Returns:
        DataFrame with existing data, or empty DataFrame if not found
    """
    try:
        from datasets import load_dataset

        existing_ds = load_dataset(DATASET_REPO, split="train", token=token)
        df = existing_ds.to_pandas()
        # Drop index column that HuggingFace stores to avoid duplicates on re-upload
        if "__index_level_0__" in df.columns:
            df = df.drop(columns=["__index_level_0__"])
        return df
    except Exception as e:
        print(f"Warning: Could not load existing dataset: {e}")
        return pd.DataFrame()


def is_null_score(value) -> bool:
    """Check if a score value is null/missing.

    Args:
        value: Score value to check (can be None, list, array, etc.)

    Returns:
        True if the score is considered null/missing
    """
    if value is None:
        return True
    if pd.isna(value) if not isinstance(value, (list, tuple)) else False:
        return True
    # Check for empty lists/arrays
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    return False


def merge_with_preservation(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Merge new entries with existing data, intelligently preserving scores.

    Logic:
    - If model EXISTS in existing_df:
        - For each benchmark column in new_df:
            - If new value is not null: update (overwrite submitted benchmarks)
            - If new value is null: keep existing value
    - If model is NEW: add entire row

    Args:
        existing_df: Current leaderboard data
        new_df: New submission data

    Returns:
        Tuple of (merged DataFrame, sync report dict)
    """
    sync_report: dict = {
        "new_models": [],
        "updated_models": [],
        "updated_scores": [],
        "preserved_scores": [],
    }

    if existing_df.empty:
        sync_report["new_models"] = new_df["model"].tolist()
        return new_df.copy(), sync_report

    if new_df.empty:
        return existing_df.copy(), sync_report

    result_df = existing_df.copy()
    existing_models = set(existing_df["model"].tolist())

    for _, new_row in new_df.iterrows():
        model_name = new_row["model"]

        if model_name in existing_models:
            # Model EXISTS - intelligent merge
            idx = result_df[result_df["model"] == model_name].index[0]

            model_updated = False
            for col in BENCHMARK_COLUMNS:
                new_val = new_row.get(col)
                existing_val = result_df.at[idx, col]

                new_is_null = is_null_score(new_val)
                existing_is_null = is_null_score(existing_val)

                if not new_is_null:
                    # New submission has a score for this benchmark - UPDATE
                    result_df.at[idx, col] = new_val
                    model_updated = True

                    if existing_is_null:
                        sync_report["updated_scores"].append(f"{model_name}:{col} (new)")
                    else:
                        sync_report["updated_scores"].append(f"{model_name}:{col} (updated)")
                elif not existing_is_null:
                    # New is null but existing has a value - PRESERVE
                    sync_report["preserved_scores"].append(f"{model_name}:{col}")

            # Update date if any scores were updated
            if model_updated:
                if "date" in new_row and new_row["date"]:
                    result_df.at[idx, "date"] = new_row["date"]
                sync_report["updated_models"].append(model_name)

        else:
            # Model is NEW - add entire row
            new_row_df = pd.DataFrame([new_row])
            result_df = pd.concat([result_df, new_row_df], ignore_index=True)
            sync_report["new_models"].append(model_name)

    return result_df, sync_report


def upload_to_huggingface(df: pd.DataFrame, token: str) -> None:
    """Upload DataFrame to HuggingFace dataset.

    Args:
        df: DataFrame to upload
        token: HuggingFace API token
    """
    # Drop any lingering index columns and reset to avoid schema conflicts
    index_cols = [c for c in df.columns if c.startswith("__index_level")]
    if index_cols:
        df = df.drop(columns=index_cols)
    df = df.reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(DATASET_REPO, token=token)
    print(f"Successfully uploaded {len(df)} entries to {DATASET_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync submissions to HuggingFace")
    parser.add_argument(
        "--files",
        required=True,
        help="Space-separated parquet files to sync",
    )
    args = parser.parse_args()

    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        sys.exit(1)

    # Parse file list
    files = args.files.strip().split()

    if not files or all(not f for f in files):
        print("No files to sync")
        sys.exit(0)

    # Load new submissions
    new_rows = []
    for file_path in files:
        if not file_path:
            continue

        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue

        if path.suffix != ".parquet":
            continue

        try:
            df = pd.read_parquet(path)
            new_rows.append(df)
            print(f"Loaded {len(df)} entries from {path}")
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
            continue

    if not new_rows:
        print("No valid submissions to sync")
        sys.exit(0)

    # Combine new submissions
    new_df = pd.concat(new_rows, ignore_index=True)
    print(f"Total new entries: {len(new_df)}")

    # Load existing dataset
    print(f"Loading existing dataset from {DATASET_REPO}...")
    existing_df = load_existing_dataset(hf_token)
    print(f"Existing entries: {len(existing_df)}")

    # Intelligent merge with preservation
    merged_df, sync_report = merge_with_preservation(existing_df, new_df)
    print(f"Merged entries: {len(merged_df)}")

    # Log sync report
    print("\n=== Sync Report ===")
    if sync_report.get("new_models"):
        print(f"New models added: {sync_report['new_models']}")
    if sync_report.get("updated_models"):
        print(f"Models updated: {sync_report['updated_models']}")
    if sync_report.get("updated_scores"):
        print(f"Scores updated: {sync_report['updated_scores']}")
    if sync_report.get("preserved_scores"):
        print(f"Scores preserved: {sync_report['preserved_scores']}")
    print("==================\n")

    # Write sync report for workflow to post as comment
    with open("sync_report.json", "w") as f:
        json.dump(sync_report, f, indent=2)

    # Upload
    print("Uploading to HuggingFace...")
    upload_to_huggingface(merged_df, hf_token)

    print("Sync complete!")


if __name__ == "__main__":
    main()
