#!/usr/bin/env python3
"""Sync validated submissions to HuggingFace dataset.

All benchmarks are required for submission. Existing models are overwritten
with new scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

from registry import get_benchmark_columns, rename_aliased_columns

DATASET_REPO = "GSMA/leaderboard"


def _normalize_score(value: object) -> str | None:
    """Convert a submission score cell to the leaderboard's stored encoding.

    Submissions carry ``[score, stderr, n_samples]`` on a 0-100 scale; the
    dataset stores the string ``"[score, stderr]"`` on a 0-1 scale.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value  # already in dataset encoding
    try:
        score, stderr = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if pd.isna(score):
        return None
    # ponytail: >1 means percent. Only ambiguous for a genuine 1% score,
    # which no benchmark here produces. Add an explicit scale column if it ever does.
    if score > 1.0:
        score, stderr = score / 100, stderr / 100
    return f"[{round(score, 6)}, {round(stderr, 6)}]"


def normalize_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a submission parquet into the live leaderboard schema.

    Splits ``"name (Provider)"`` into separate model/provider columns, renames
    legacy benchmark columns, drops columns the leaderboard does not carry
    (``date``, benchmarks not yet on the board), and rewrites score cells to
    the dataset's 0-1 string encoding.  ``average``/``rank``/
    ``benchmarks_completed`` are deliberately left out — calculate_avg_score.py
    recomputes them across the whole board after the merge.
    """
    df = rename_aliased_columns(df).copy()

    models = df["model"].astype(str)
    parenthesised = models.str.match(r"^.+ \(.+\)$")
    df["provider"] = models.where(~parenthesised, models.str.rsplit(" (", n=1).str[-1].str.rstrip(")"))
    df["provider"] = df["provider"].where(parenthesised, None)
    df["model"] = models.where(~parenthesised, models.str.rsplit(" (", n=1).str[0])

    benchmark_cols = [c for c in get_benchmark_columns() if c in df.columns]
    for col in benchmark_cols:
        df[col] = [_normalize_score(v) for v in df[col]]

    return df[["model", "provider", *benchmark_cols]]


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
        print(f"Warning: Could not load existing dataset: {type(e).__name__}")
        return pd.DataFrame()


def merge_submissions(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Merge new entries into the leaderboard, overwriting existing models.

    Args:
        existing_df: Current leaderboard data
        new_df: New submission data (all benchmark scores required)

    Returns:
        Tuple of (merged DataFrame, sync report dict)
    """
    sync_report: dict = {
        "new_models": [],
        "updated_models": [],
    }

    if existing_df.empty:
        sync_report["new_models"] = new_df["model"].tolist()
        return new_df.copy(), sync_report

    if new_df.empty:
        return existing_df.copy(), sync_report

    # Align columns: union of both DataFrames, preserving order (existing first)
    all_columns = list(dict.fromkeys(
        existing_df.columns.tolist() + new_df.columns.tolist()
    ))
    for col in all_columns:
        if col not in existing_df.columns:
            existing_df[col] = None
        if col not in new_df.columns:
            new_df[col] = None

    result_df = existing_df.copy()
    existing_models = set(existing_df["model"].tolist())

    for _, new_row in new_df.iterrows():
        model_name = new_row["model"]

        if model_name in existing_models:
            idx = result_df[result_df["model"] == model_name].index[0]
            result_df.loc[idx] = new_row
            sync_report["updated_models"].append(model_name)
            continue

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
            df = normalize_submission(pd.read_parquet(path))
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

    merged_df, sync_report = merge_submissions(existing_df, new_df)
    print(f"Merged entries: {len(merged_df)}")

    print("\n=== Sync Report ===")
    if sync_report.get("new_models"):
        print(f"New models added: {sync_report['new_models']}")
    if sync_report.get("updated_models"):
        print(f"Models updated: {sync_report['updated_models']}")
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
