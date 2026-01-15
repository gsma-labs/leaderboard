#!/usr/bin/env python3
"""Calculate TCI for all leaderboard entries and update HuggingFace dataset.

This script is called by the GitHub Action after syncing parquet files to
HuggingFace. It:
1. Fetches the current dataset from GSMA/leaderboard
2. Calculates TCI for all entries using dynamic IRT fitting
3. Adds/updates the 'tci' column
4. Pushes the updated dataset back to HuggingFace
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

# Add tci module to path
sys.path.insert(0, str(Path(__file__).parent))

from tci import LeaderboardEntry, calculate_all_tci, calculate_error

DATASET_REPO = "GSMA/leaderboard"


def parse_model_provider(combined: str) -> tuple[str, str]:
    """Parse 'model (Provider)' format.

    Args:
        combined: Model string in format "model_name (Provider)"

    Returns:
        Tuple of (model_name, provider)
    """
    match = re.match(r"^(.+?)\s*\(([^)]+)\)$", combined)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return combined, "Unknown"


def extract_score(value) -> tuple[float | None, float | None]:
    """Extract score and stderr from [score, stderr, n_samples] format.

    Args:
        value: Array in format [score, stderr, n_samples] or None
               Can be a list, tuple, or numpy array.

    Returns:
        Tuple of (score, stderr)
    """
    if value is None:
        return None, None
    # Check if value is array-like (has length and is indexable)
    try:
        if len(value) < 2:
            return None, None
        return float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None, None


def load_leaderboard(token: str) -> pd.DataFrame:
    """Load existing leaderboard from HuggingFace.

    Args:
        token: HuggingFace API token

    Returns:
        DataFrame with leaderboard data
    """
    try:
        ds = load_dataset(DATASET_REPO, split="train", token=token)
        df = ds.to_pandas()
        # Drop index column that HuggingFace stores
        if "__index_level_0__" in df.columns:
            df = df.drop(columns=["__index_level_0__"])
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def transform_to_entries(df: pd.DataFrame) -> list[LeaderboardEntry]:
    """Transform DataFrame rows to LeaderboardEntry objects.

    Args:
        df: DataFrame with leaderboard data

    Returns:
        List of LeaderboardEntry objects
    """
    entries = []
    for _, row in df.iterrows():
        model_str = row.get("model", "Unknown")
        model, provider = parse_model_provider(model_str)

        teleqna, teleqna_stderr = extract_score(row.get("teleqna"))
        telelogs, telelogs_stderr = extract_score(row.get("telelogs"))
        telemath, telemath_stderr = extract_score(row.get("telemath"))
        tsg, tsg_stderr = extract_score(row.get("3gpp_tsg"))
        teletables, teletables_stderr = extract_score(row.get("teletables"))

        entries.append(
            LeaderboardEntry(
                model=model,
                provider=provider,
                teleqna=teleqna,
                teleqna_stderr=teleqna_stderr,
                telelogs=telelogs,
                telelogs_stderr=telelogs_stderr,
                telemath=telemath,
                telemath_stderr=telemath_stderr,
                tsg=tsg,
                tsg_stderr=tsg_stderr,
                teletables=teletables,
                teletables_stderr=teletables_stderr,
            )
        )

    return entries


def add_tci_column(df: pd.DataFrame, entries: list[LeaderboardEntry]) -> pd.DataFrame:
    """Add TCI column to DataFrame from calculated entries.

    Args:
        df: Original DataFrame
        entries: LeaderboardEntry objects with calculated TCI

    Returns:
        DataFrame with 'tci' column added
    """
    tci_values = []
    for entry in entries:
        if entry.tci is not None:
            stderr = calculate_error(entry.tci, "tci")
            # Format as [score, stderr, 0] to match benchmark column format
            tci_values.append([entry.tci, stderr, 0])
        else:
            tci_values.append(None)

    df["tci"] = tci_values
    return df


def main() -> None:
    """Main entry point for TCI calculation."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        sys.exit(1)

    print(f"Loading dataset from {DATASET_REPO}...")
    df = load_leaderboard(hf_token)
    print(f"Loaded {len(df)} entries")

    if df.empty:
        print("Dataset is empty, skipping TCI calculation")
        sys.exit(0)

    print("Transforming to LeaderboardEntry objects...")
    entries = transform_to_entries(df)

    print("Calculating TCI using dynamic IRT fitting...")
    entries, irt_params = calculate_all_tci(entries)

    # Log IRT parameters for debugging
    print(f"IRT fit residual: {irt_params.fit_residual:.4f}")
    print(f"Benchmark difficulties: {irt_params.difficulty}")
    print(f"Benchmark slopes: {irt_params.slope}")

    tci_count = sum(1 for e in entries if e.tci is not None)
    print(f"Calculated TCI for {tci_count}/{len(entries)} models")

    print("Adding TCI column to dataset...")
    df = add_tci_column(df, entries)

    print("Uploading to HuggingFace...")
    # Clean up any lingering index columns
    index_cols = [c for c in df.columns if c.startswith("__index_level")]
    if index_cols:
        df = df.drop(columns=index_cols)
    df = df.reset_index(drop=True)

    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(DATASET_REPO, token=hf_token)
    print(f"Successfully updated {DATASET_REPO} with TCI column")


if __name__ == "__main__":
    main()
