#!/usr/bin/env python3
"""Self-check for submission -> leaderboard normalization.

Run: python .github/scripts/test_normalize.py

Guards the merge that used to corrupt the dataset: a raw submission parquet
carries 0-100 scores, [score, stderr, n] triples, a `3gpp_tsg` column, a `date`
column and a "name (Provider)" model string, none of which match the live
schema. Merging one unnormalized added three junk columns and a row with no
provider, no rank and scores 100x everyone else's.
"""

import pandas as pd

from registry import rename_aliased_columns
from sync_to_huggingface import merge_submissions, normalize_submission

# Shape of a real submission (cf. PR #31/#32/#35 on gsma-labs/leaderboard).
SUBMISSION = pd.DataFrame([{
    "model": "TSLAM-30B (NetoAISolutions)",
    "oranbench": [79.33, 3.317187, 150.0],
    "srsranbench": [84.67, 2.951762, 150.0],
    "telelogs": [44.0, 4.988877, 100.0],
    "telemath": [72.0, 4.512609, 100.0],
    "teleqna": [78.7, 1.295372, 1000.0],
    "teletables": [34.0, 4.760952, 100.0],
    "3gpp_tsg": [42.0, 4.96045, 100.0],
    "sixg_bench": None,
    "date": "2026-06-05",
}])

# Shape of the live GSMA/leaderboard rows.
LIVE = pd.DataFrame([{
    "model": "gpt-5-nano",
    "provider": "OpenAI",
    "rank": 31,
    "average": 0.603043,
    "benchmarks_completed": 7,
    "teleqna": "[0.812, 0.0124]",
    "teletables": "[0.42, 0.0496]",
    "oranbench": "[0.66, 0.0387]",
    "srsranbench": "[0.6733, 0.0384]",
    "telemath": "[0.62, 0.0488]",
    "telelogs": "[0.51, 0.0502]",
    "three_gpp": "[0.526, 0.0501]",
}])


def test_aliases():
    assert "three_gpp" in rename_aliased_columns(SUBMISSION).columns
    assert "3gpp_tsg" not in rename_aliased_columns(SUBMISSION).columns
    # Idempotent: renaming an already-current frame is a no-op.
    assert list(rename_aliased_columns(LIVE).columns) == list(LIVE.columns)


def test_model_and_provider_split():
    row = normalize_submission(SUBMISSION).iloc[0]
    assert row["model"] == "TSLAM-30B", row["model"]
    assert row["provider"] == "NetoAISolutions", row["provider"]


def test_scores_match_live_encoding():
    row = normalize_submission(SUBMISSION).iloc[0]
    # 0-100 triple -> 0-1 string pair, matching the dataset.
    assert row["oranbench"] == "[0.7933, 0.033172]", row["oranbench"]
    assert row["three_gpp"] == "[0.42, 0.049604]", row["three_gpp"]
    assert isinstance(row["teleqna"], str)


def test_already_normalized_passes_through():
    # Re-syncing a card that is already in dataset encoding must not rescale it.
    out = normalize_submission(LIVE)
    assert out.iloc[0]["teleqna"] == "[0.812, 0.0124]"
    assert out.iloc[0]["model"] == "gpt-5-nano"


def test_merge_adds_no_columns():
    """The actual corruption check: a merge must not change the schema."""
    merged, report = merge_submissions(LIVE.copy(), normalize_submission(SUBMISSION))
    assert list(merged.columns) == list(LIVE.columns), list(merged.columns)
    assert report["new_models"] == ["TSLAM-30B"], report
    new = merged[merged.model == "TSLAM-30B"].iloc[0]
    assert new["provider"] == "NetoAISolutions"
    # Scores land on the same scale as the incumbent row.
    assert float(new["oranbench"].strip("[]").split(",")[0]) <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
