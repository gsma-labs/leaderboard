#!/usr/bin/env python3
"""Publish one immutable, revalidated submission with a single guarded Hub commit."""
from __future__ import annotations

import argparse
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from calculate_avg_score import compute_derived_columns
from registry import get_benchmark_columns, rename_aliased_columns
from validate_submission import read_card, validate_submission

DATASET_REPO = 'GSMA/leaderboard'
PROVENANCE_COLUMN = 'submission_provenance'


def _normalize_score(value):
    """Submission triples are percent; existing stored string pairs are fractions."""
    if isinstance(value, str):
        return value
    score, stderr = float(value[0]) / 100, float(value[1]) / 100
    return f'[{round(score, 6)}, {round(stderr, 6)}]'


def normalize_submission(df):
    df = rename_aliased_columns(df).copy()
    models = df['model'].astype(str)
    parenthesised = models.str.match(r'^.+ \(.+\)$')
    if 'provider' not in df:
        df['provider'] = None
    df.loc[parenthesised, 'provider'] = models[parenthesised].str.rsplit(' (', n=1).str[-1].str.rstrip(')')
    df.loc[parenthesised, 'model'] = models[parenthesised].str.rsplit(' (', n=1).str[0]
    columns = [c for c in get_benchmark_columns() if c in df]
    for col in columns:
        df[col] = [_normalize_score(v) for v in df[col]]
    return df[['model', 'provider', *columns]]


def load_existing_dataset(token=None):
    """Compatibility reader: unlike the former publisher, errors never mean empty."""
    return HubStore(token).snapshot().frame


def merge_submissions(existing_df, new_df):
    """Merge only selected rows, preserving incumbent metadata and provider credit."""
    result = existing_df.copy(deep=True)
    if result['model'].duplicated().any():
        raise ValueError('Existing board has ambiguous duplicate model names')
    report = {'new_models': [], 'updated_models': []}
    for _, row in new_df.iterrows():
        match = result.index[result['model'].str.casefold() == row['model'].casefold()]
        if len(match):
            idx = match[0]
            provider = result.at[idx, 'provider']
            if not isinstance(provider,str) or provider.casefold() != row['provider'].casefold():
                raise ValueError(f"Same model name with a different provider: {row['model']}")
            for col, value in row.items():
                if col not in ('model', 'provider'):
                    if col not in result:
                        result[col] = None
                    result.at[idx, col] = value
            report['updated_models'].append(result.at[idx, 'model'])
        else:
            result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
            report['new_models'].append(row['model'])
    return result, report


@dataclass
class Snapshot:
    sha: str
    frame: pd.DataFrame
    readme: str
    parquet_path: str


class HubStore:
    """Small Hub I/O boundary. Reads are public and pinned; writes use parent SHA."""
    def __init__(self, token=None):
        from huggingface_hub import HfApi
        self.token = token
        self.api = HfApi(token=token or False)

    def snapshot(self, revision=None):
        from huggingface_hub import hf_hub_download
        info = self.api.dataset_info(DATASET_REPO, revision=revision, token=False)
        sha = info.sha
        if not sha or (revision and sha != revision):
            raise ValueError('Hub did not resolve the requested immutable board revision')
        files = [s.rfilename for s in info.siblings]
        parquets = [p for p in files if p.endswith('.parquet')]
        if len(parquets) != 1:
            raise ValueError(f'Expected one authoritative parquet, found {parquets}')
        parquet_path = parquets[0]
        parquet = hf_hub_download(DATASET_REPO, parquet_path, repo_type='dataset', revision=sha, token=False)
        readme = hf_hub_download(DATASET_REPO, 'README.md', repo_type='dataset', revision=sha, token=False)
        frame = pd.read_parquet(parquet)
        if frame.empty:
            raise ValueError('Unexpected empty existing leaderboard; publication aborted')
        return Snapshot(sha, frame, Path(readme).read_text(), parquet_path)

    def commit(self, operations, parent_commit):
        from huggingface_hub import CommitOperationAdd
        if not self.token:
            raise ValueError('HF_TOKEN is required for publication; dry-run needs no write token')
        commit = self.api.create_commit(repo_id=DATASET_REPO, repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=data) for p,data in operations.items()],
            parent_commit=parent_commit, commit_message='Publish validated leaderboard submission')
        return commit.oid


def updated_readme(original, frame, parquet_path, parquet_size):
    """Preserve Hub prose/metadata and refresh schema/split metadata atomically."""
    from datasets import Dataset
    if not original.startswith('---\n') or '\n---' not in original[4:]:
        raise ValueError('Leaderboard README must have YAML metadata')
    front, prose = original[4:].split('\n---', 1)
    metadata = yaml.safe_load(front) or {}
    info = metadata.get('dataset_info', {})
    if isinstance(info, list):
        if len(info) != 1:
            raise ValueError('Ambiguous dataset_info configurations')
        info = info[0]
    info = dict(info)
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    features = dataset.info._to_yaml_dict()['features']
    arrow_bytes = dataset.data.nbytes
    info.update(features=features, splits=[{'name':'train', 'num_bytes':arrow_bytes, 'num_examples':len(frame)}], download_size=parquet_size,
                dataset_size=arrow_bytes)
    metadata['dataset_info'] = info
    # Make parquet authoritative; historic CSV/JSON files remain unchanged.
    configs = metadata.get('configs', [{'config_name':'default'}])
    if len(configs) != 1:
        raise ValueError('Ambiguous leaderboard data configurations')
    configs[0]['data_files'] = [{'split':'train', 'path':parquet_path}]
    metadata['configs'] = configs
    prose = re.sub(r'\*\*\d+ models\*\*', f'**{len(frame)} models**', prose)
    note = '\n\nThe parquet referenced in `configs` is the authoritative leaderboard. Historic CSV/JSON exports are retained unchanged.\n'
    if 'The parquet referenced in `configs` is the authoritative leaderboard.' not in prose:
        prose = prose.rstrip() + note
    return '---\n' + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + '---' + prose


def _assert_parity(expected, actual):
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True),
                                  check_dtype=False, check_exact=True)


def publish_submission(manifest, *, hub=None, dry_run=False, count_loader=None, is_ancestor=None):
    """Revalidate, merge locally, commit once, then verify the immutable readback.

    The caller obtains manifests from the trusted GitHub API helper. A receipt is
    returned only after parity succeeds; dry runs explicitly remain unverified.
    """
    if not manifest.get('applicable'):
        return {'no_op':True, 'reason':'No changed model card', 'verified':False, 'dry_run':dry_run}
    if manifest.get('mode') != 'publish' or not manifest.get('merge_sha'):
        raise ValueError('Publication requires a merged-PR manifest')
    paths = [Path(manifest['data_dir']) / p for p in manifest['paths']]
    validation = validate_submission(paths, dataset_variant=manifest.get('dataset_variant'), count_loader=count_loader)
    if not validation['passed']:
        raise ValueError(f"Submission validation failed: {validation['errors']}")
    hub = hub or HubStore(None if dry_run else os.environ.get('HF_TOKEN'))
    current = hub.snapshot()
    if current.frame.empty:
        raise ValueError('Unexpected empty existing leaderboard; publication aborted')
    new = normalize_submission(read_card(Path(manifest['data_dir']) / manifest['card']))
    old_matches = current.frame[current.frame.model.str.casefold() == new.iloc[0]['model'].casefold()]
    old_provenance = None
    if not old_matches.empty and PROVENANCE_COLUMN in current.frame:
        encoded = old_matches.iloc[0][PROVENANCE_COLUMN]
        if encoded is not None and not pd.isna(encoded):
            old_provenance = json.loads(encoded)
    if old_provenance and old_provenance.get('merge_sha'):
        old_merge = old_provenance['merge_sha']
        if old_merge != manifest['merge_sha']:
            if not is_ancestor or not is_ancestor(old_merge, manifest['merge_sha']):
                raise ValueError('Refusing stale same-model update: merge ancestry is not newer')
        elif old_provenance.get('reviewed_head_sha') != manifest['reviewed_head_sha']:
            raise ValueError('Same merge SHA has conflicting reviewed head provenance')
    provenance = {**validation['provenance'], 'pr_number':manifest['pr_number'],
                  'repository':manifest['repo'], 'reviewed_head_sha':manifest['reviewed_head_sha'],
                  'merge_sha':manifest['merge_sha'], 'merged_at':manifest['merged_at'],
                  'task_counts':{t:d['card_count'] for t,d in validation['sample_details'].items()}}
    # An immutable submission retry retains its original validation receipt,
    # even if the public validation dataset HEAD has since advanced.
    if old_provenance and old_provenance.get('merge_sha') == manifest['merge_sha']:
        provenance = old_provenance
    new[PROVENANCE_COLUMN] = json.dumps(provenance, sort_keys=True, separators=(',', ':'))
    before = current.frame.copy(deep=True)
    if PROVENANCE_COLUMN not in before:
        before[PROVENANCE_COLUMN] = None  # unknown historic provenance stays unknown
    merged, report = merge_submissions(before, new)
    merged = compute_derived_columns(merged).reset_index(drop=True)
    receipt = {**report, 'pr_number':manifest['pr_number'], 'reviewed_head_sha':manifest['reviewed_head_sha'],
               'merge_sha':manifest['merge_sha'], 'previous_hub_sha':current.sha,
               'row_count':len(merged), 'validation':validation, 'dry_run':dry_run, 'verified':False, 'no_op':False}
    unchanged = False
    try:
        _assert_parity(before, merged)
        unchanged = True
    except AssertionError:
        pass
    if dry_run:
        receipt.update(no_op=unchanged, proposed_rows=merged.to_dict('records'))
        return receipt
    if unchanged:
        # Verify at the immutable existing SHA even when no new write is needed.
        _assert_parity(merged, hub.snapshot(revision=current.sha).frame)
        receipt.update(verified=True, no_op=True, hub_commit_sha=current.sha)
        return receipt
    buffer = io.BytesIO()
    merged.to_parquet(buffer, index=False)
    parquet = buffer.getvalue()
    readme = updated_readme(current.readme, merged, current.parquet_path, len(parquet))
    operations = {current.parquet_path:parquet}
    if readme != current.readme:
        operations['README.md'] = readme.encode()
    commit_sha = hub.commit(operations, parent_commit=current.sha)
    readback = hub.snapshot(revision=commit_sha)
    _assert_parity(merged, readback.frame)
    if readback.readme != readme or readback.parquet_path != current.parquet_path:
        raise ValueError('Immutable Hub readback metadata does not match publication')
    receipt.update(verified=True, hub_commit_sha=commit_sha)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('sync_report.json'))
    args = parser.parse_args()
    from submission_manifest import GitHubAPI
    manifest = json.loads(args.manifest.read_text())
    api = GitHubAPI(manifest['repo'], os.environ.get('GITHUB_TOKEN'))
    result = publish_submission(manifest, dry_run=args.dry_run, is_ancestor=api.is_ancestor)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'proposed_rows'}, indent=2))


if __name__ == '__main__':
    main()
