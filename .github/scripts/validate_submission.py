#!/usr/bin/env python3
"""Fail-closed validation of one model card and its complete optional Inspect logs."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from registry import get_benchmark_to_hf_map, get_dataset_config, rename_aliased_columns

TASKS = get_benchmark_to_hf_map()
# Card triples are explicitly percentages, including 0% and 1%.
SCORE_SCALE = 'percent'
# Existing cards round percentages to two decimals and stderr to six decimals.
SCORE_TOLERANCE = 0.005001
STDERR_TOLERANCE = 0.00000051


def fetch_expected_sample_counts(variant: str) -> tuple[str, dict[str, int]]:
    """Resolve the public dataset HEAD once, then read all counts at that SHA.

    Failure to read even one config aborts validation. This revision is evidence
    about our validation, not a claim about the data used by an older evaluation.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo = f'GSMA/ot-{variant}'
    revision = HfApi().dataset_info(repo, token=False).sha
    if not revision or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError(f'Cannot resolve immutable revision for {repo}')
    counts = {}
    for task in TASKS:
        dataset = load_dataset(repo, name=get_dataset_config(task), split='test',
                               revision=revision, token=False)
        counts[task] = len(dataset)
    return revision, counts


def read_log(path: Path) -> dict:
    with (gzip.open(path, 'rt', encoding='utf-8') if path.name.endswith('.json.gz')
          else path.open(encoding='utf-8')) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('log must be a JSON object')
    return value


def read_card(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if '3gpp_tsg' in df and 'three_gpp' in df:
        raise ValueError('Ambiguous three_gpp and 3gpp_tsg columns')
    df = rename_aliased_columns(df)
    if len(df) != 1:
        raise ValueError('Exactly one model row is required')
    missing = {'model', 'date', *TASKS.values()} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if not isinstance(df.iloc[0]['model'], str) or not re.fullmatch(r'.+ \([^()]+\)', df.iloc[0]['model']):
        raise ValueError('Model must have format name (Provider)')
    if pd.isna(pd.to_datetime(df.iloc[0]['date'], errors='coerce')):
        raise ValueError('A valid evaluation date is required')
    for task, col in TASKS.items():
        value = df.iloc[0][col]
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
            raise ValueError(f'{task}: expected finite [score, stderr, count] triple')
        if any(isinstance(v, (bool, np.bool_)) for v in value):
            raise ValueError(f'{task}: boolean values are not numeric scores/counts')
        score, stderr, count = map(float, value)
        if not all(math.isfinite(v) for v in (score, stderr, count)):
            raise ValueError(f'{task}: triple must be finite')
        if not 0 <= score <= 100 or not 0 <= stderr <= 100:
            raise ValueError(f'{task}: score and stderr must be percentages in [0, 100]')
        if count <= 0 or not count.is_integer():
            raise ValueError(f'{task}: count must be a positive integer')
    return df


def _sample_value(value):
    if isinstance(value, str) and value in ('C', 'I'):
        return 1.0 if value == 'C' else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value) and 0 <= value <= 1:
            return value
    raise ValueError(f'unsupported saved sample scoring: {value!r}')


def _ids(values):
    if not isinstance(values, list):
        raise ValueError('sample IDs must be a list')
    if any(isinstance(x, (bool, list, dict)) or x is None for x in values):
        raise ValueError('sample IDs must be scalar non-null strings or numbers')
    return [json.dumps(x, sort_keys=True) for x in values]


def validate_log(log: dict, task: str, expected: int, triple) -> dict:
    """Check actual samples, declarations and saved metrics, not just ID metadata."""
    if log.get('status') != 'success':
        raise ValueError(f'{task}: log status must be success')
    if log.get('error') or log.get('invalidated'):
        raise ValueError(f'{task}: log contains an error or is invalidated')
    ev, results, samples = log['eval'], log['results'], log['samples']
    if not isinstance(samples, list) or len(samples) != expected:
        raise ValueError(f'{task}: actual completed sample count does not match {expected}')
    actual = _ids([s.get('id') for s in samples])
    declared = _ids(ev['dataset'].get('sample_ids'))
    if len(set(actual)) != expected or len(declared) != expected or set(actual) != set(declared):
        raise ValueError(f'{task}: unique actual and declared sample IDs must match exactly')
    if ev['dataset'].get('samples') != expected or results.get('total_samples') != expected or results.get('completed_samples') != expected:
        raise ValueError(f'{task}: declared/completed sample counts must equal {expected}')
    scores = results.get('scores', [])
    if len(scores) != 1:
        raise ValueError(f'{task}: exactly one scored aggregate is supported')
    aggregate = scores[0]
    if aggregate.get('scored_samples') != expected or aggregate.get('unscored_samples', 0) != 0:
        raise ValueError(f'{task}: every sample must be scored')
    scorer = aggregate['name']
    values = []
    for sample in samples:
        if sample.get('error') or sample.get('error_retries'):
            raise ValueError(f'{task}: sample {sample.get("id")} contains errors')
        values.append(_sample_value(sample['scores'][scorer]['value']))
    metrics = aggregate['metrics']
    mean, stderr = float(metrics['accuracy']['value']), float(metrics['stderr']['value'])
    if not math.isfinite(mean) or not math.isfinite(stderr) or not 0 <= mean <= 1 or not 0 <= stderr <= 1:
        raise ValueError(f'{task}: aggregate accuracy/stderr must be finite fractions')
    if abs(sum(values) / expected - mean) > 1e-10:
        raise ValueError(f'{task}: saved sample score mean disagrees with aggregate accuracy')
    if abs(float(triple[0]) - 100 * mean) > SCORE_TOLERANCE or abs(float(triple[1]) - 100 * stderr) > STDERR_TOLERANCE:
        raise ValueError(f'{task}: aggregate accuracy/stderr disagrees with card percentage triple')
    return {'expected': expected, 'actual': len(samples), 'unique': len(set(actual)),
            'scored': len(values), 'declared': len(declared), 'valid': True}


def validate_submission(files, *, dataset_variant=None, count_loader=None) -> dict:
    """Return JSON-serialisable validation evidence; all failures remain failures."""
    result = {'passed': False, 'errors': [], 'warnings': [], 'checks': {'card': False, 'dataset_counts': False, 'logs': 'not-performed'}, 'sample_details': {},
              'provenance': {'dataset_variant': None, 'dataset_variant_source': 'explicit' if dataset_variant else 'inferred',
                             'validation_dataset_revision': None, 'evaluation_dataset_revision': None,
                             'evidence_level': 'card-only', 'score_scale': SCORE_SCALE}}
    try:
        paths = [Path(p) for p in files]
        cards = [p for p in paths if p.suffix == '.parquet']
        logs = [p for p in paths if p.name.endswith(('.json', '.json.gz'))]
        if len(paths) != len(cards) + len(logs):
            raise ValueError('Only a model card and JSON/JSON.gz logs are supported')
        if len(cards) != 1:
            raise ValueError('Exactly one model card is required')
        df = read_card(cards[0]); row = df.iloc[0]
        result['checks']['card'] = True
        actual_counts = {t: int(row[c][2]) for t,c in TASKS.items()}
        loader = count_loader or fetch_expected_sample_counts
        if dataset_variant and dataset_variant not in ('full', 'lite'):
            raise ValueError('dataset variant must be full or lite')
        matches = []
        for variant in ([dataset_variant] if dataset_variant else ['full', 'lite']):
            revision, expected = loader(variant)
            if not re.fullmatch(r'[a-f0-9]{40}', revision or '') or set(expected) != set(TASKS) or any(not isinstance(n,int) or n <= 0 for n in expected.values()):
                raise ValueError(f'{variant}: incomplete pinned dataset counts')
            if actual_counts == expected:
                matches.append((variant, revision, expected))
        if len(matches) != 1:
            raise ValueError('All seven card counts must unambiguously match the pinned Full or Lite dataset')
        variant, revision, expected = matches[0]
        result['provenance'].update(dataset_variant=variant, validation_dataset_revision=revision,
                                    evaluation_date=str(row['date']))
        result['checks']['dataset_counts'] = True
        result['sample_details'] = {t:{'expected':n,'card_count':actual_counts[t],'valid':True} for t,n in expected.items()}
        if logs:
            result['checks']['logs'] = False
            result['provenance']['evidence_level'] = 'card-and-logs'
            by_task = {}; models = set(); flags = set(); revisions = set()
            for path in logs:
                log = read_log(path)
                ev = log.get('eval', {})
                task = ev.get('task', '').removeprefix('evals/')
                if task not in TASKS or task in by_task:
                    raise ValueError(f'{path.name}: unknown or duplicate benchmark task {task}')
                by_task[task] = log
                model = ev.get('model')
                if not isinstance(model,str) or not model:
                    raise ValueError(f'{task}: model is missing')
                models.add(model)
                flag = ev.get('task_args', {}).get('full')
                if not isinstance(flag,bool):
                    raise ValueError(f'{task}: explicit boolean full flag required')
                flags.add(flag)
                revisions.add(ev.get('dataset',{}).get('revision'))
            if set(by_task) != set(TASKS):
                raise ValueError('Exactly one log for each of the seven benchmarks is required')
            if len(models) != 1 or flags != {variant == 'full'}:
                raise ValueError('All logs must have a consistent model and full flag matching the card variant')
            display, provider = row['model'].rsplit(' (', 1)
            canonical = lambda text: text.casefold().replace('_', '-').replace(' ', '-')
            display, provider = canonical(display), canonical(provider[:-1])
            log_model = canonical(next(iter(models)).rsplit('/', 1)[-1])
            if log_model not in (display, provider + '-' + display):
                raise ValueError('Log model must exactly match the card display name or provider-prefixed display name')
            if len(revisions) != 1:
                raise ValueError('Evaluation dataset revisions are inconsistent')
            evaluation_revision = next(iter(revisions))
            if evaluation_revision is not None and not re.fullmatch(r'[a-f0-9]{40}', evaluation_revision):
                raise ValueError('Evaluation dataset revision must be an immutable SHA when present')
            result['provenance']['evaluation_dataset_revision'] = evaluation_revision
            result['provenance']['evaluation_model'] = next(iter(models))
            for task,log in by_task.items():
                result['sample_details'][task].update(validate_log(log, task, expected[task], row[TASKS[task]]))
            result['checks']['logs'] = True
        else:
            result['warnings'].append('Card-only evidence: log status, actual samples, identities and metrics were not inspected.')
        result['passed'] = True
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', nargs='+', help='Explicit card/log paths; do not interpolate submitted paths in a shell')
    group.add_argument('--manifest', type=Path)
    parser.add_argument('--dataset-variant', choices=['full', 'lite'])
    parser.add_argument('--output', type=Path, default=Path('validation_result.json'))
    args = parser.parse_args()
    files = args.files
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        files = [str(Path(manifest['data_dir']) / p) for p in manifest['paths']]
    result = validate_submission(files, dataset_variant=args.dataset_variant)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
