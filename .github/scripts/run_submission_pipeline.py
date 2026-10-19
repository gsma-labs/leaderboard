#!/usr/bin/env python3
"""Trusted workflow orchestration. Forks supply data only, never executable code."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from submission_manifest import GitHubAPI, build_manifest, materialize_manifest
from sync_to_huggingface import publish_submission
from validate_submission import validate_submission

CONTEXT = 'submission-validation'


def status(api, sha, state, description):
    api.request(f'statuses/{sha}', {'state':state, 'context':CONTEXT, 'description':description[:140]})


def comment(api, number, body):
    api.request(f'issues/{number}/comments', {'body':body})


def run_validation(api, number, expected_head, data_dir, output_dir, *, count_loader=None):
    if not re.fullmatch(r'[a-f0-9]{40}', expected_head):
        raise ValueError('Expected PR head must be an immutable commit SHA')
    # Commit statuses are shared by SHA across PRs. A release-targeted PR must
    # not overwrite main's context, even with pending/failure or the same head.
    if api.pull(number)['base']['ref'] != 'main':
        raise ValueError('Only PRs targeting main may report submission-validation')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status(api, expected_head, 'pending', 'Inspecting the immutable submission head')
    manifest = None
    try:
        manifest = build_manifest(api, number, mode='validate')
        if manifest['reviewed_head_sha'] != expected_head:
            raise ValueError('PR head changed while queued; this event cannot validate the new head')
        manifest = materialize_manifest(api, manifest, data_dir)
        (output_dir / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if manifest['applicable']:
            result = validate_submission([Path(manifest['data_dir']) / p for p in manifest['paths']], count_loader=count_loader)
            result['evidence_level'] = result['provenance']['evidence_level']
        else:
            result = {'passed':True, 'errors':[], 'warnings':[], 'checks':{'submission':'not-applicable'},
                      'sample_details':{}, 'evidence_level':'not-applicable'}
        result['reviewed_head_sha'] = expected_head
        result['logs_inspected'] = sum(p.endswith(('.json', '.json.gz')) for p in manifest['paths']) if result['passed'] else 0
        result['logs_selected'] = sum(p.endswith(('.json', '.json.gz')) for p in manifest['paths'])
    except Exception as exc:
        result = {'passed':False, 'errors':[f'{type(exc).__name__}: {exc}'], 'warnings':[], 'checks':{},
                  'sample_details':{}, 'evidence_level':'not-performed', 'reviewed_head_sha':expected_head,
                  'logs_inspected':0, 'logs_selected':len(manifest['paths']) - 1 if manifest and manifest['applicable'] else 0}
    (output_dir / 'validation_result.json').write_text(json.dumps(result, indent=2) + '\n')
    state = 'success' if result['passed'] else 'failure'
    description = 'Not applicable: no changed submission data' if result['evidence_level'] == 'not-applicable' else f"{result['evidence_level']}: validation {state}"
    status(api, expected_head, state, description)
    body = (f"Submission validation **{state}** for `{expected_head}`.\n\n"
            f"Evidence: **{result['evidence_level']}**; logs selected: {result['logs_selected']}; "
            f"logs fully inspected: {result['logs_inspected']}.\n\n")
    if result['errors']:
        body += 'Errors:\n' + '\n'.join('- ' + error for error in result['errors']) + '\n'
    if result['warnings']:
        body += '\n' + '\n'.join(result['warnings'])
    comment(api, number, body)
    return result


def run_publication(api, number, data_dir, output_dir, *, dry_run=False, hub=None, count_loader=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = materialize_manifest(api, build_manifest(api, number, mode='publish'), data_dir)
    (output_dir / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Publisher repeats all validation here, regardless of earlier status results.
    receipt = publish_submission(manifest, hub=hub, dry_run=dry_run, count_loader=count_loader, is_ancestor=api.is_ancestor)
    receipt.update(reporting_errors=[], reporting_complete=False)
    report_path = output_dir / 'sync_report.json'

    def save_receipt():
        report_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')

    # Persist verified Hub evidence before any fallible GitHub reporting call.
    save_receipt()
    notifications = []
    if not dry_run and receipt.get('verified'):
        body = (f"Published and verified at immutable Hub commit `{receipt['hub_commit_sha']}`.\n\n"
                f"Reviewed head: `{manifest['reviewed_head_sha']}`; merged source: `{manifest['merge_sha']}`. "
                f"Evidence: **{receipt['validation']['provenance']['evidence_level']}**. "
                'The existing Space will display it after its cached dataset refresh.')
        notifications = [
            ('comment', lambda: comment(api, number, body)),
            ('label', lambda: api.request(f'issues/{number}/labels', {'labels':['synced-to-hf']})),
        ]
    elif not dry_run:
        notifications = [('comment', lambda: comment(api, number,
            'Publication not applicable: this merged PR changed no model card. No Hub write was performed.'))]
    for operation, notify in notifications:
        try:
            notify()
        except Exception as exc:
            receipt['reporting_errors'].append({'operation':operation,
                'error':f'{type(exc).__name__}: {exc}', 'retryable':True})
        # Reporting failures cannot erase or relabel the verified Hub result.
        save_receipt()
    receipt['reporting_complete'] = not receipt['reporting_errors']
    save_receipt()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate','publish'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=Path('submitted-data'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    args = parser.parse_args()
    if args.mode == 'validate' and args.dry_run:
        parser.error('--dry-run is supported only for publication; validation reports statuses/comments')
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    api = GitHubAPI(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN'])
    if args.mode == 'validate':
        if 'pull_request' in event:
            number = int(event['pull_request']['number'])
            expected_head = event['pull_request']['head']['sha']
        else:
            number = int(event['inputs']['pr_number'])
            current_pr = api.pull(number)
            if current_pr['base']['ref'] != 'main':
                raise ValueError('Only PRs targeting main may report submission-validation')
            expected_head = current_pr['head']['sha']
        result = run_validation(api, number, expected_head, args.data_dir, args.output_dir)
        raise SystemExit(0 if result['passed'] else 1)
    number = int(event.get('inputs', {}).get('pr_number') or event['pull_request']['number'])
    try:
        receipt = run_publication(api, number, args.data_dir, args.output_dir, dry_run=args.dry_run)
        print(json.dumps({k:v for k,v in receipt.items() if k != 'proposed_rows'}, indent=2))
        if receipt.get('reporting_errors'):
            print('Hub publication result is preserved; GitHub notification failed and can be retried.')
            raise SystemExit(1)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'sync_report.json').write_text(json.dumps({'verified':False, 'errors':[f'{type(exc).__name__}: {exc}']}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
