#!/usr/bin/env python3
"""Select immutable submitted data through GitHub's API; never execute PR code."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

SHA = re.compile(r'[a-f0-9]{40}')
CARD = re.compile(r'model_cards/([A-Za-z0-9][A-Za-z0-9_.-]*)\.parquet')


class GitHubAPI:
    def __init__(self, repo, token=None):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise ValueError('Invalid repository identity')
        self.repo = repo
        self.token = token

    def request(self, path, data=None):
        headers = {'Accept':'application/vnd.github+json', 'X-GitHub-Api-Version':'2022-11-28'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        url = f'https://api.github.com/repos/{self.repo}/{path}'
        request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def pull(self, number):
        return self.request(f'pulls/{int(number)}')

    def changed_files(self, number):
        files = []
        for page in range(1, 32):
            batch = self.request(f'pulls/{int(number)}/files?per_page=100&page={page}')
            files.extend(batch)
            if len(batch) < 100:
                return files
        raise ValueError('PR file list exceeds GitHub supported pagination; refusing partial selection')

    def tree_at(self, sha):
        if not SHA.fullmatch(sha):
            raise ValueError('An immutable commit SHA is required')
        data = self.request(f'git/trees/{sha}?recursive=1')
        if data.get('truncated'):
            raise ValueError('Git tree is truncated; refusing partial log discovery')
        return data['tree']

    def is_ancestor(self, ancestor, descendant):
        if not SHA.fullmatch(ancestor) or not (SHA.fullmatch(descendant) or descendant == 'main'):
            raise ValueError('Invalid commit comparison')
        comparison = self.request(f'compare/{ancestor}...{descendant}')
        return comparison['status'] in ('ahead', 'identical') and comparison['merge_base_commit']['sha'] == ancestor

    def blob(self, sha):
        if not SHA.fullmatch(sha):
            raise ValueError('Invalid blob SHA')
        blob = self.request(f'git/blobs/{sha}')
        if blob.get('encoding') != 'base64':
            raise ValueError('Unsupported blob encoding')
        content = base64.b64decode(blob['content'])
        digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if digest != sha:
            raise ValueError('Downloaded data does not match the immutable Git blob')
        return content


def safe_path(path):
    if not isinstance(path,str) or not path or path.startswith('/') or '\\' in path:
        raise ValueError(f'Unsafe submitted path: {path!r}')
    parts = path.split('/')
    if any(p in ('', '.', '..') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+@=:-]*', p) for p in parts):
        raise ValueError(f'Unsafe submitted path: {path!r}')
    return path


def build_manifest(api, number, *, mode='validate'):
    if mode not in ('validate', 'publish'):
        raise ValueError('Unknown manifest mode')
    pr = api.pull(number)
    if pr['base']['ref'] != 'main':
        raise ValueError('Only PRs targeting main belong to this submission pipeline')
    head = pr['head']['sha']
    if not SHA.fullmatch(head):
        raise ValueError('PR head must be immutable')
    sha = head
    if mode == 'publish':
        if not pr.get('merged'):
            raise ValueError('Only PRs merged into main can publish')
        sha = pr.get('merge_commit_sha', '')
        if not SHA.fullmatch(sha) or not api.is_ancestor(sha, 'main'):
            raise ValueError('Merged submission is not in the main ancestry')
    files = api.changed_files(number)
    relevant = []
    for entry in files:
        # A rename can move a failed log/card outside the allowlist. Inspect both
        # sides before classifying it, otherwise it can disappear from evidence.
        changed_paths = [entry['filename'], entry.get('previous_filename')]
        data_paths = [path for path in changed_paths if path and (
            path.startswith(('model_cards', 'trajectories')) or
            '/model_cards/' in path or '/trajectories/' in path)]
        if data_paths:
            for path in data_paths:
                safe_path(path)
            if entry['status'] not in ('added', 'modified') or entry.get('previous_filename'):
                raise ValueError(f'Unsupported changed data status/rename {entry["status"]}: {entry["filename"]}')
            relevant.append(entry['filename'])
    cards = [p for p in relevant if p.startswith('model_cards/')]
    manifest = {'mode':mode, 'repo':api.repo, 'pr_number':int(number), 'reviewed_head_sha':head,
                'merge_sha':sha if mode == 'publish' else None, 'merged_at':pr.get('merged_at'),
                'applicable':bool(cards) if mode == 'publish' else bool(relevant), 'paths':[], 'blobs':{}, 'card':None}
    if not relevant or (mode == 'publish' and not cards):
        return manifest
    if len(cards) != 1 or not CARD.fullmatch(cards[0]):
        raise ValueError('Exactly one changed model_cards/<stem>.parquet is required')
    card = cards[0]; stem = CARD.fullmatch(card).group(1)
    prefix = f'trajectories/{stem}/'
    if any(p != card and (not p.startswith(prefix) or not p.endswith(('.json', '.json.gz'))) for p in relevant):
        raise ValueError('All changed data must belong to the selected card and its matching logs')
    tree = {e['path']: e for e in api.tree_at(sha)}
    paths = [card, *sorted(p for p in tree if p.startswith(prefix) and p.endswith(('.json', '.json.gz')))]
    for path in paths:
        safe_path(path)
        parts = PurePosixPath(path).parts
        for i in range(1, len(parts)):
            ancestor = '/'.join(parts[:i]); entry = tree.get(ancestor)
            if not entry or entry['type'] != 'tree' or entry['mode'] != '040000':
                raise ValueError(f'Symlink or missing/non-tree ancestor: {ancestor}')
        entry = tree.get(path)
        if not entry or entry['type'] != 'blob' or entry['mode'] not in ('100644', '100755'):
            raise ValueError(f'Symlink or non-regular submitted file: {path}')
        if not SHA.fullmatch(entry['sha']):
            raise ValueError('Invalid immutable blob SHA')
        manifest['blobs'][path] = entry['sha']
    manifest.update(card=card, paths=paths)
    return manifest


def materialize_manifest(api, manifest, directory):
    """Write only verified data blobs into a distinct empty directory."""
    root = Path(directory).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError('Materialisation directory cannot have symlink ancestors')
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Submitted data directory must be empty and separate from trusted scripts')
    for path in manifest['paths']:
        safe_path(path)
        output = root / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(api.blob(manifest['blobs'][path]))
    manifest['data_dir'] = str(root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--pr', required=True, type=int)
    parser.add_argument('--mode', choices=['validate','publish'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission_manifest.json'))
    args = parser.parse_args()
    api = GitHubAPI(args.repo, os.environ.get('GITHUB_TOKEN'))
    manifest = materialize_manifest(api, build_manifest(api, args.pr, mode=args.mode), args.data_dir)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
