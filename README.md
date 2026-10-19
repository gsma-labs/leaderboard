# Open Telco Leaderboard Submissions

This repository stores reviewed evaluation submissions for the [Open Telco Leaderboard](https://huggingface.co/datasets/GSMA/leaderboard). The public score is the arithmetic mean of the seven benchmark accuracies, giving each benchmark equal weight.

Use the Open Telco CLI (`uv run open-telco`, then `run-evals` and `submit`) to create a submission PR. Include exactly one single-row card at `model_cards/<stem>.parquet`. Its optional Inspect logs belong in `trajectories/<stem>/` and may use `.json` or `.json.gz`. A card contains `model` as `name (Provider)`, the evaluation `date`, and a finite `[score, stderr, sample_count]` triple for every benchmark. Card scores and standard errors are percentages, including 0% and 1%; stored leaderboard pairs use fractions. The legacy `3gpp_tsg` card column maps to `three_gpp`.

## Dataset and evidence

All seven counts must match one variant. Validation resolves each public dataset's exact Hub SHA before reading its test configurations, and fails if any count cannot be read. Legacy cards without an explicit variant are accepted only when all seven counts unambiguously identify Full or Lite. The validator also supports `--dataset-variant full` or `lite`.

| Benchmark | Full (`GSMA/ot-full`) | Lite (`GSMA/ot-lite`) |
| --- | ---: | ---: |
| oranbench | 1,500 | 150 |
| srsranbench | 1,502 | 150 |
| telelogs | 864 | 100 |
| telemath | 500 | 100 |
| teleqna | 10,000 | 1,000 |
| teletables | 500 | 100 |
| three_gpp (dataset config `3gpp_tsg`) | 2,000 | 100 |

These counts describe the datasets at the time of this repair; validation reads the pinned live revisions. Card-only submissions remain supported and are labelled **card-only** evidence. Their log checks are explicitly **not performed**. When logs are supplied, all seven are required, with exactly one successful log per task. Validation checks actual completed/scored/unique samples, declared IDs, model and Full/Lite consistency, errors, saved numeric or C/I score means, and aggregate accuracy/stderr against the card. Card rounding permits at most 0.005001 percentage points for accuracy and 0.00000051 for stderr. Unsupported saved scoring fails explicitly. The log's final model basename must exactly match the display name, optionally prefixed by the credited provider; spaces and underscores are normalised to hyphens for this comparison. Provider credit always comes from the card.

Published rows carry JSON `submission_provenance`: PR number, reviewed head, merge SHA, evaluation date, per-task counts, evidence level, dataset variant and whether it was inferred or explicit. `validation_dataset_revision` records the dataset used by this check. `evaluation_dataset_revision` records an evaluation revision only if the logs actually declare one; legacy logs leave it null. Historic rows with unknown provenance stay null.

## Review, merge and publication

Every PR targeting `main` receives the exact-head commit status `submission-validation`. PRs changing no submission data receive a truthful not-applicable success. Trusted scripts from `main` use GitHub's paginated PR-file API and immutable Git tree to select the changed card and every matching log, including logs unchanged by the PR. Submitted code is never executed by the privileged workflows. Deletions, renames, ambiguous cards, unsafe paths and symlinks are rejected. The validation job uses only its scoped `GITHUB_TOKEN`, with no Hugging Face secret.

**Deployment prerequisite:** after deploying these workflows, configure branch protection for `main` to require the exact-head commit status context `submission-validation`. Preserve the existing required independent approval and dismiss-stale-review settings. Until this required status is configured, the premerge validation gate is not enforced. The publisher still revalidates every merged submission before writing to the Hub. Already-open PRs do not automatically receive a new status when this workflow is deployed. Before enabling the required status for the initial rollout, run **Actions → Validate Leaderboard Submission → Run workflow** with each already-open submission PR number. Use the same manual validation retry after deploying an updated validator, or whenever an existing PR needs a fresh check. This fetches the current immutable PR head from GitHub, requires base `main` before posting any status, and uses trusted `main` scripts without an HF secret. Normal PR events remain bound to their original event head. No close/reopen or contributor commit is needed.

A maintainer reviews the evidence and merges the PR. The merged-PR workflow then verifies the merge belongs to `main`, materialises data at that immutable merge SHA, and repeats validation. Publication reads the current Hub board at an exact SHA, merges only the selected model, computes averages/completion counts/ranks locally, and writes the parquet and README metadata together in one commit guarded by the parent Hub SHA. It preserves unrelated rows and files. Provider collisions and stale same-model updates fail. The workflow reads back that immutable resulting commit and verifies all fields and ranks before posting success or adding `synced-to-hf`. An identical retry verifies the existing commit without another upload.

The existing public Space refreshes its cached dataset independently. A verified Hub publication can therefore precede its appearance in the Space. This workflow does not redeploy or change the Space. The parquet identified in the dataset README is authoritative; historic CSV/JSON exports can remain unchanged.

To retry a failed or previously missed publication, open **Actions → Publish Merged Leaderboard Submission → Run workflow** and enter the merged PR number. All publishers share the fixed `leaderboard-publication` concurrency group with `queue: max` and `cancel-in-progress: false`. A PR changing no model card performs no Hub write. A later source for the same model prevents an older retry from replacing it. A Hub parent-commit conflict aborts and can be retried. If a GitHub comment or label fails after verified publication, `sync_report.json` keeps `verified: true` and the Hub commit SHA, with a separate retryable `reporting_errors` entry. The workflow reports the notification failure separately. Retrying verifies the existing publication and retries notifications without another data upload.

## Local checks

Use Python 3.11 or newer in an isolated environment:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r .github/scripts/requirements.txt
python -m pytest -q .github/scripts/
git diff --check
actionlint
```

Actionlint 1.7.12 predates GitHub's supported `concurrency.queue` property. With that version, run `actionlint -ignore 'unexpected key "queue" for "concurrency" section'` to ignore only that schema diagnostic. Keep all other workflow diagnostics enabled. The ordinary PR code-test workflow has read-only permissions and no privileged secrets. The orchestration command rejects `validate --dry-run`; validation retries post the exact-head status and report. `--dry-run` is supported only for publication.

Validate explicit local data paths (the output defaults to `validation_result.json`):

```sh
python .github/scripts/validate_submission.py --files model_cards/example.parquet trajectories/example/task.json.gz
```

The command requires a full seven-log set whenever any logs are supplied; omit logs for a card-only check. To inspect a real merged PR with no write token or mutation:

```sh
export GITHUB_TOKEN=YOUR_READ_TOKEN
python .github/scripts/submission_manifest.py --repo gsma-labs/leaderboard --pr 50 --mode publish --data-dir /tmp/leaderboard-pr50-data --output /tmp/leaderboard-pr50-manifest.json
python .github/scripts/sync_to_huggingface.py --manifest /tmp/leaderboard-pr50-manifest.json --dry-run --output /tmp/leaderboard-pr50-dry-run.json
```

Use a new, empty data directory for each materialisation. Dry runs read public pinned datasets/board and produce a proposed merge without `HF_TOKEN`, a Hub write, GitHub status/comment changes, or inference. Real publication requires `HF_TOKEN` and should run through the serialised merged-PR workflow. `calculate_avg_score.py` retains its reusable local computation function; its former standalone uploader is disabled.
