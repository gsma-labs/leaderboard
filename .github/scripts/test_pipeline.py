"""Offline regressions for the trusted submission/publication boundary."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import registry
import validate_submission as validator
import sync_to_huggingface as publisher

TASKS = dict(registry.BENCHMARK_HF_COLUMNS)
FULL = dict(zip(['oranbench','srsranbench','telelogs','telemath','teleqna','teletables','three_gpp'], [1500,1502,864,500,10000,500,2000]))
LITE = dict(zip(FULL, [150,150,100,100,1000,100,100]))


def counts(variant):
    return ('a' * 40, FULL if variant == 'full' else LITE)


def card(tmp_path, score=50., counts_=LITE):
    path = tmp_path / 'model_cards' / 'selected.parquet'
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame([{'model':'selected (Provider)', 'date':'2026-10-14', **{k:[score,0.,n] for k,n in counts_.items()}}]).to_parquet(path,index=False)
    return path


def logs(tmp_path, counts_=LITE, *, compressed=True):
    paths=[]
    for task,n in counts_.items():
        path = tmp_path / 'trajectories' / 'selected' / (task + ('.json.gz' if compressed else '.json'))
        path.parent.mkdir(parents=True,exist_ok=True)
        d={'status':'success','eval':{'task':'evals/'+task,'model':'provider/selected','task_args':{'full':counts_ == FULL},'dataset':{'samples':n,'sample_ids':list(range(n))}},'results':{'total_samples':n,'completed_samples':n,'scores':[{'name':'choice','scored_samples':n,'unscored_samples':0,'metrics':{'accuracy':{'value':0.5},'stderr':{'value':0.}}}]},'samples':[{'id':i,'scores':{'choice':{'value': 'C' if i % 2 else 'I'}}} for i in range(n)]}
        with (gzip.open(path,'wt') if compressed else path.open('w')) as f: json.dump(d,f)
        paths.append(path)
    return paths


def validate(files, **kw):
    assert hasattr(validator, 'validate_submission'), 'validator must expose reusable fail-closed validation'
    return validator.validate_submission(files, count_loader=counts, **kw)


def test_full_card_infers_pinned_full_without_inventing_evaluation_revision(tmp_path):
    r=validate([card(tmp_path,counts_=FULL)])
    assert r['passed'], r['errors']
    assert r['provenance']['dataset_variant']=='full'
    assert r['provenance']['dataset_variant_source']=='inferred'
    assert r['provenance']['validation_dataset_revision']=='a'*40
    assert r['provenance']['evaluation_dataset_revision'] is None
    assert r['provenance']['evidence_level']=='card-only'
    assert r['checks']['logs']=='not-performed'


def test_compressed_success_logs_are_actually_inspected(tmp_path):
    paths=[card(tmp_path),*logs(tmp_path)]
    r=validate(paths)
    assert r['passed'],r['errors']
    assert r['provenance']['evidence_level']=='card-and-logs'
    assert len(r['sample_details'])==7
    p=paths[1]
    with gzip.open(p,'rt') as f: d=json.load(f)
    d['status']='cancelled'
    with gzip.open(p,'wt') as f: json.dump(d,f)
    r=validate(paths)
    assert not r['passed']
    assert any('success' in x for x in r['errors'])


@pytest.mark.parametrize('change', ['missing','null','nan','negative','fractional_count','extra_row'])
def test_malformed_card_fails(tmp_path,change):
    p=card(tmp_path);df=pd.read_parquet(p)
    if change=='missing': df=df.drop(columns=['telelogs'])
    elif change=='null':df.at[0,'telelogs']=None
    elif change=='nan':df.at[0,'telelogs']=[float('nan'),0.,100.]
    elif change=='negative':df.at[0,'telelogs']=[-1.,0.,100.]
    elif change=='fractional_count':df.at[0,'telelogs']=[50.,0.,100.5]
    else:df=pd.concat([df,df])
    df.to_parquet(p,index=False)
    assert not validate([p])['passed']


def test_zero_scores_and_percent_one_are_not_scale_guessed(tmp_path):
    assert validate([card(tmp_path,score=0.)])['passed']
    assert publisher.normalize_submission(pd.read_parquet(card(tmp_path,score=1.))).iloc[0]['telelogs']=='[0.01, 0.0]'


def test_dataset_failure_is_a_validation_failure(tmp_path):
    def fail(_):raise OSError('network unavailable')
    assert hasattr(validator,'validate_submission')
    r=validator.validate_submission([card(tmp_path)],count_loader=fail)
    assert not r['passed']
    assert any('network unavailable' in x for x in r['errors'])


@pytest.mark.parametrize('change',['duplicate_id','missing_sample','sample_error','wrong_mean','unsupported_score','wrong_stderr','model','full','declared_ids','duplicate_task','missing_task'])
def test_logs_require_actual_complete_consistent_scored_samples(tmp_path,change):
    paths=logs(tmp_path);p=paths[0]
    with gzip.open(p,'rt') as f:d=json.load(f)
    if change=='duplicate_id':d['samples'][-1]['id']=0
    elif change=='missing_sample':d['samples'].pop()
    elif change=='sample_error':d['samples'][0]['error']={'message':'failed'}
    elif change=='wrong_mean':d['samples'][0]['scores']['choice']['value']='C'
    elif change=='unsupported_score':d['samples'][0]['scores']['choice']['value']={'x':1}
    elif change=='wrong_stderr':d['results']['scores'][0]['metrics']['stderr']['value']=0.1
    elif change=='model':d['eval']['model']='another/model'
    elif change=='full':d['eval']['task_args']['full']=True
    elif change=='declared_ids':d['eval']['dataset']['sample_ids'][-1]='wrong'
    elif change=='duplicate_task':paths.append(p)
    elif change=='missing_task':paths.pop()
    with gzip.open(p,'wt') as f:json.dump(d,f)
    r=validate([card(tmp_path),*paths])
    assert not r['passed'],change


def board():
    return pd.DataFrame([{'model':'old','provider':'Other','source':'original','submission_provenance':'historic',**{t:'[0.6, 0.01]' for t in TASKS},'average':.6,'benchmarks_completed':7,'rank':1}])


def test_hf_load_failure_does_not_become_empty_board(monkeypatch):
    from huggingface_hub import HfApi
    monkeypatch.setattr(HfApi,'dataset_info',lambda *a,**k: (_ for _ in ()).throw(OSError('unavailable')))
    with pytest.raises(Exception):publisher.load_existing_dataset(None)


def test_same_name_other_provider_is_not_overwritten():
    with pytest.raises(ValueError,match='provider'):
        publisher.merge_submissions(board(),pd.DataFrame([{'model':'old','provider':'Imposter',**{t:'[0.9, 0.0]' for t in TASKS}}]))


def test_manifest_and_publisher_apis_exist():
    import importlib.util
    assert importlib.util.find_spec('submission_manifest'), 'trusted immutable-data manifest is required'
    assert hasattr(publisher,'publish_submission'), 'atomic publisher entrypoint is required'

class MemoryHub:
    """Hub boundary double: immutable snapshots and one compare-and-swap write."""
    def __init__(self,df=None):
        self.df=board() if df is None else df
        self.sha='b'*40;self.writes=[];self.corrupt=False;self.fail=False
        self.readme='---\nlicense: apache-2.0\ndataset_info:\n  features: []\n  splits: [{name: train, num_examples: 1}]\n---\nOriginal prose.\n'
    def snapshot(self, revision=None):
        if self.fail:raise OSError('cannot read board')
        df=self.df.copy(deep=True)
        if self.corrupt and revision:df.loc[0,'teleqna']='[0.1, 0.0]'
        return SimpleNamespace(sha=self.sha,frame=df,readme=self.readme,parquet_path='data/train-00000-of-00001.parquet')
    def commit(self,operations,parent_commit):
        assert parent_commit==self.sha
        self.writes.append(operations)
        import io
        self.df=pd.read_parquet(io.BytesIO(operations['data/train-00000-of-00001.parquet']))
        self.readme=operations['README.md'].decode()
        assert self.df['rank'].notna().all(), 'ranks must be computed before upload'
        self.sha='c'*40
        return self.sha


def manifest(tmp_path):
    p=card(tmp_path)
    return {'applicable':True,'repo':'owner/repo','pr_number':50,'reviewed_head_sha':'d'*40,'merge_sha':'e'*40,'merged_at':'2026-10-15T00:00:00Z','data_dir':str(tmp_path),'paths':[str(p.relative_to(tmp_path))],'card':str(p.relative_to(tmp_path)),'mode':'publish'}


def publish(m,hub,**kw):
    assert hasattr(publisher,'publish_submission')
    return publisher.publish_submission(m,hub=hub,count_loader=counts,is_ancestor=lambda a,b: True,**kw)


def test_publisher_one_commit_final_ranks_preserves_rows_and_null_history(tmp_path):
    hub=MemoryHub();before=hub.df.iloc[0].copy()
    report=publish(manifest(tmp_path),hub)
    assert report['verified'] and len(hub.writes)==1
    assert set(hub.writes[0])=={'data/train-00000-of-00001.parquet','README.md'}
    old=hub.df[hub.df.model=='old'].iloc[0]
    for c in before.index:assert old[c]==before[c]
    new=hub.df[hub.df.model=='selected'].iloc[0]
    provenance=json.loads(new['submission_provenance'])
    assert provenance['merge_sha']=='e'*40 and provenance['evaluation_dataset_revision'] is None
    assert provenance['evidence_level']=='card-only'
    assert 'Original prose.' in hub.readme and 'submission_provenance' in hub.readme


def test_publisher_dry_run_no_write_token_and_no_mutation(tmp_path):
    hub=MemoryHub();r=publish(manifest(tmp_path),hub,dry_run=True)
    assert r['dry_run'] and not r['verified'] and not hub.writes


def test_publisher_idempotent_retry_without_fresh_timestamp(tmp_path):
    m=manifest(tmp_path);hub=MemoryHub()
    publish(m,hub);r=publish(m,hub)
    assert r['verified'] and r['no_op'] and len(hub.writes)==1


@pytest.mark.parametrize('problem',['read_failure','empty','bad_card','readback'])
def test_publisher_failures_do_not_report_success(tmp_path,problem):
    m=manifest(tmp_path);hub=MemoryHub()
    if problem=='read_failure':hub.fail=True
    elif problem=='empty':hub.df=hub.df.iloc[:0]
    elif problem=='bad_card':Path(m['data_dir'],m['card']).write_bytes(b'not parquet')
    elif problem=='readback':hub.corrupt=True
    with pytest.raises(Exception):publish(m,hub)
    if problem!='readback':assert not hub.writes


def test_stale_same_model_update_refused(tmp_path):
    m=manifest(tmp_path);hub=MemoryHub();publish(m,hub)
    m['merge_sha']='f'*40
    with pytest.raises(ValueError,match='stale|ancestry'):
        publisher.publish_submission(m,hub=hub,count_loader=counts,is_ancestor=lambda a,b:False)
    assert len(hub.writes)==1

class GitHubFixture:
    def __init__(self):
        self.repo='owner/repo'
        self.pr={'number':50,'head':{'sha':'d'*40},'base':{'ref':'main'},'merged':True,'merge_commit_sha':'e'*40,'merged_at':'2026-10-15T00:00:00Z'}
        self.files=[{'filename':'model_cards/selected.parquet','status':'added'}]
        self.tree=[{'path':'model_cards','type':'tree','mode':'040000','sha':'1'*40}, {'path':'trajectories','type':'tree','mode':'040000','sha':'2'*40}, {'path':'trajectories/selected','type':'tree','mode':'040000','sha':'3'*40}, {'path':'model_cards/selected.parquet','type':'blob','mode':'100644','sha':'4'*40},{'path':'model_cards/old.parquet','type':'blob','mode':'100644','sha':'5'*40},{'path':'trajectories/selected/unchanged-failed.json.gz','type':'blob','mode':'100644','sha':'6'*40}]
    def pull(self,n):return self.pr
    def changed_files(self,n):return self.files
    def tree_at(self,sha):self.requested_sha=sha;return self.tree
    def is_ancestor(self,a,b):return True


def build_manifest(api,mode='validate'):
    import importlib
    module=importlib.import_module('submission_manifest')
    return module.build_manifest(api,50,mode=mode)


def test_manifest_selects_only_changed_card_and_includes_unchanged_failed_log():
    api=GitHubFixture();m=build_manifest(api)
    assert m['card']=='model_cards/selected.parquet'
    assert m['paths']==['model_cards/selected.parquet','trajectories/selected/unchanged-failed.json.gz']
    assert api.requested_sha=='d'*40
    m=build_manifest(api,'publish')
    assert api.requested_sha=='e'*40 and m['reviewed_head_sha']=='d'*40


@pytest.mark.parametrize('path',['../model_cards/a.parquet','model_cards/../a.parquet','/model_cards/a.parquet','model_cards/x;echo.parquet','model_cards/x\n.parquet','model_cards/x.parquet/extra'])
def test_manifest_rejects_unsafe_paths(path):
    api=GitHubFixture();api.files=[{'filename':path,'status':'added'}]
    with pytest.raises(ValueError):build_manifest(api)


@pytest.mark.parametrize('change',['renamed','removed','multiple','symlink','symlink_parent','unmerged','wrong_base','stale_merge','unrelated_logs'])
def test_manifest_rejects_ambiguous_or_untrusted_submissions(change):
    api=GitHubFixture();mode='validate'
    if change in ('renamed','removed'):api.files[0]['status']=change
    elif change=='multiple':api.files.append({'filename':'model_cards/old.parquet','status':'modified'})
    elif change=='symlink':api.tree[3]['mode']='120000'
    elif change=='symlink_parent':api.tree[0]['mode']='120000'
    elif change=='unmerged':api.pr['merged']=False;mode='publish'
    elif change=='wrong_base':api.pr['base']['ref']='elsewhere';mode='publish'
    elif change=='stale_merge':api.is_ancestor=lambda a,b:False;mode='publish'
    elif change=='unrelated_logs':api.files.append({'filename':'trajectories/old/test.json','status':'added'})
    with pytest.raises(ValueError):build_manifest(api,mode)


def test_non_data_pr_is_truthful_not_applicable():
    api=GitHubFixture();api.files=[{'filename':'README.md','status':'modified'}]
    m=build_manifest(api)
    assert not m['applicable'] and m['paths']==[]


def test_github_file_pagination_does_not_drop_later_card():
    import submission_manifest as sm
    api=sm.GitHubAPI('owner/repo')
    pages=[]
    def request(path,data=None):
        pages.append(path)
        return [{'filename':f'other/{n}','status':'added'} for n in range(100)] if path.endswith('&page=1') else [{'filename':'model_cards/selected.parquet','status':'added'}]
    api.request=request
    assert len(api.changed_files(50))==101 and len(pages)==2


def test_materializer_rejects_symlink_destination(tmp_path):
    import submission_manifest as sm
    real=tmp_path/'real';real.mkdir();link=tmp_path/'link';link.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError):sm.materialize_manifest(GitHubFixture(),{'paths':[]},link)


def test_validator_rejects_card_log_model_mismatch(tmp_path):
    paths=logs(tmp_path)
    for p in paths:
        with gzip.open(p,'rt') as f:d=json.load(f)
        d['eval']['model']='provider/unrelated'
        with gzip.open(p,'wt') as f:json.dump(d,f)
    assert not validate([card(tmp_path),*paths])['passed']


def test_validation_reports_exact_head_not_applicable(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('run_submission_pipeline')
    import run_submission_pipeline as run
    api=GitHubFixture();api.files=[{'filename':'README.md','status':'modified'}];writes=[]
    api.request=lambda path,data=None:writes.append((path,data))
    r=run.run_validation(api,50,'d'*40,tmp_path/'data',tmp_path,count_loader=counts)
    statuses=[(p,d) for p,d in writes if p.startswith('statuses/')]
    assert [(p,d['state']) for p,d in statuses]==[('statuses/'+'d'*40,'pending'),('statuses/'+'d'*40,'success')]
    assert r['passed'] and r['evidence_level']=='not-applicable'


def test_validation_never_marks_changed_head_success(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('run_submission_pipeline')
    import run_submission_pipeline as run
    api=GitHubFixture();api.files=[];api.pr['head']['sha']='c'*40;writes=[]
    api.request=lambda path,data=None:writes.append((path,data))
    r=run.run_validation(api,50,'d'*40,tmp_path/'data',tmp_path,count_loader=counts)
    assert not r['passed']
    assert writes[-2][0]=='statuses/'+'d'*40 and writes[-2][1]['state']=='failure'


def test_readme_count_and_schema_match_board_without_replacing_prose():
    original='---\nlicense: apache-2.0\ndataset_info:\n  features: []\n---\nBenchmark scores for **84 models**.\n\nHuman-written notes.\n'
    updated=publisher.updated_readme(original,board(),'data/train.parquet',123)
    assert '**1 models**' in updated and '**84 models**' not in updated
    assert 'Human-written notes.' in updated


def test_hub_conflict_aborts_without_success_receipt(tmp_path):
    hub=MemoryHub()
    def conflict(operations,parent_commit):raise RuntimeError('409 parent commit conflict')
    hub.commit=conflict
    with pytest.raises(RuntimeError,match='409'):publish(manifest(tmp_path),hub)


def test_ranking_change_preserves_all_unrelated_fields(tmp_path):
    m=manifest(tmp_path);card(tmp_path,score=90.)
    hub=MemoryHub();before=hub.df.iloc[0].copy()
    publish(m,hub)
    old=hub.df[hub.df.model=='old'].iloc[0]
    assert old['rank']==2
    for key in before.index:
        if key!='rank':assert old[key]==before[key]


def test_null_historic_provenance_stays_null(tmp_path):
    hub=MemoryHub();hub.df=hub.df.drop(columns='submission_provenance')
    publish(manifest(tmp_path),hub)
    assert pd.isna(hub.df.loc[hub.df.model=='old','submission_provenance'].iloc[0])


def test_numeric_samples_and_uncompressed_logs_supported(tmp_path):
    paths=logs(tmp_path,compressed=False)
    for p in paths:
        d=json.loads(p.read_text())
        for sample in d['samples']:
            sample['scores']['choice']['value']=1. if sample['id']%2 else 0.
        p.write_text(json.dumps(d))
    assert validate([card(tmp_path),*paths],dataset_variant='lite')['passed']


def test_count_loader_resolves_revision_before_any_split_reads(monkeypatch):
    import datasets
    from huggingface_hub import HfApi
    calls=[]
    def info(self,repo,**kw):calls.append(('head',repo));return SimpleNamespace(sha='a'*40)
    def load(repo,**kw):
        assert calls[0]==('head','GSMA/ot-full')
        assert kw['revision']=='a'*40
        calls.append(('split',kw['name']))
        return [0]*FULL['three_gpp' if kw['name']=='3gpp_tsg' else kw['name']]
    monkeypatch.setattr(HfApi,'dataset_info',info)
    monkeypatch.setattr(datasets,'load_dataset',load)
    revision,expected=validator.fetch_expected_sample_counts('full')
    assert revision=='a'*40 and expected==FULL and len(calls)==8


def test_publish_log_only_pr_is_truthful_no_card_no_op():
    api=GitHubFixture()
    api.files=[{'filename':'trajectories/selected/unchanged-failed.json.gz','status':'modified'}]
    m=build_manifest(api,'publish')
    assert not m['applicable'] and m['paths']==[] and m['card'] is None
    with pytest.raises(ValueError,match='Exactly one'):
        build_manifest(api,'validate')


@pytest.mark.parametrize('failed_notification',['comment','label'])
def test_verified_receipt_survives_notification_failure_and_retry(tmp_path,monkeypatch,failed_notification):
    import run_submission_pipeline as run
    source=tmp_path/'source';source.mkdir()
    m=manifest(source)
    api=GitHubFixture();api.repo=m['repo']
    writes=[]
    def request(path,data=None):
        writes.append(path)
        if (failed_notification=='comment' and path.endswith('/comments')) or (failed_notification=='label' and path.endswith('/labels')):
            raise OSError('GitHub notification temporarily unavailable')
    api.request=request
    monkeypatch.setattr(run,'build_manifest',lambda *a,**k:m.copy())
    monkeypatch.setattr(run,'materialize_manifest',lambda api,manifest,directory:manifest)
    event=tmp_path/'event.json';event.write_text(json.dumps({'inputs':{'pr_number':'50'}}))
    monkeypatch.setenv('GITHUB_EVENT_PATH',str(event))
    monkeypatch.setenv('GITHUB_REPOSITORY','owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN','read-only-test-token')
    monkeypatch.setattr(run,'GitHubAPI',lambda *a:api)
    hub=MemoryHub()
    original=run.run_publication
    def with_test_boundaries(*a,**k):
        return original(*a,**k,hub=hub,count_loader=counts)
    monkeypatch.setattr(run,'run_publication',with_test_boundaries)
    output=tmp_path/'reports'
    monkeypatch.setattr('sys.argv',['run_submission_pipeline.py','publish','--output-dir',str(output)])
    try:run.main()
    except SystemExit as exc:assert exc.code in (0,1)
    except OSError:pass  # Inspect the receipt even when the current buggy CLI raises.
    receipt=json.loads((output/'sync_report.json').read_text())
    assert receipt['verified'] and receipt['hub_commit_sha']=='c'*40
    assert receipt['reporting_errors'] and not receipt['reporting_complete']
    assert len(hub.writes)==1
    # Notification retry revalidates/verifies the existing immutable publication.
    api.request=lambda path,data=None:writes.append(path)
    try:run.main()
    except SystemExit as exc:assert exc.code==0
    retried=json.loads((output/'sync_report.json').read_text())
    assert retried['verified'] and retried['no_op'] and retried['reporting_complete']
    assert not retried['reporting_errors'] and len(hub.writes)==1


@pytest.mark.parametrize('mode',['validate','publish'])
def test_non_main_pr_cannot_claim_exact_head_validation(mode):
    api=GitHubFixture();api.pr['base']['ref']='release'
    api.files=[{'filename':'README.md','status':'modified'}]
    with pytest.raises(ValueError,match='main'):
        build_manifest(api,mode)


def test_same_sha_release_readme_pr_never_reports_not_applicable_success(tmp_path):
    import run_submission_pipeline as run
    api=GitHubFixture();api.pr['base']['ref']='release'
    api.files=[{'filename':'README.md','status':'modified'}];writes=[]
    api.request=lambda path,data=None:writes.append((path,data))
    with pytest.raises(ValueError,match='main'):
        run.run_validation(api,50,'d'*40,tmp_path/'data',tmp_path,count_loader=counts)
    assert not any(path.startswith('statuses/') for path,data in writes)


@pytest.mark.parametrize('mode',['validate','publish'])
@pytest.mark.parametrize('renamed_data',['failed_log','card'])
def test_manifest_rejects_data_renamed_out_of_namespace(mode,renamed_data):
    api=GitHubFixture()
    if renamed_data=='failed_log':
        api.files.append({'filename':'archive/unchanged-failed.json.gz','previous_filename':'trajectories/selected/unchanged-failed.json.gz','status':'renamed'})
        api.tree=[entry for entry in api.tree if not entry['path'].startswith('trajectories/selected/')]
    else:
        api.files=[{'filename':'archive/selected.parquet','previous_filename':'model_cards/selected.parquet','status':'renamed'}]
    with pytest.raises(ValueError,match='status|rename'):
        build_manifest(api,mode)


def test_readme_sizes_are_arrow_data_bytes():
    from datasets import Dataset
    import yaml
    frame=board().astype(object)
    arrow_bytes=Dataset.from_pandas(frame,preserve_index=False).data.nbytes
    assert arrow_bytes!=int(frame.memory_usage(index=False,deep=True).sum())
    readme=publisher.updated_readme('---\nlicense: apache-2.0\n---\nNotes.\n',frame,'data/train.parquet',123)
    metadata=yaml.safe_load(readme.split('---',2)[1])
    assert metadata['dataset_info']['dataset_size']==arrow_bytes
    assert metadata['dataset_info']['splits'][0]['num_bytes']==arrow_bytes


def test_manual_validation_event_uses_current_immutable_api_head(tmp_path,monkeypatch):
    import run_submission_pipeline as run
    event=tmp_path/'event.json';event.write_text(json.dumps({'inputs':{'pr_number':'50'}}))
    monkeypatch.setenv('GITHUB_EVENT_PATH',str(event))
    monkeypatch.setenv('GITHUB_REPOSITORY','owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN','test-scoped-token')
    monkeypatch.delenv('HF_TOKEN',raising=False)
    api=GitHubFixture();api.pr['head']['sha']='c'*40
    api.files=[{'filename':'README.md','status':'modified'}];writes=[]
    api.request=lambda path,data=None:writes.append((path,data))
    monkeypatch.setattr(run,'GitHubAPI',lambda *a:api)
    monkeypatch.setattr('sys.argv',['run_submission_pipeline.py','validate','--data-dir',str(tmp_path/'data'),'--output-dir',str(tmp_path/'reports')])
    with pytest.raises(SystemExit) as stopped:run.main()
    assert stopped.value.code==0
    statuses=[(p,d['state']) for p,d in writes if p.startswith('statuses/')]
    assert statuses==[('statuses/'+'c'*40,'pending'),('statuses/'+'c'*40,'success')]
    result=json.loads((tmp_path/'reports'/'validation_result.json').read_text())
    assert result['reviewed_head_sha']=='c'*40 and result['evidence_level']=='not-applicable'


def test_manual_validation_checks_main_before_any_status(tmp_path,monkeypatch):
    import run_submission_pipeline as run
    event=tmp_path/'event.json';event.write_text(json.dumps({'inputs':{'pr_number':'50'}}))
    monkeypatch.setenv('GITHUB_EVENT_PATH',str(event))
    monkeypatch.setenv('GITHUB_REPOSITORY','owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN','test-scoped-token')
    api=GitHubFixture();api.pr['base']['ref']='release';writes=[]
    api.request=lambda path,data=None:writes.append((path,data))
    monkeypatch.setattr(run,'GitHubAPI',lambda *a:api)
    monkeypatch.setattr('sys.argv',['run_submission_pipeline.py','validate'])
    with pytest.raises(ValueError,match='main'):run.main()
    assert not writes


def test_validation_dry_run_is_rejected_before_reporting(monkeypatch):
    import run_submission_pipeline as run
    monkeypatch.delenv('GITHUB_EVENT_PATH',raising=False)
    monkeypatch.setattr('sys.argv',['run_submission_pipeline.py','validate','--dry-run'])
    with pytest.raises(SystemExit) as stopped:run.main()
    assert stopped.value.code==2


def test_normal_validation_event_keeps_event_head_binding(tmp_path,monkeypatch):
    import run_submission_pipeline as run
    event=tmp_path/'event.json';event.write_text(json.dumps({'pull_request':{'number':50,'head':{'sha':'d'*40}}}))
    monkeypatch.setenv('GITHUB_EVENT_PATH',str(event))
    monkeypatch.setenv('GITHUB_REPOSITORY','owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN','test-scoped-token')
    api=GitHubFixture();api.pr['head']['sha']='c'*40;api.files=[];writes=[]
    api.request=lambda path,data=None:writes.append((path,data))
    monkeypatch.setattr(run,'GitHubAPI',lambda *a:api)
    monkeypatch.setattr('sys.argv',['run_submission_pipeline.py','validate','--data-dir',str(tmp_path/'data'),'--output-dir',str(tmp_path/'reports')])
    with pytest.raises(SystemExit) as stopped:run.main()
    assert stopped.value.code==1
    statuses=[(p,d['state']) for p,d in writes if p.startswith('statuses/')]
    assert statuses==[('statuses/'+'d'*40,'pending'),('statuses/'+'d'*40,'failure')]
