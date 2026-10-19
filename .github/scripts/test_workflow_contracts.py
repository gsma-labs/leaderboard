"""Semantic workflow contracts for trusted data handling and serial publication."""
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / 'workflows'


def load(name):
    # BaseLoader keeps the literal GitHub `on` key, rather than YAML 1.1 True.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def steps(workflow):
    return [step for job in workflow['jobs'].values() for step in job['steps']]


def test_publication_only_merged_or_manual_with_trusted_main():
    w=load('approval-sync.yaml')
    assert set(w['on'])=={'pull_request_target','workflow_dispatch'}
    assert w['on']['pull_request_target']['types']==['closed']
    assert w['on']['pull_request_target']['branches']==['main']
    assert w['on']['workflow_dispatch']['inputs']['pr_number']['required']=='true'
    job=w['jobs']['publish']
    assert job['if']=="github.event_name == 'workflow_dispatch' || github.event.pull_request.merged == true"
    checkouts=[s for s in steps(w) if s.get('uses','').startswith('actions/checkout@')]
    assert len(checkouts)==1 and checkouts[0]['with']['ref']=='main'
    assert checkouts[0]['with']['path']=='trusted'
    assert checkouts[0]['with']['persist-credentials']=='false'
    runs=[s['run'] for s in steps(w) if 'run' in s]
    assert sum('run_submission_pipeline.py publish' in command for command in runs)==1
    assert all('trusted/' in command for command in runs)
    text=(WORKFLOWS/'approval-sync.yaml').read_text()
    for obsolete in ('pull_request_review','pulls.merge','ot_hub','calculate_avg_score.py','push_to_hub','repository-dispatch'):
        assert obsolete not in text
    assert w['permissions']=={'contents':'read','pull-requests':'write'}


def test_all_publishers_use_one_fixed_queued_group():
    publishers=[]
    for path in WORKFLOWS.iterdir():
        if path.suffix not in ('.yaml','.yml'):continue
        w=load(path.name)
        if any('HF_TOKEN' in step.get('env',{}) for step in steps(w)):
            publishers.append(w)
    assert len(publishers)==1
    for w in publishers:
        assert w['concurrency']=={'group':'leaderboard-publication','queue':'max','cancel-in-progress':'false'}
        assert '${{' not in w['concurrency']['group']


def test_validation_all_prs_scoped_privileges_no_hf_secret_trusted_main():
    w=load('validate-submission.yaml')
    assert set(w['on'])=={'pull_request_target','workflow_dispatch'}
    assert w['on']['workflow_dispatch']['inputs']['pr_number']['required']=='true'
    assert w['on']['pull_request_target']=={'types':['opened','synchronize','reopened'],'branches':['main']}
    assert w['permissions']=={'contents':'read','statuses':'write','pull-requests':'write'}
    assert all('permissions' not in j for j in w['jobs'].values())
    checkouts=[s for s in steps(w) if s.get('uses','').startswith('actions/checkout@')]
    assert len(checkouts)==1 and checkouts[0]['with']=={'ref':'main','path':'trusted','persist-credentials':'false'}
    envs=[step.get('env',{}) for step in steps(w)]
    assert {'GITHUB_TOKEN':'${{ github.token }}'} in envs
    assert all(set(env) <= {'GITHUB_TOKEN'} for env in envs)
    text=(WORKFLOWS/'validate-submission.yaml').read_text()
    assert 'secrets.' not in text and 'HF_TOKEN' not in text
    runs=[s['run'] for s in steps(w) if 'run' in s]
    assert sum('run_submission_pipeline.py validate' in command for command in runs)==1
    assert all('trusted/' in command for command in runs)


def test_ordinary_code_tests_have_read_only_token_and_no_privileged_secrets():
    w=load('code-tests.yaml')
    assert set(w['on'])=={'pull_request'}
    assert w['permissions']=={'contents':'read'}
    assert all('permissions' not in j for j in w['jobs'].values())
    text=(WORKFLOWS/'code-tests.yaml').read_text()
    assert 'secrets.' not in text and 'HF_TOKEN' not in text
    assert any('pytest' in s.get('run','') for s in steps(w))
    assert all(s.get('with',{}).get('persist-credentials')=='false' for s in steps(w) if s.get('uses','').startswith('actions/checkout@'))


def test_manual_validation_artifact_name_is_available_for_both_triggers():
    w=load('validate-submission.yaml')
    artifacts=[step for step in steps(w) if step.get('uses','').startswith('actions/upload-artifact@')]
    assert len(artifacts)==1
    assert artifacts[0]['with']['name']=='submission-validation-${{ github.run_id }}'
    assert not any('HF_TOKEN' in step.get('env',{}) for step in steps(w))
