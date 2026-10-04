import json
import os
import sys
import time
import pytest
from server.schema import TrainConfig,ModelTestConfig
from server.architectures import parameter_estimate
from test_security import system,upload,create


def completed(c,k):
    j=create(c,k['alice'],upload(c,k['alice'])).json();m=c.app.state.manager
    with m.db() as db:db.execute("UPDATE jobs SET status='succeeded' WHERE id=?",(j['id'],))
    path=m.jobdir(j['id'])/'output'/'final';path.mkdir(parents=True)
    for name in ('config.json','tokenizer.json','model.safetensors'):(path/name).write_text('{}')
    return j['id']


def test_architecture_validation_and_legacy_defaults():
    assert TrainConfig(dataset_id='a'*32).architecture=='gpt2'
    for arch in ('qwen3','qwen3_5'):
        c=TrainConfig(dataset_id='a'*32,architecture=arch)
        assert c.estimate()['estimated_parameters']>0
        for patch in ({'heads':6,'kv_heads':4},{'intermediate_size':128},{'hidden_size':130,'heads':2}):
            with pytest.raises(ValueError):TrainConfig(dataset_id='a'*32,architecture=arch,**patch)
    with pytest.raises(ValueError):TrainConfig(dataset_id='a'*32,architecture='qwen3_5',layers=2)
    with pytest.raises(ValueError):TrainConfig(dataset_id='a'*32,architecture='remote-code')
    for patch in ({'checkpoint':'../secret'},{'prompt':''},{'max_new_tokens':257},{'max_blocks':129},{'temperature':float('nan')},{'device':'auto'},{'max_runtime_seconds':601}):
        with pytest.raises(ValueError):ModelTestConfig(**({'prompt':'test'}|patch))


def test_model_test_auth_project_isolation_and_idempotence(system):
    c,k=system;jid=completed(c,k);url=f'/api/jobs/{jid}/tests';payload={'prompt':'Hello'}
    assert c.get('/api/architectures',headers=k['alice']).json()[0]['id']=='qwen3_5'
    assert c.post(url,json=payload).status_code==401
    assert c.post(url,json=payload,headers=k['viewer']).status_code==403
    assert c.post(url,json=payload,headers=k['bob']).status_code==404
    assert c.get(url,headers=k['bob']).status_code==404
    a=c.post(url,json=payload,headers={**k['alice'],'Idempotency-Key':'c'*32});assert a.status_code==200,a.text
    b=c.post(url,json=payload,headers={**k['alice'],'Idempotency-Key':'c'*32});assert a.json()['id']==b.json()['id']
    assert c.post(url,json={'prompt':'changed'},headers={**k['alice'],'Idempotency-Key':'c'*32}).status_code==422
    assert len(c.get(url,headers=k['viewer']).json())==1
    tid=a.json()['id']
    assert c.post(url+'/'+tid+'/cancel',headers=k['viewer']).status_code==403
    assert c.post(url+'/'+tid+'/cancel',headers=k['alice']).json()['status']=='cancelled'
    other=upload(c,k['bob'])
    assert c.post(url,json={'mode':'score','dataset_id':other},headers=k['alice']).status_code==404
    assert c.post(url,json={'mode':'score','dataset_id':other},headers=k['admin']).status_code==422


def test_models_must_be_complete_and_paths_are_fixed(system):
    c,k=system;jid=completed(c,k);url=f'/api/jobs/{jid}/tests';m=c.app.state.manager
    assert c.get('/api/jobs/'+jid,headers=k['alice']).json()['testable_models']==['final']
    assert c.post(url,json={'prompt':'test','checkpoint':'../../keys'},headers=k['alice']).status_code==422
    root=m.jobdir(jid)/'output';(root/'checkpoint-5').mkdir()
    for name in ('config.json','tokenizer.json','model.safetensors'):(root/'checkpoint-5'/name).write_text('{}')
    assert c.post(url,json={'prompt':'test','checkpoint':'checkpoint-5'},headers=k['alice']).status_code==422
    (root/'checkpoint-5'/'complete.json').write_text('{}')
    assert c.post(url,json={'prompt':'test','checkpoint':'checkpoint-5'},headers=k['alice']).status_code==200
    (root/'final'/'leak').symlink_to(m.root/'trainlab.sqlite3')
    assert c.post(url,json={'prompt':'test'},headers=k['alice']).status_code==422
    (root/'final'/'leak').unlink()
    with m.db() as db:db.execute("UPDATE jobs SET status='running' WHERE id=?",(jid,))
    assert c.post(url,json={'prompt':'test'},headers=k['alice']).status_code==422


def test_test_queue_disk_device_and_environment(system,monkeypatch):
    c,k=system;jid=completed(c,k);url=f'/api/jobs/{jid}/tests';m=c.app.state.manager
    assert c.post(url,json={'prompt':'x','device':'cuda','gpu_id':7},headers=k['alice']).status_code==422
    r=c.post(url,json={'prompt':'test'},headers=k['alice']);assert r.status_code==200
    m.max_pending=1
    assert c.post(url,json={'prompt':'another'},headers=k['alice']).status_code==422
    assert create(c,k['alice'],upload(c,k['alice']),'d'*32).status_code==422
    monkeypatch.setenv('OPENAI_API_KEY','must-not-leak');cmd,env=m.test_command(r.json(),[])
    assert env['CUDA_VISIBLE_DEVICES']=='' and 'OPENAI_API_KEY' not in env and 'server.evaluate' in cmd
    m.max_pending=20;m.min_free_bytes=10**30
    assert c.post(url,json={'prompt':'x'},headers=k['alice']).status_code==422


def test_scheduler_runs_tests_and_training_serially_and_persists_results(system,monkeypatch):
    c,k=system;jid=completed(c,k);m=c.app.state.manager
    t=c.post(f'/api/jobs/{jid}/tests',json={'prompt':'test'},headers=k['alice']).json()
    result=m.testdir(t['id'])/'result.json'
    payload=json.dumps({'mode':'generate','checkpoint':'final','prompt':'test','completion':'real child result','prompt_tokens':1,'generated_tokens':3,'generation_seconds':.1,'tokens_per_second':30})
    code='import time,pathlib;time.sleep(.2);pathlib.Path('+repr(str(result))+').write_text('+repr(payload)+')'
    monkeypatch.setattr(m,'test_command',lambda j,cards:([sys.executable,'-c',code],{}))
    m.start()
    try:
        for _ in range(80):
            if m.get_test(t['id'])['status']=='succeeded':break
            time.sleep(.05)
        done=m.get_test(t['id']);assert done['status']=='succeeded';assert done['result']['completion']=='real child result'
        assert m.thread.is_alive()
        # A running test uses the same slot: later training remains queued.
        monkeypatch.setattr(m,'test_command',lambda j,cards:([sys.executable,'-c','import time;time.sleep(20)'],{}))
        t2=c.post(f'/api/jobs/{jid}/tests',json={'prompt':'second'},headers=k['alice']).json()
        for _ in range(50):
            if m.get_test(t2['id'])['status']=='running':break
            time.sleep(.05)
        new=create(c,k['alice'],upload(c,k['alice']),'f'*32).json()
        assert m.get(new['id'])['status']=='queued'
        assert m.cancel_test(t2['id'])['status'] in ('cancelling','cancelled')
        m.cancel(new['id'])
    finally:m.close()


def test_original_training_dataset_cannot_be_relabeled_as_external_test(system):
    c,k=system;jid=completed(c,k);m=c.app.state.manager
    did=m.get(jid)['config']['dataset_id']
    r=c.post(f'/api/jobs/{jid}/tests',json={'mode':'score','dataset_id':did},headers=k['alice'])
    assert r.status_code==422 and '相同' in r.text


def test_invalid_test_result_is_rejected():
    from server.model_tests import validate_result
    config={'mode':'score','checkpoint':'final'}
    for result in ([],{}, {'mode':'score','checkpoint':'final'}, {'mode':'score','checkpoint':'final','loss':float('nan'),'blocks':1,'sequence_length':64,'evaluated_tokens':63}):
        with pytest.raises(ValueError):validate_result(result,config)
