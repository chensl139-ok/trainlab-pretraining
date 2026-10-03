import json
import os
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from server.api import create_app
from server.manager import Manager
from server.schema import TrainConfig

TOKEN='test-only-token-'+'x'*40
AUTH={'Authorization':'Bearer '+TOKEN}

@pytest.fixture
def client(tmp_path,monkeypatch):
    cards={'available':True,'gpus':[{'index':i,'uuid':f'GPU-{i}','name':'test','memory_total_mib':98304,'memory_used_mib':0,'utilization_percent':0,'driver':'test'} for i in range(8)],'error':None}
    monkeypatch.setattr('server.api.gpu_inventory',lambda:cards)
    app=create_app(tmp_path,TOKEN,start_scheduler=False)
    with TestClient(app) as c:
        yield c


def upload(client):
    data='\n'.join(json.dumps({'text':f'document {i} text'}) for i in range(30))
    r=client.post('/api/datasets?name=example.jsonl',content=data,headers=AUTH)
    assert r.status_code==200,r.text
    return r.json()['id']


def config(did):
    return TrainConfig(dataset_id=did).model_dump()


def test_auth_and_static(client):
    assert client.get('/api/health').status_code==200
    assert client.get('/api/jobs').status_code==401
    assert client.get('/api/system',headers={'Authorization':'Bearer wrong'}).status_code==401
    assert client.get('/gpu.html').status_code==200
    assert client.get('/api/system',headers=AUTH).json()['gpus'][7]['index']==7


def test_invalid_dataset_cleanup(client):
    for text in ['bad json','\n'.join(json.dumps({'text':'ok','command':'bad'}) for _ in range(20))]:
        assert client.post('/api/datasets',content=text,headers=AUTH).status_code==422
    assert client.get('/api/datasets',headers=AUTH).json()==[]
    assert not list((client.app.state.manager.root/'datasets').glob('*.tmp'))


def test_queue_validation_and_cancel(client):
    did=upload(client)
    c=config(did)
    assert client.post('/api/jobs',json={**c,'gpu_ids':[0,0]},headers=AUTH).status_code==422
    assert client.post('/api/jobs',json={**c,'gpu_ids':[9]},headers=AUTH).status_code==422
    assert client.post('/api/jobs',json={**c,'heads':5},headers=AUTH).status_code==422
    assert client.post('/api/jobs',json={**c,'command':'rm'},headers=AUTH).status_code==422
    j=client.post('/api/jobs',json=c,headers=AUTH).json()
    assert j['status']=='queued'
    assert client.post('/api/jobs/'+j['id']+'/cancel',headers=AUTH).json()['status']=='cancelled'
    assert client.post('/api/jobs/'+j['id']+'/resume',headers=AUTH).status_code==409
    assert client.get('/api/jobs/invalid',headers=AUTH).status_code==404
    d=client.get('/api/jobs/'+j['id'],headers=AUTH).json()
    assert d['metrics']==[] and d['artifacts']==[]


def test_checkpoint_integrity_and_resume_idempotence(client):
    j=client.post('/api/jobs',json=config(upload(client)),headers=AUTH).json()
    client.post('/api/jobs/'+j['id']+'/cancel',headers=AUTH)
    cp=client.app.state.manager.jobdir(j['id'])/'output'/'checkpoint-25'
    cp.mkdir(parents=True)
    (cp/'trainer_state.json').write_text('{}')
    (cp/'optimizer.pt').write_text('test')
    assert client.post('/api/jobs/'+j['id']+'/resume',headers=AUTH).status_code==409
    (cp/'complete.json').write_text('{}')
    r1=client.post('/api/jobs/'+j['id']+'/resume',headers=AUTH).json()
    r2=client.post('/api/jobs/'+j['id']+'/resume',headers=AUTH).json()
    assert r1['id']==r2['id'] and r1['config']==j['config']


def test_download_cannot_escape_job_directory(client):
    j=client.post('/api/jobs',json=config(upload(client)),headers=AUTH).json()
    root=client.app.state.manager.jobdir(j['id'])/'output'
    root.mkdir()
    (root/'summary.json').write_text('{}')
    outside=client.app.state.manager.root/'secret.txt';outside.write_text('secret')
    (root/'leak').symlink_to(outside)
    assert client.get('/api/jobs/'+j['id']+'/artifact/summary.json',headers=AUTH).status_code==200
    assert client.get('/api/jobs/'+j['id']+'/artifact/leak',headers=AUTH).status_code==404
    assert client.get('/api/jobs/'+j['id']+'/artifact/%2e%2e/%2e%2e/secret.txt',headers=AUTH).status_code==404


def test_estimate_matches_formula(client):
    c=config(upload(client));c['gpu_ids']=list(range(8))
    e=client.post('/api/estimate',json=c,headers=AUTH).json()
    assert e['effective_batch']==128 and e['tokens_per_update']==32768


def test_commands_use_gpu_uuid_and_no_shell(tmp_path):
    m=Manager(tmp_path)
    cmd,env=m.command({'id':'a'*32,'config':{'gpu_ids':[7,2]}},[{'index':7,'uuid':'GPU-777'},{'index':2,'uuid':'GPU-222'}])
    assert '--nproc_per_node=2' in cmd
    assert env['CUDA_VISIBLE_DEVICES']=='GPU-777,GPU-222'
    assert 'TRAINLAB_API_TOKEN' not in env


def test_scheduler_process_failure_and_cancellation(tmp_path,monkeypatch):
    m=Manager(tmp_path)
    with m.db() as db:
        db.execute('INSERT INTO datasets VALUES (?,?,?,?,?,?)',('a'*32,'test',30,100,'hash','2020'))
    monkeypatch.setattr('server.manager.gpu_inventory',lambda:{'gpus':[]})
    monkeypatch.setattr(m,'command',lambda job,cards:([sys.executable,'-c','import time; time.sleep(30)'],os.environ.copy()))
    j=m.create(config('a'*32))
    m.start()
    try:
        for _ in range(50):
            if m.get(j['id'])['status']=='running':break
            time.sleep(.05)
        assert m.get(j['id'])['status']=='running'
        m.cancel(j['id'])
        for _ in range(50):
            if m.get(j['id'])['status']=='cancelled':break
            time.sleep(.05)
        assert m.get(j['id'])['status']=='cancelled'
        monkeypatch.setattr(m,'command',lambda job,cards:([sys.executable,'-c','raise SystemExit(3)'],os.environ.copy()))
        second=m.create(config('a'*32))
        for _ in range(50):
            if m.get(second['id'])['status']=='failed':break
            time.sleep(.05)
        assert m.get(second['id'])['status']=='failed'
    finally:m.close()


def test_restart_marks_running_interrupted_and_preserves_queued(tmp_path):
    m=Manager(tmp_path)
    with m.db() as db:
        db.execute('INSERT INTO datasets VALUES (?,?,?,?,?,?)',('a'*32,'test',30,100,'hash','2020'))
    j=m.create(config('a'*32))
    with m.db() as db:db.execute("UPDATE jobs SET status='running' WHERE id=?",(j['id'],))
    m.start()
    try:
        assert m.get(j['id'])['status']=='interrupted'
        other=Manager(tmp_path)
        with pytest.raises(RuntimeError):other.start()
    finally:m.close()


def test_submission_retry_does_not_duplicate_gpu_work(client):
    c=config(upload(client))
    headers={**AUTH,'Idempotency-Key':'a'*32}
    first=client.post('/api/jobs',json=c,headers=headers)
    second=client.post('/api/jobs',json=c,headers=headers)
    assert first.status_code==200 and first.json()['id']==second.json()['id']
    assert len(client.get('/api/jobs',headers=AUTH).json())==1
    assert client.post('/api/jobs',json={**c,'max_steps':200},headers=headers).status_code==422


def test_metrics_tolerate_partial_writes(client):
    j=client.post('/api/jobs',json=config(upload(client)),headers=AUTH).json()
    folder=client.app.state.manager.jobdir(j['id'])
    (folder/'metrics.jsonl').write_text('{"step":1,"loss":2.5}\n{"step":')
    (folder/'train.log').write_text('x'*200000)
    result=client.get('/api/jobs/'+j['id'],headers=AUTH).json()
    assert result['metrics']==[{'step':1,'loss':2.5}]
    assert len(client.get('/api/jobs/'+j['id']+'/logs',headers=AUTH).json()['text'])==128*1024


def test_document_split_deduplicates_without_leakage(tmp_path):
    from server.train import documents
    path=tmp_path/'data.jsonl'
    rows=[{'text':f'document {i}'} for i in range(100)]
    path.write_text('\n'.join(json.dumps(x) for x in rows+rows))
    a=list(documents(path,42));b=list(documents(path,42))
    assert a==b and len(a)==100
    train={t for t,s in a if s=='train'};val={t for t,s in a if s=='validation'}
    assert train and val and not train.intersection(val)
