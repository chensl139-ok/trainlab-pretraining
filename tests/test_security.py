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
from server.security import AuthStore
from scripts.backup import backup,verify,restore

@pytest.fixture
def system(tmp_path,monkeypatch):
    monkeypatch.delenv('TRAINLAB_API_TOKEN',raising=False)
    monkeypatch.setenv('TRAINLAB_ENV','production')
    monkeypatch.setattr('server.api.gpu_inventory',lambda:{'available':True,'gpus':[{'index':0,'uuid':'GPU-0'}]})
    app=create_app(tmp_path,start_scheduler=False)
    keys={}
    for name,role,project in [('alice','operator','a'),('bob','operator','b'),('viewer','viewer','a'),('admin','admin','ops')]:
        kid,raw=app.state.identities.issue(name,role,project)
        keys[name]={'Authorization':'Bearer '+raw}
    with TestClient(app) as c:yield c,keys

def upload(c,h):
    data='\n'.join(json.dumps({'text':f'Doc {i}'}) for i in range(30))
    r=c.post('/api/datasets',content=data,headers=h)
    assert r.status_code==200,r.text
    return r.json()['id']

def create(c,h,did,key='a'*32):
    return c.post('/api/jobs',json=TrainConfig(dataset_id=did).model_dump(),headers={**h,'Idempotency-Key':key})

def test_project_and_role_enforcement(system):
    c,k=system;did=upload(c,k['alice']);j=create(c,k['alice'],did).json();jid=j['id']
    assert c.get('/api/datasets',headers=k['bob']).json()==[]
    assert c.get('/api/jobs',headers=k['bob']).json()==[]
    assert create(c,k['bob'],did).status_code==404
    for suffix in ('','/logs','/artifact/summary.json'):
        assert c.get('/api/jobs/'+jid+suffix,headers=k['bob']).status_code==404
    for action in ('cancel','resume'):
        assert c.post('/api/jobs/'+jid+'/'+action,headers=k['bob']).status_code==404
        assert c.post('/api/jobs/'+jid+'/'+action,headers=k['viewer']).status_code==403
    assert c.post('/api/datasets',content='x',headers=k['viewer']).status_code==403
    assert create(c,k['viewer'],did).status_code==403
    assert c.get('/api/jobs/'+jid,headers=k['viewer']).status_code==200
    assert c.get('/api/jobs/'+jid,headers=k['admin']).status_code==200
    other=upload(c,k['bob']);other_j=create(c,k['bob'],other).json()
    assert jid!=other_j['id']

def test_admin_resume_preserves_original_project(system):
    c,k=system;j=create(c,k['alice'],upload(c,k['alice'])).json();jid=j['id']
    c.post('/api/jobs/'+jid+'/cancel',headers=k['alice'])
    cp=c.app.state.manager.jobdir(jid)/'output'/'checkpoint-25';cp.mkdir(parents=True)
    for f in ('trainer_state.json','optimizer.pt','complete.json'):(cp/f).write_text('{}')
    r=c.post('/api/jobs/'+jid+'/resume',headers=k['admin']);assert r.status_code==200
    new=r.json()['id']
    assert c.get('/api/jobs/'+new,headers=k['alice']).status_code==200
    assert c.get('/api/jobs/'+new,headers=k['bob']).status_code==404
    assert c.post('/api/jobs/'+jid+'/resume',headers=k['alice']).json()['id']==new

def test_credentials_revocation_expiry_and_audit(system):
    c,k=system
    assert c.get('/api/me',headers=k['alice']).json()=={'subject':'alice','project':'a','role':'operator'}
    c.get('/api/jobs',headers={'Authorization':'Bearer secret-never-log'})
    with c.app.state.manager.db() as db:
        db.execute("UPDATE api_keys SET revoked=1 WHERE subject='alice'")
        db.execute("UPDATE api_keys SET expires=0 WHERE subject='bob'")
    assert c.get('/api/jobs',headers=k['alice']).status_code==401
    assert c.get('/api/jobs',headers=k['bob']).status_code==401
    audit=c.get('/api/audit',headers=k['admin']);assert audit.status_code==200
    assert 'secret-never-log' not in audit.text and k['admin']['Authorization'][7:] not in audit.text
    assert any(x['status']==401 for x in audit.json())
    assert c.get('/api/audit',headers=k['viewer']).status_code==403
    assert c.get('/api/metrics',headers=k['viewer']).status_code==403
    assert 'trainlab_scheduler_alive 0' in c.get('/api/metrics',headers=k['admin']).text
    assert c.get('/api/ready').status_code==503

def test_shared_token_rejected_in_production(tmp_path,monkeypatch):
    monkeypatch.setenv('TRAINLAB_ENV','production')
    with pytest.raises(RuntimeError,match='individual'):create_app(tmp_path,token='x'*64)

def test_queue_and_disk_guards(system):
    c,k=system;m=c.app.state.manager;did=upload(c,k['alice']);m.max_pending=1
    assert create(c,k['alice'],did).status_code==200
    assert create(c,k['alice'],did,'b'*32).status_code==422
    assert create(c,k['alice'],did).status_code==200 # retries do not consume quota
    m.min_free_bytes=10**30
    assert c.post('/api/datasets',content='x',headers=k['alice']).status_code==507
    assert not list((m.root/'datasets').glob('*.tmp'))

def test_training_environment_does_not_leak_credentials(tmp_path,monkeypatch):
    for key in ('AWS_SECRET_ACCESS_KEY','OPENAI_API_KEY','TRAINLAB_API_TOKEN','DATABASE_URL'):
        monkeypatch.setenv(key,'do-not-pass')
    cmd,env=Manager(tmp_path).command({'id':'a'*32,'config':{'gpu_ids':[0]}},[{'index':0,'uuid':'GPU-0'}])
    assert 'do-not-pass' not in env.values()
    assert env['HF_HUB_OFFLINE']=='1'

def test_runtime_deadline_terminates_process(tmp_path,monkeypatch):
    m=Manager(tmp_path)
    with m.db() as db:db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',('a'*32,'test',30,100,'hash','2020'))
    monkeypatch.setattr('server.manager.gpu_inventory',lambda:{'gpus':[]})
    monkeypatch.setattr(m,'command',lambda j,c:([sys.executable,'-c','import time;time.sleep(20)'],os.environ.copy()))
    cfg=TrainConfig(dataset_id='a'*32).model_dump();cfg['max_runtime_seconds']=.1
    j=m.create(cfg);m.start()
    try:
        for _ in range(80):
            if m.get(j['id'])['status']=='failed':break
            time.sleep(.05)
        assert m.get(j['id'])['status']=='failed'
        assert '运行时限' in m.get(j['id'])['error']
        assert m.thread.is_alive()
    finally:m.close()

def test_backup_restore_and_tampering(tmp_path):
    m=Manager(tmp_path/'state');auth=AuthStore(m);kid,raw=auth.issue('alice','operator','a')
    (m.root/'datasets'/'example.jsonl').write_text('{"text":"example"}')
    snap=tmp_path/'snapshot';backup(m.root,snap)
    verify(snap);m.root.rename(tmp_path/'original-state')
    restore(snap,tmp_path/'state')
    restored=Manager(tmp_path/'state');assert AuthStore(restored).authenticate(raw).subject=='alice'
    assert (restored.root/'datasets'/'example.jsonl').read_text()=='{"text":"example"}'
    with pytest.raises(ValueError,match='empty'):restore(snap,restored.root)
    (snap/'datasets'/'example.jsonl').write_text('corrupted')
    with pytest.raises(ValueError,match='integrity'):verify(snap)

def test_backup_rejects_running_service(tmp_path):
    m=Manager(tmp_path/'state');m.start()
    try:
        with pytest.raises(RuntimeError,match='Stop'):backup(m.root,tmp_path/'backup')
    finally:m.close()


def test_json_body_is_bounded(system):
    c,k=system
    r=c.post('/api/jobs',content='x'*70000,headers=k['alice'])
    assert r.status_code==413 and r.headers['X-Request-ID']
