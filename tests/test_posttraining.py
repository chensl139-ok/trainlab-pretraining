import json
import pytest
from server.schema import TrainConfig
from server.posttraining import record_format,encode_pair,model_fingerprint
from test_security import system,upload,create
from test_model_tests import completed


def post_data(c,auth,stage):
    rows=[{'prompt':f'Question {i}',**({'response':'yes'} if stage=='sft' else {'chosen':'yes','rejected':'no'})} for i in range(40)]
    r=c.post('/api/datasets?name='+stage+'.jsonl',content=''.join(json.dumps(x)+'\n' for x in rows),headers=auth)
    assert r.status_code==200,r.text
    assert r.json()['format']==stage
    return r.json()['id']


def test_formats_masking_and_no_silent_truncation():
    assert record_format({'text':'hello'})=='pretrain'
    for bad in ({'prompt':'a','response':''},{'prompt':'a','chosen':'x','rejected':'x'},{'messages':[]},{'text':3}):
        with pytest.raises(ValueError):record_format(bad)
    class Tokenizer:
        eos_token_id=1
        chat_template=None
        def encode(self,s,**kwargs):return list(s.encode())
    result=encode_pair(Tokenizer(),'hello','yes',128)
    assert result['labels'][-4:]==[121,101,115,1]
    assert all(x==-100 for x in result['labels'][:-4])
    with pytest.raises(ValueError):encode_pair(Tokenizer(),'hello','answer'*30,32)
    with pytest.raises(ValueError):TrainConfig(dataset_id='a'*32,stage='sft')
    with pytest.raises(ValueError):TrainConfig(dataset_id='a'*32,stage='pretrain',base_job_id='b'*32)
    with pytest.raises(ValueError):TrainConfig(dataset_id='a'*32,dpo_beta=float('nan'))


def test_posttraining_data_and_base_model_permissions(system):
    c,k=system;jid=completed(c,k);did=post_data(c,k['alice'],'sft')
    payload={'dataset_id':did,'stage':'sft','base_job_id':jid}
    assert c.post('/api/jobs',json=payload).status_code==401
    assert c.post('/api/jobs',json=payload,headers=k['viewer']).status_code==403
    assert c.post('/api/jobs',json=payload,headers=k['bob']).status_code==404
    bases=c.get('/api/base-models',headers=k['alice']).json();assert bases[0]['job_id']==jid
    assert c.get('/api/base-models',headers=k['bob']).json()==[]
    headers={**k['alice'],'Idempotency-Key':'e'*32}
    a=c.post('/api/jobs',json=payload,headers=headers);assert a.status_code==200,a.text
    assert c.post('/api/jobs',json=payload,headers=headers).json()['id']==a.json()['id']
    m=c.app.state.manager;runtime=json.loads((m.jobdir(a.json()['id'])/'config.json').read_text())
    assert runtime['base_model_sha256']==model_fingerprint(m.jobdir(jid)/'output'/'final')
    assert runtime['stage']=='sft' and runtime['base_model_path'].endswith('/final')
    assert c.post('/api/jobs',json={**payload,'stage':'dpo'},headers=k['alice']).status_code==422
    assert c.post('/api/jobs',json={'dataset_id':did},headers=k['alice']).status_code==422
    other=post_data(c,k['bob'],'sft')
    assert c.post('/api/jobs',json={**payload,'dataset_id':other},headers=k['admin']).status_code==422
    with m.db() as db:db.execute("UPDATE jobs SET status='running' WHERE id=?",(jid,))
    assert c.post('/api/jobs',json=payload,headers=k['alice']).status_code==422
    preview=c.get('/api/datasets/'+did+'/preview',headers=k['alice']).json()
    assert 'prompt' in preview['samples'][0]['text']
    assert next(d for d in c.get('/api/datasets',headers=k['alice']).json() if d['id']==did)['format']=='sft'


def test_posttraining_rejects_mixed_upload_and_wrong_test_format(system):
    c,k=system;jid=completed(c,k)
    r=c.post('/api/datasets',content='{"text":"a"}\n{"prompt":"b","response":"c"}\n',headers=k['alice'])
    assert r.status_code==422
    did=post_data(c,k['alice'],'dpo')
    assert c.post(f'/api/jobs/{jid}/tests',json={'mode':'score','dataset_id':did},headers=k['alice']).status_code==422


def test_dpo_loss_and_prompt_mask_math():
    torch=pytest.importorskip('torch')
    from server.posttraining import preference_loss,response_logps,collator
    ref=torch.tensor([-2.,-3.]);policy=ref.clone().requires_grad_(True)
    loss=preference_loss(policy,ref,.1).mean()
    assert loss.item()==pytest.approx(.693147,abs=1e-6)
    loss.backward();assert policy.grad[0]<0 and policy.grad[1]>0
    assert preference_loss(torch.tensor([-1.,-4.]),ref,.1)<loss
    logits=torch.zeros(1,4,3);labels=torch.tensor([[-100,-100,1,2]])
    assert response_logps(logits,labels).item()==pytest.approx(-2*torch.log(torch.tensor(3.)).item())
    batch=collator(0)([{'input_ids':[1,2,1],'labels':[-100,2,1]},{'input_ids':[1,2],'labels':[-100,2]}])
    assert batch['labels'][1,-1]==-100 and batch['attention_mask'][1,-1]==0


def test_imported_model_project_isolation_and_binding(system):
    c,k=system;m=c.app.state.manager;did=post_data(c,k['alice'],'sft')
    mid='f'*32;path=m.root/'models'/mid;path.mkdir()
    for name in ('config.json','tokenizer.json','model.safetensors'):(path/name).write_text('{}')
    cfg=TrainConfig(dataset_id=did).model_dump()
    from server.posttraining import MODEL_FIELDS
    cfg={key:cfg[key] for key in MODEL_FIELDS}
    with m.db() as db:
        db.execute('INSERT INTO base_models VALUES(?,?,?,?,?,?,?)',(mid,'alpha','fixture','Qwen/Qwen3-0.6B-Base','test',model_fingerprint(path),json.dumps(cfg)))
    # The fixture's project name follows the authenticated principal, not caller metadata.
    with m.db() as db:
        project=db.execute("SELECT project FROM resource_acl WHERE id=?",(did,)).fetchone()[0]
        db.execute('UPDATE base_models SET project=? WHERE id=?',(project,mid))
    assert c.get('/api/base-models',headers=k['bob']).json()==[]
    payload={'stage':'sft','dataset_id':did,'base_model_id':mid}
    r=c.post('/api/jobs',headers=k['alice'],json=payload);assert r.status_code==200,r.text
    assert r.json()['config']['base_model_id']==mid
    (path/'config.json').write_text('{"changed":true}')
    assert c.post('/api/jobs',headers=k['alice'],json=payload).status_code==422
    assert c.post('/api/jobs',headers=k['alice'],json={**payload,'base_job_id':'e'*32}).status_code==422


def test_import_lock_excludes_backup(tmp_path):
    import fcntl
    from server.manager import Manager
    from scripts.backup import backup
    m=Manager(tmp_path/'state')
    with (m.root/'import.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with pytest.raises(RuntimeError):backup(m.root,tmp_path/'snapshot')


def test_native_template_is_used_consistently():
    from server.posttraining import prompt_text
    class Tokenizer:
        chat_template='fixture'
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs=={'tokenize':False,'add_generation_prompt':True,'enable_thinking':False}
            return 'USER:'+messages[0]['content']+' ASSISTANT:'
    assert prompt_text(' hello ',Tokenizer())=='USER:hello ASSISTANT:'
