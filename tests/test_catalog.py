import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from server.api import create_app
from server.catalog import CATALOG, CatalogError, collect, NoRedirect
from server.manager import ROOT

@pytest.fixture
def catalog_system(tmp_path):
    app=create_app(tmp_path,start_scheduler=False)
    keys={}
    for who,role,project in [('alice','operator','a'),('bob','operator','b'),('viewer','viewer','a')]:
        _,raw=app.state.identities.issue(who,role,project)
        keys[who]={'Authorization':'Bearer '+raw}
    with TestClient(app) as client:yield client,keys

def test_catalog_import_authorization_reuse_and_provenance(catalog_system):
    c,k=catalog_system
    assert c.get('/api/dataset-catalog').status_code==401
    assert len(c.get('/api/dataset-catalog',headers=k['viewer']).json())==4
    url='/api/dataset-catalog/sample/import'
    assert c.post(url,json={},headers=k['viewer']).status_code==403
    r=c.post(url,json={'documents':50},headers=k['alice']);assert r.status_code==200,r.text
    d=r.json();assert d['rows']==50 and not d['reused']
    assert c.post(url,json={'documents':50},headers=k['alice']).json()['id']==d['id']
    assert c.post(url,json={'documents':50},headers=k['alice']).json()['reused']
    assert c.post(url,json={'documents':50},headers=k['bob']).json()['id']!=d['id']
    path=c.app.state.manager.root/'datasets'/(d['id']+'.jsonl')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==d['sha256']
    manifest=c.get('/api/datasets/'+d['id']+'/source',headers=k['viewer']).json()
    assert manifest['documents']==50 and len(manifest['origins'])==50
    assert manifest['dataset_sha256']==d['sha256']
    assert c.get('/api/datasets/'+d['id']+'/source',headers=k['bob']).status_code==404
    preview=c.get('/api/datasets/'+d['id']+'/preview',headers=k['viewer']).json()
    assert preview['provenance']['source']['key']=='sample'
    assert 'origins' not in preview['provenance']
    assert c.post(url,json={'documents':100000},headers=k['alice']).status_code==422
    assert c.post(url,json={'documents':50,'url':'http://localhost'},headers=k['alice']).status_code==422
    assert c.post('/api/dataset-catalog/unknown/import',json={},headers=k['alice']).status_code==404

def test_import_failure_cleanup_and_disk_gate(catalog_system,monkeypatch):
    c,k=catalog_system
    def fail(*args):raise CatalogError('network unavailable')
    monkeypatch.setattr('server.api.collect',fail)
    r=c.post('/api/dataset-catalog/tinystories/import',json={},headers=k['alice'])
    assert r.status_code==502
    assert c.get('/api/datasets',headers=k['alice']).json()==[]
    assert not list((c.app.state.manager.root/'datasets').iterdir())
    c.app.state.manager.min_free_bytes=10**30
    assert c.post('/api/dataset-catalog/sample/import',json={},headers=k['alice']).status_code==507

def test_collect_keeps_document_boundaries_deduplicates_and_records_sources(monkeypatch):
    def rows(source,offset,length,deadline):
        data=[{'row_idx':i,'truncated_cells':[], 'row':{'text':f'Article {i}\n\nsecond paragraph','title':f'title {i}','url':'https://example.org/'+str(i)}} for i in range(25)]
        data+=[data[0],{'row':{'text':'cut'},'truncated_cells':['text']},{'row':{'text':''}}]
        return {'rows':data,'num_rows_total':28}
    monkeypatch.setattr('server.catalog.fetch_rows',rows)
    payload,manifest=collect(CATALOG[2],50,ROOT)
    docs=[json.loads(x) for x in payload.splitlines()]
    assert len(docs)==25 and manifest['skipped']==3
    assert '\n\nsecond paragraph' in docs[0]['text']
    assert manifest['origins'][0]['url']=='https://example.org/0'
    assert manifest['split']=='train'

def test_collect_rejects_too_few_documents_and_excess_size(monkeypatch):
    monkeypatch.setattr('server.catalog.fetch_rows',lambda *a:{'rows':[]})
    with pytest.raises(CatalogError,match='20'):collect(CATALOG[1],50,ROOT)
    monkeypatch.setattr('server.catalog.MAX_OUTPUT',10)
    with pytest.raises(CatalogError,match='20 MiB'):collect(CATALOG[0],50,ROOT)

def test_network_reader_has_bounded_responses_and_rejects_redirects(monkeypatch):
    from server.catalog import fetch_rows
    import time
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,size):return b'x'*size
    class Opener:
        def open(self,req,timeout):
            assert req.full_url.startswith('https://datasets-server.huggingface.co/rows?')
            assert not req.has_header('Authorization')
            return Response()
    monkeypatch.setattr('server.catalog.urllib.request.build_opener',lambda *a:Opener())
    monkeypatch.setattr('server.catalog.MAX_RESPONSE',100)
    with pytest.raises(CatalogError,match='单页过大'):fetch_rows(CATALOG[1],0,50,time.monotonic()+10)
    with pytest.raises(CatalogError,match='重定向'):NoRedirect().redirect_request(None,None,302,'',{},'http://127.0.0.1')
