import hashlib
import hmac
import json
import os
import re
import shutil
import time
import threading
import logging
import platform
import importlib.metadata
import psutil
import anyio
from pathlib import Path
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from server.manager import Manager, ROOT, gpu_inventory, now
from server.schema import TrainConfig, CatalogImport, ModelTestConfig, SchedulerControl
from server.catalog import CATALOG, collect, CatalogError, MAX_OUTPUT
from server.security import AuthStore, Principal
from server.architectures import ARCHITECTURES
from server.posttraining import record_format, MODEL_FIELDS, inherited_config

MAX_UPLOAD = 100*1024*1024
AUDIT_LOG = logging.getLogger('trainlab.audit')
if not AUDIT_LOG.handlers:
    AUDIT_LOG.addHandler(logging.StreamHandler())
AUDIT_LOG.setLevel(logging.INFO)
AUDIT_LOG.propagate = False

def create_app(state_dir=None, token=None, start_scheduler=True):
    secret = token if token is not None else os.environ.get('TRAINLAB_API_TOKEN', '')
    state = Path(state_dir or os.environ.get('TRAINLAB_STATE_DIR', ROOT/'state'))
    production = os.environ.get('TRAINLAB_ENV', 'development') == 'production'
    if production and secret:
        raise RuntimeError('Production requires individual credentials. Remove TRAINLAB_API_TOKEN and use scripts.users.')
    if secret and len(secret) < 32:
        raise RuntimeError('Development token must contain at least 32 characters.')
    manager = Manager(state)
    identities = AuthStore(manager)
    upload_slots = threading.BoundedSemaphore(2)
    import_slot = threading.Lock()
    with manager.db() as db:
        db.execute('CREATE TABLE IF NOT EXISTS dataset_sources (dataset_id TEXT PRIMARY KEY, project TEXT, catalog_key TEXT, selection INTEGER, manifest TEXT, UNIQUE(project,catalog_key,selection))')

    @asynccontextmanager
    async def lifespan(app):
        if start_scheduler:
            manager.start()
        yield
        if start_scheduler:
            manager.close()

    app = FastAPI(title='TrainLab self-hosted pretraining', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = manager

    app.state.identities = identities

    @app.middleware('http')
    async def bounded_json(request, call_next):
        if request.url.path.startswith('/api/') and request.method in ('POST','PUT','PATCH') and request.url.path != '/api/datasets':
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>64*1024:
                    return JSONResponse({'detail':'配置请求不能超过 64 KiB'},status_code=413)
            request._body = bytes(body)
        return await call_next(request)

    @app.middleware('http')
    async def security_headers_and_audit(request, call_next):
        rid = uuid.uuid4().hex
        status = 500
        started = time.monotonic()
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers.update({'X-Request-ID':rid,'X-Content-Type-Options':'nosniff',
                'X-Frame-Options':'DENY','Referrer-Policy':'no-referrer',
                'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'",
                'Cache-Control':'no-store' if request.url.path.startswith('/api/') else 'no-cache'})
            return response
        finally:
            if request.url.path.startswith('/api/') and request.url.path not in ('/api/health','/api/ready'):
                actor = getattr(request.state,'principal',Principal('anonymous','viewer',''))
                route = request.scope.get('route')
                route_path = getattr(route,'path','unmatched')
                target=request.path_params.get('jid','')
                if re.fullmatch(r'[a-f0-9]{32}',target):
                    route_path=route_path.replace('{jid}',target)
                def record():
                    with manager.db() as db:
                        db.execute('INSERT INTO audit_events(time,request_id,actor,project,method,path,status) VALUES(?,?,?,?,?,?,?)',
                            (now(),rid,actor.subject,actor.project,request.method,route_path,status))
                await anyio.to_thread.run_sync(record)
                AUDIT_LOG.info(json.dumps({'request_id':rid,'actor':actor.subject,
                    'method':request.method,'route':route_path,'status':status,'duration_ms':round((time.monotonic()-started)*1000)}))

    def auth(request: Request, authorization: str = Header(default='')):
        raw = authorization[7:] if authorization.startswith('Bearer ') else ''
        if secret and hmac.compare_digest(raw.encode(),secret.encode()):
            principal = Principal('development-admin','admin','default')
        else:
            principal = identities.authenticate(raw)
        request.state.principal = principal
        return principal

    def writer(p=Depends(auth)):
        p.write()
        return p

    def job(jid, p):
        identities.require(p,"job",jid)
        if not re.fullmatch(r'[a-f0-9]{32}', jid):
            raise HTTPException(404, '任务不存在')
        found = manager.get(jid)
        if not found:
            raise HTTPException(404, '任务不存在')
        return found

    @app.get('/api/health')
    def health():
        return {'service':'trainlab-pretraining','version':6,'auth_required':True}

    @app.get('/api/ready')
    def ready():
        healthy = bool(manager.thread and manager.thread.is_alive()) and shutil.disk_usage(state).free >= manager.min_free_bytes
        healthy = healthy and (manager.active_id is not None or manager.lease_available())
        return JSONResponse({'ready':healthy},status_code=200 if healthy else 503)

    @app.get('/api/me')
    def me(p=Depends(auth)):
        return {'subject':p.subject,'role':p.role,'project':p.project}

    @app.get('/api/audit')
    def audit(p=Depends(auth)):
        if p.role!='admin':
            raise HTTPException(403,'需要管理员权限')
        with manager.db() as db:
            return [dict(r) for r in db.execute('SELECT * FROM audit_events ORDER BY seq DESC LIMIT 200')]

    @app.get('/api/metrics',response_class=PlainTextResponse)
    def metrics(p=Depends(auth)):
        if p.role!='admin':
            raise HTTPException(403,'需要管理员权限')
        with manager.db() as db:
            counts = dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status').fetchall())
            test_counts = dict(db.execute('SELECT status,COUNT(*) FROM model_tests GROUP BY status').fetchall())
        lines = ['# TYPE trainlab_jobs gauge']
        for st in ('queued','running','cancelling','succeeded','failed','interrupted','cancelled'):
            lines.append('trainlab_jobs{status="'+st+'"} '+str(counts.get(st,0)))
        lines.append('# TYPE trainlab_model_tests gauge')
        for st in ('queued','running','cancelling','succeeded','failed','interrupted','cancelled'):
            lines.append('trainlab_model_tests{status="'+st+'"} '+str(test_counts.get(st,0)))
        lines += ['# TYPE trainlab_scheduler_paused gauge','trainlab_scheduler_paused '+str(int(manager.control()['paused'])),
            '# TYPE trainlab_scheduler_blocked gauge','trainlab_scheduler_blocked '+str(int(bool(manager.blocked_reason))),
            '# TYPE trainlab_worker_active gauge','trainlab_worker_active '+str(int(manager.active_id is not None))]
        lines += ['# TYPE trainlab_disk_free_bytes gauge','trainlab_disk_free_bytes '+str(shutil.disk_usage(state).free),
            '# TYPE trainlab_scheduler_alive gauge','trainlab_scheduler_alive '+str(int(bool(manager.thread and manager.thread.is_alive())))]
        return '\n'.join(lines)+'\n'

    @app.get('/api/system')
    def system(p=Depends(auth)):
        return {**gpu_inventory(), 'scheduler':'single_job_queue', 'max_upload_mib':100,
                'backend':'Qwen3.5 / Qwen3 / GPT-2 from scratch + byte-level BPE + PyTorch DDP',
                'active_job_id':manager.active_id if manager.active_kind=='jobs' and identities.allowed(p,'job',manager.active_id) else None,
                'identity':{'subject':p.subject,'role':p.role,'project':p.project},
                'limits':{'max_pending_jobs':manager.max_pending,'min_free_bytes':manager.min_free_bytes,'max_log_bytes':manager.max_log_bytes,'gpu_idle_mib':manager.gpu_idle_mib},
                'control':manager.control(),'blocked_reason':manager.blocked_reason,'scheduler_error':manager.scheduler_error,
                'worker_kind':manager.active_kind,'execution_blocked':manager.active_id is None and not manager.lease_available(),
                'environment':'production' if production else 'development',
                'scheduler_alive':bool(manager.thread and manager.thread.is_alive()),
                'disk':dict(zip(('total','used','free'),shutil.disk_usage(state)))}

    @app.post('/api/scheduler/control')
    def scheduler_control(config: SchedulerControl,p=Depends(auth)):
        if p.role!='admin':raise HTTPException(403,'仅管理员可改变维护模式')
        return manager.set_control(config.paused,config.reason.strip(),p.subject)

    @app.get('/api/operations')
    def operations(p=Depends(auth)):
        if p.role!='admin':raise HTTPException(403,'仅管理员可查看运维诊断')
        with manager.db() as db:
            integrity=db.execute('PRAGMA quick_check(1)').fetchone()[0]
            jobs=dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status').fetchall())
            tests=dict(db.execute('SELECT status,COUNT(*) FROM model_tests GROUP BY status').fetchall())
        versions={}
        for package in ('torch','transformers','fastapi','accelerate'):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]='未安装'
        return {'generated_at':now(),'environment':'production' if production else 'development',
            'service_version':6,'revision':os.environ.get('TRAINLAB_REVISION','local'),'platform':platform.system()+' '+platform.machine(),'python':platform.python_version(),'packages':versions,
            'started_at':manager.started_at,'scheduler_alive':bool(manager.thread and manager.thread.is_alive()),
            'scheduler_error':manager.scheduler_error,'control':manager.control(),'blocked_reason':manager.blocked_reason,
            'execution_blocked':manager.active_id is None and not manager.lease_available(),
            'active':{'id':manager.active_id,'kind':manager.active_kind},'jobs':jobs,'model_tests':tests,
            'database_integrity':integrity,'disk':dict(zip(('total','used','free'),shutil.disk_usage(state))),
            'host_memory':{'total':psutil.virtual_memory().total,'available':psutil.virtual_memory().available},
            'gpu':gpu_inventory(),'limits':{'min_free_bytes':manager.min_free_bytes,'max_pending':manager.max_pending,'max_log_bytes':manager.max_log_bytes,'gpu_idle_mib':manager.gpu_idle_mib},
            'scope':'服务自检；主机内存不是容器内存配额。未代替 CUDA/NCCL、模型质量、吞吐与恢复演练验收。'}

    @app.get('/api/architectures')
    def architectures(p=Depends(auth)):
        return ARCHITECTURES

    @app.get('/api/datasets')
    def datasets(p=Depends(auth)):
        with manager.db() as db:
            if p.role=='admin':
                rows=db.execute('SELECT * FROM datasets ORDER BY created_at DESC LIMIT 500')
            else:
                rows=db.execute("SELECT d.* FROM datasets d JOIN resource_acl a ON a.kind='dataset' AND a.id=d.id WHERE a.project=? ORDER BY d.created_at DESC LIMIT 500",(p.project,))
            items=[dict(r) for r in rows]
            formats=dict(db.execute('SELECT id,format FROM dataset_formats').fetchall())
            return [{**r,'format':formats.get(r['id'],'pretrain')} for r in items]

    @app.get('/api/datasets/{did}/preview')
    def preview_dataset(did: str, p=Depends(auth)):
        if not re.fullmatch(r'[a-f0-9]{32}',did):
            raise HTTPException(404,'数据集不存在')
        identities.require(p,'dataset',did)
        with manager.db() as db:
            row=db.execute('SELECT * FROM datasets WHERE id=?',(did,)).fetchone()
        if not row:
            raise HTTPException(404,'数据集不存在')
        samples=[]
        path=manager.root/'datasets'/(did+'.jsonl')
        if not path.is_file():
            raise HTTPException(409,'语料文件缺失，请检查存储或重新上传')
        # Inspect at most 64 physical lines; no full-corpus allocation.
        with path.open('rb') as src:
            for _ in range(64):
                line=src.readline(1024*1024+1)
                if not line:break
                if not line.strip():continue
                try:
                    item=json.loads(line)
                    kind=record_format(item)
                    text=item['text'] if kind=='pretrain' else json.dumps(item,ensure_ascii=False,indent=2)
                except (ValueError,KeyError,TypeError):
                    raise HTTPException(409,'语料文件已损坏，请检查备份')
                samples.append({'text':text[:800],'truncated':len(text)>800})
                if len(samples)==3:break
        with manager.db() as db:
            origin=db.execute('SELECT manifest FROM dataset_sources WHERE dataset_id=?',(did,)).fetchone()
        provenance=json.loads(origin[0]) if origin else None
        if provenance:provenance.pop('origins',None)
        return {'dataset':dict(row),'samples':samples,'preview_limit':3,'character_limit':800,'provenance':provenance}

    @app.get('/api/overview')
    def overview(p=Depends(auth)):
        project=None if p.role=='admin' else p.project
        with manager.db() as db:
            clause='' if project is None else " JOIN resource_acl a ON a.kind='job' AND a.id=j.id WHERE a.project=?"
            args=() if project is None else (project,)
            counts=dict(db.execute('SELECT j.status,COUNT(*) FROM jobs j'+clause+' GROUP BY j.status',args).fetchall())
            test_clause='' if project is None else " JOIN resource_acl a ON a.kind='job' AND a.id=t.job_id WHERE a.project=?"
            tests=dict(db.execute('SELECT t.status,COUNT(*) FROM model_tests t'+test_clause+' GROUP BY t.status',args).fetchall())
            ds_clause='' if project is None else " JOIN resource_acl a ON a.kind='dataset' AND a.id=d.id WHERE a.project=?"
            ds=db.execute('SELECT COUNT(*),COALESCE(SUM(d.bytes),0) FROM datasets d'+ds_clause,args).fetchone()
        return {'jobs':counts,'model_tests':tests,'total_jobs':sum(counts.values()),'dataset_count':ds[0],'dataset_bytes':ds[1]}

    @app.get('/api/jobs/page')
    def jobs_page(q: str=Query(default='',max_length=80), status: str=Query(default='',max_length=20),
                  offset: int=Query(default=0,ge=0,le=1000000), limit: int=Query(default=12,ge=1,le=100), p=Depends(auth)):
        if status and status not in ('queued','running','cancelling','succeeded','failed','interrupted','cancelled'):
            raise HTTPException(422,'任务状态无效')
        return manager.page_jobs(None if p.role=='admin' else p.project,q,status,offset,limit)

    @app.get('/api/dataset-catalog')
    def catalog(p=Depends(auth)):
        return CATALOG

    @app.post('/api/dataset-catalog/{key}/import')
    def import_catalog(key: str, config: CatalogImport, p=Depends(writer)):
        source=next((x for x in CATALOG if x['key']==key),None)
        if not source:raise HTTPException(404,'数据源不存在')
        if not import_slot.acquire(blocking=False):raise HTTPException(429,'已有数据集正在导入，请稍后重试')
        did=uuid.uuid4().hex
        path=manager.root/'datasets'/(did+'.jsonl')
        temp=path.with_suffix('.tmp')
        saved=False
        try:
            with manager.db() as db:
                old=db.execute('SELECT d.* FROM datasets d JOIN dataset_sources s ON s.dataset_id=d.id WHERE s.project=? AND s.catalog_key=? AND s.selection=?',(p.project,key,config.documents)).fetchone()
            if old:
                if not (manager.root/'datasets'/(old['id']+'.jsonl')).is_file():raise HTTPException(409,'已导入的数据文件缺失，请恢复备份')
                return {**dict(old),'reused':True}
            if shutil.disk_usage(state).free<manager.min_free_bytes+MAX_OUTPUT:raise HTTPException(507,'磁盘空闲不足，无法导入')
            payload,manifest=collect(source,config.documents,ROOT)
            if shutil.disk_usage(state).free<manager.min_free_bytes+len(payload):raise HTTPException(507,'磁盘空闲不足，无法保存')
            sha=hashlib.sha256(payload).hexdigest();created=now()
            name=f"{source['name']} · {manifest['documents']} 篇.jsonl"
            manifest.update(dataset_sha256=sha,imported_at=created)
            temp.write_bytes(payload);temp.replace(path)
            with manager.db() as db:
                db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',(did,name,manifest['documents'],len(payload),sha,created))
                identities.grant(p,'dataset',did,db)
                db.execute('INSERT INTO dataset_sources VALUES(?,?,?,?,?)',(did,p.project,key,config.documents,json.dumps(manifest,ensure_ascii=False)))
            saved=True
            return {'id':did,'name':name,'rows':manifest['documents'],'bytes':len(payload),'sha256':sha,'reused':False}
        except CatalogError as exc:raise HTTPException(502,str(exc))
        finally:
            temp.unlink(missing_ok=True)
            if not saved:path.unlink(missing_ok=True)
            import_slot.release()

    @app.get('/api/datasets/{did}/source')
    def dataset_source(did: str, p=Depends(auth)):
        identities.require(p,'dataset',did)
        with manager.db() as db:
            row=db.execute('SELECT manifest FROM dataset_sources WHERE dataset_id=?',(did,)).fetchone()
        return json.loads(row[0]) if row else None

    @app.post('/api/datasets')
    async def upload(request: Request, name: str='corpus.jsonl', p=Depends(writer)):
        if len(name)>100 or '/' in name or '\\' in name:
            raise HTTPException(422,'文件名无效')
        if not upload_slots.acquire(blocking=False):
            raise HTTPException(429,'上传并发已达上限，请稍后重试')
        did = uuid.uuid4().hex
        path = manager.root/'datasets'/(did+'.jsonl')
        temp = path.with_suffix('.tmp')
        size = 0
        sha = hashlib.sha256()
        try:
            with temp.open('wb') as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413,'最多上传 100 MiB 的 UTF-8 JSONL')
                    if shutil.disk_usage(state).free < manager.min_free_bytes+len(chunk):
                        raise HTTPException(507,'磁盘可用空间不足')
                    sha.update(chunk)
                    await anyio.to_thread.run_sync(out.write,chunk)
            def validate_and_save():
                rows = 0
                kind = None
                with temp.open('rb') as src:
                    while True:
                        line = src.readline(1024*1024+1)
                        if not line:
                            break
                        if len(line)>1024*1024:
                            raise ValueError('单行不能超过 1 MiB')
                        if not line.strip():
                            continue
                        item = json.loads(line.decode('utf-8'))
                        current=record_format(item)
                        if kind and kind!=current:raise ValueError('同一文件不能混合不同训练格式')
                        kind=current
                        rows += 1
                        if rows>100000:raise ValueError('最多 100000 条样本')
                if rows < 20:
                    raise ValueError('至少需要 20 篇文档；每行一篇，请勿将同篇文档切成跨集合的行')
                temp.replace(path)
                with manager.db() as db:
                    db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',(did,name,rows,size,sha.hexdigest(),now()))
                    identities.grant(p,'dataset',did,db)
                    db.execute('INSERT INTO dataset_formats VALUES(?,?)',(did,kind))
                return {'format':kind,'id':did,'name':name,'rows':rows,'bytes':size,'sha256':sha.hexdigest()}
            return await anyio.to_thread.run_sync(validate_and_save)
        except (ValueError,UnicodeError) as e:
            raise HTTPException(422,str(e))
        finally:
            temp.unlink(missing_ok=True)
            upload_slots.release()
            with manager.db() as db:
                saved=db.execute("SELECT 1 FROM datasets WHERE id=?",(did,)).fetchone()
            if not saved:
                path.unlink(missing_ok=True)

    @app.post('/api/estimate', dependencies=[Depends(auth)])
    def estimate(config: TrainConfig):
        return config.estimate()

    @app.post('/api/jobs')
    def create(config: TrainConfig, idempotency_key: str = Header(default=None), p=Depends(writer)):
        identities.require(p,'dataset',config.dataset_id)
        if config.stage!='pretrain':
            with manager.db() as db:
                data_acl=db.execute("SELECT project FROM resource_acl WHERE kind='dataset' AND id=?",(config.dataset_id,)).fetchone()
            if not data_acl or data_acl['project']!=p.project:
                raise HTTPException(422,'后训练语料与新任务必须属于同一项目')
            if config.base_model_id:
                from server.model_registry import get_model
                try:source_config=get_model(manager,config.base_model_id,p.project)['config']
                except ValueError as exc:raise HTTPException(404,str(exc))
            else:
                source=job(config.base_job_id,p)
                with manager.db() as db:
                    base_acl=db.execute("SELECT project FROM resource_acl WHERE kind='job' AND id=?",(source['id'],)).fetchone()
                if not base_acl or base_acl['project']!=p.project:
                    raise HTTPException(422,'基础模型与后训练任务必须属于同一项目')
                source_config=source['config']
            config=TrainConfig(**{**config.model_dump(),'parameter_count':0,**inherited_config(source_config)})
        else:
            config=config.model_copy(update={'parameter_count':0})
        if idempotency_key and not re.fullmatch(r'[a-f0-9-]{32,36}', idempotency_key):
            raise HTTPException(422, '提交标识无效')
        available = {g['index'] for g in gpu_inventory()['gpus']}
        if not set(config.gpu_ids)<=available:
            raise HTTPException(422,'所选 GPU 不可用，请先检查机器与驱动')
        try:
            key = hashlib.sha256((p.project+'\0'+p.subject+'\0'+idempotency_key).encode()).hexdigest() if idempotency_key else None
            result = manager.create(config.model_dump(), request_key=key, principal=p)
        except ValueError as e:
            raise HTTPException(422,str(e))
        return result

    @app.get('/api/jobs')
    def jobs(p=Depends(auth)):
        return manager.list_jobs(None if p.role=='admin' else p.project)

    @app.get('/api/base-models')
    def base_models(p=Depends(auth)):
        models=[]
        with manager.db() as db:
            for r in db.execute('SELECT * FROM base_models WHERE project=? ORDER BY name',(p.project,)):
                models.append({'model_id':r['id'],'job_id':None,'name':r['name'],'checkpoint':'final','stage':'imported','revision':r['revision'],'config':json.loads(r['config'])})
        for found in manager.list_jobs(p.project):
            for name in ['final',*[x.name for x in reversed(manager.checkpoints(found['id']))]]:
                try:manager.test_model_path(found,name)
                except ValueError:continue
                models.append({'job_id':found['id'],'name':found['config']['name'],'checkpoint':name,
                    'stage':found['config'].get('stage','pretrain'),
                    'config':inherited_config(found['config'])})
        return models

    @app.get('/api/jobs/{jid}', dependencies=[Depends(auth)])
    def detail(jid: str, p=Depends(auth)):
        found = job(jid,p)
        folder = manager.jobdir(jid)
        found['checkpoints'] = [p.name for p in manager.checkpoints(jid)]
        found['testable_models']=[]
        found['test_model_info']={}
        for name in ['final',*reversed(found['checkpoints'])]:
            try:
                model_path=manager.test_model_path(found,name)
                weight_bytes=sum(f.stat().st_size for f in model_path.rglob('*.safetensors'))
                found['test_model_info'][name]={'weight_bytes':weight_bytes,'cpu_supported':weight_bytes<=400*1024**2}
                found['testable_models'].append(name)
            except ValueError:pass
        found['metrics'] = []
        mp = folder/'metrics.jsonl'
        if mp.exists():
            # Bound response and tolerate a concurrently written final line.
            with mp.open('rb') as f:
                f.seek(max(0,mp.stat().st_size-256*1024))
                for line in f.read().splitlines():
                    try:
                        found['metrics'].append(json.loads(line))
                    except (ValueError,UnicodeError):
                        continue
            found['metrics'] = found['metrics'][-2000:]
        summary = folder/'output'/'summary.json'
        found['summary'] = json.loads(summary.read_text()) if summary.exists() else None
        found['artifacts'] = [{'path':str(p.relative_to(folder/'output')),'bytes':p.stat().st_size} for p in (folder/'output').rglob('*') if p.is_file() and not p.is_symlink() and 'checkpoint-' not in str(p.relative_to(folder/'output'))] if (folder/'output').exists() else []
        return found

    @app.post('/api/jobs/{jid}/tests')
    def create_model_test(jid: str, config: ModelTestConfig, idempotency_key: str = Header(default=None), p=Depends(writer)):
        job(jid,p)
        if config.dataset_id:
            identities.require(p,'dataset',config.dataset_id)
            with manager.db() as db:
                job_acl=db.execute("SELECT project FROM resource_acl WHERE kind='job' AND id=?",(jid,)).fetchone()
                data_acl=db.execute("SELECT project FROM resource_acl WHERE kind='dataset' AND id=?",(config.dataset_id,)).fetchone()
            if not job_acl or not data_acl or job_acl['project']!=data_acl['project']:
                raise HTTPException(422,'评估语料必须属于训练任务的同一项目')
        if config.device=='cuda' and config.gpu_id not in {g['index'] for g in gpu_inventory()['gpus']}:
            raise HTTPException(422,'所选测试 GPU 不可用')
        if idempotency_key and not re.fullmatch(r'[a-f0-9-]{32,36}',idempotency_key):raise HTTPException(422,'提交标识无效')
        key=hashlib.sha256((p.subject+'\0'+p.project+'\0'+jid+'\0'+idempotency_key).encode()).hexdigest() if idempotency_key else None
        try:return manager.create_test(jid,config.model_dump(),key)
        except ValueError as e:raise HTTPException(422,str(e))

    @app.get('/api/jobs/{jid}/tests')
    def model_tests(jid: str, p=Depends(auth)):
        job(jid,p)
        return manager.list_tests(jid)

    @app.get('/api/jobs/{jid}/tests/{tid}')
    def model_test_detail(jid: str, tid: str, p=Depends(auth)):
        job(jid,p)
        found=manager.get_test(tid)
        if not found or found['job_id']!=jid:raise HTTPException(404,'测试不存在')
        path=manager.testdir(tid)/'test.log'
        found['log']=''
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-32*1024))
                found['log']=f.read().decode('utf-8',errors='replace')
        return found

    @app.post('/api/jobs/{jid}/tests/{tid}/cancel')
    def cancel_model_test(jid: str, tid: str, p=Depends(writer)):
        job(jid,p)
        found=manager.get_test(tid)
        if not found or found['job_id']!=jid:raise HTTPException(404,'测试不存在')
        return manager.cancel_test(tid)

    @app.get('/api/jobs/{jid}/logs', dependencies=[Depends(auth)])
    def logs(jid: str, p=Depends(auth)):
        job(jid,p)
        path = manager.jobdir(jid)/'train.log'
        if not path.exists():
            return {'text':''}
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-128*1024))
            return {'text':f.read().decode('utf-8',errors='replace')}

    @app.post('/api/jobs/{jid}/cancel', dependencies=[Depends(auth)])
    def cancel(jid: str, p=Depends(writer)):
        job(jid,p)
        return manager.cancel(jid)

    @app.post('/api/jobs/{jid}/resume', dependencies=[Depends(auth)])
    def resume(jid: str, p=Depends(writer)):
        with manager.lock:
            found = job(jid,p)
            if found['status'] not in ['failed','cancelled','interrupted']:
                raise HTTPException(409,'只能恢复已失败、已取消或已中断的任务')
            checkpoints = manager.checkpoints(jid)
            if not checkpoints:
                raise HTTPException(409,'没有包含优化器状态的完整检查点，需重新提交训练')
            # Configuration, tokenizer and dataset remain unchanged for resume.
            existing = [j for j in manager.list_jobs() if j['resume_from']==str(checkpoints[-1]) and j['status'] in ['queued','running','cancelling']]
            if existing:
                return existing[0]
            with manager.db() as db:
                acl=db.execute("SELECT project FROM resource_acl WHERE kind='job' AND id=?",(jid,)).fetchone()
            owner=Principal(p.subject,p.role,acl['project'] if acl else p.project,p.key_id)
            try:
                return manager.create(found['config'],str(checkpoints[-1]),principal=owner)
            except ValueError as e:
                raise HTTPException(422,str(e))

    @app.get('/api/jobs/{jid}/artifact/{file_path:path}', dependencies=[Depends(auth)])
    def artifact(jid: str, file_path: str, p=Depends(auth)):
        job(jid,p)
        root = (manager.jobdir(jid)/'output').resolve()
        candidate = root/file_path
        path = candidate.resolve()
        if root not in path.parents or not path.is_file() or any(x.is_symlink() for x in (candidate,*candidate.parents)):
            raise HTTPException(404,'文件不存在')
        return FileResponse(path,filename=path.name)

    app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='frontend')
    return app
