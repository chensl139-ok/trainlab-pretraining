"""One scheduler per state directory; SQLite history; one process group at a time."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import shutil
import threading
import time
import uuid
import traceback
import psutil
from datetime import datetime, timezone
from contextlib import contextmanager
from server.model_tests import ModelTestQueue,validate_result
from server.architectures import parameter_estimate
from server.posttraining import MODEL_FIELDS, model_fingerprint, inherited_config

ROOT = Path(__file__).resolve().parent.parent

def now():
    return datetime.now(timezone.utc).isoformat()

def gpu_inventory():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=6, check=True)
        cards = []
        for row in csv.reader(p.stdout.splitlines()):
            i,u,n,total,used,util,driver = [x.strip() for x in row]
            cards.append({'index': int(i), 'uuid': u, 'name': n, 'memory_total_mib': int(total), 'memory_used_mib': int(used), 'utilization_percent': int(util), 'driver': driver})
        return {'available': bool(cards), 'gpus': cards, 'error': None}
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return {'available': False, 'gpus': [], 'error': '无法读取 NVIDIA GPU，请检查驱动及容器 GPU 挂载。'}

class Manager(ModelTestQueue):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for x in ['datasets', 'jobs']:
            (self.root/x).mkdir(exist_ok=True)
        self.db_path = self.root/'trainlab.sqlite3'
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.proc = None
        self.active_id = None
        self.active_kind = None
        self.cancelled = set()
        self.thread = None
        self.lockfile = None
        self.execution_lease = None
        self.blocked_reason = None
        self.scheduler_error = None
        self.started_at = now()
        self.max_log_bytes = int(os.environ.get('TRAINLAB_MAX_LOG_BYTES',str(256*1024**2)))
        self.gpu_idle_mib = int(os.environ.get('TRAINLAB_GPU_IDLE_MIB','1024'))
        self.max_pending = int(os.environ.get('TRAINLAB_MAX_PENDING_JOBS','20'))
        self.min_free_bytes = int(os.environ.get('TRAINLAB_MIN_FREE_BYTES',str(5*1024**3)))
        if self.max_pending<1 or self.min_free_bytes<0 or self.gpu_idle_mib<0 or self.max_log_bytes<1024**2:
            raise ValueError('Invalid runtime limits: pending >= 1, disk/GPU thresholds >= 0, log limit >= 1 MiB')
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, name TEXT, rows INTEGER, bytes INTEGER, sha256 TEXT, created_at TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, config TEXT, created_at TEXT, started_at TEXT, ended_at TEXT, error TEXT, resume_from TEXT, request_key TEXT)')
            if 'request_key' not in {r[1] for r in db.execute('PRAGMA table_info(jobs)')}:
                db.execute('ALTER TABLE jobs ADD COLUMN request_key TEXT')
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS jobs_request_key ON jobs(request_key)')
            db.execute('CREATE TABLE IF NOT EXISTS dataset_formats (id TEXT PRIMARY KEY, format TEXT NOT NULL)')
        from server.model_registry import init_registry
        init_registry(self)
        self.init_tests()
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS scheduler_control (id INTEGER PRIMARY KEY CHECK(id=1), paused INTEGER NOT NULL, reason TEXT NOT NULL, updated_at TEXT NOT NULL, actor TEXT NOT NULL)')
            db.execute('INSERT OR IGNORE INTO scheduler_control VALUES(1,0,?,?,?)',('',now(),'system'))

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def control(self):
        with self.db() as db:
            value=dict(db.execute('SELECT paused,reason,updated_at,actor FROM scheduler_control WHERE id=1').fetchone())
        value['paused']=bool(value['paused'])
        return value

    def set_control(self,paused,reason,actor):
        with self.lock:
            with self.db() as db:
                db.execute('UPDATE scheduler_control SET paused=?,reason=?,updated_at=?,actor=? WHERE id=1',(int(paused),reason,now(),actor))
        return self.control()

    def require_accepting(self):
        if self.control()['paused']:raise ValueError('维护模式：已停止接收新训练和测试任务')

    def lease_available(self):
        with (self.root/'execution.lock').open('a') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
            fcntl.flock(lease,fcntl.LOCK_UN)
        return True

    def launch_guard(self,job,is_test,cards):
        c=job['config']
        with self.db() as db:
            ds=db.execute('SELECT bytes FROM datasets WHERE id=?',(c.get('dataset_id'),)).fetchone()
        reserve=self.max_log_bytes if is_test else parameter_estimate(c)*48+(ds['bytes'] if ds else 0)*8+self.max_log_bytes
        required=self.min_free_bytes+reserve
        if shutil.disk_usage(self.root).free<required:
            return f'磁盘不足：启动预留约 {required/1024**3:.1f} GiB（含安全水位）；队列保留，释放空间后重试'
        c=job['config'];ids=[c['gpu_id']] if is_test and c['device']=='cuda' else [] if is_test else c['gpu_ids']
        busy=[g['index'] for g in cards if g['index'] in ids and g.get('memory_used_mib',0)>self.gpu_idle_mib]
        if busy:return 'GPU '+','.join(map(str,busy))+' 已有显存占用；等待其他进程释放，不抢占运行'
        return None

    def start(self):
        self.lockfile = open(self.root/'scheduler.lock', 'a')
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lockfile.close()
            raise RuntimeError('Only one API process may use this state directory; use --workers 1.')
        with self.db() as db:
            db.execute("UPDATE jobs SET status='interrupted', ended_at=?, error='服务重启；可从已有检查点恢复' WHERE status IN ('running','cancelling')", (now(),))
        with self.db() as db:
            db.execute("UPDATE model_tests SET status='interrupted',ended_at=?,error='服务重启；请重新提交测试' WHERE status IN ('running','cancelling')",(now(),))
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.terminate(self.proc)
        if self.thread:
            self.thread.join(timeout=20)
        if self.lockfile:
            fcntl.flock(self.lockfile, fcntl.LOCK_UN)
            self.lockfile.close()

    @staticmethod
    def terminate(proc):
        # torchrun may create worker sessions; track descendants as well as its group.
        try:
            descendants = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        finally:
            _, alive = psutil.wait_procs(descendants, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=3)

    def jobdir(self, jid):
        return self.root/'jobs'/jid

    def list_jobs(self, project=None):
        with self.db() as db:
            if project is None:
                rows=db.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100')
            else:
                rows=db.execute("SELECT j.* FROM jobs j JOIN resource_acl a ON a.kind='job' AND a.id=j.id WHERE a.project=? ORDER BY j.created_at DESC LIMIT 100",(project,))
            return [self.decode(r) for r in rows]

    def page_jobs(self, project=None, query='', status='', offset=0, limit=12):
        join=" JOIN resource_acl a ON a.kind='job' AND a.id=j.id" if project is not None else ''
        clauses=[];args=[]
        if project is not None:
            clauses.append('a.project=?');args.append(project)
        if query:
            # instr treats percent/underscore literally, unlike LIKE patterns.
            clauses.append("(instr(lower(json_extract(j.config,'$.name')),lower(?))>0 OR instr(j.id,?)>0)")
            args.extend([query,query])
        if status:
            clauses.append('j.status=?');args.append(status)
        where=(' WHERE '+' AND '.join(clauses)) if clauses else ''
        with self.db() as db:
            db.execute('BEGIN')  # Count and page share one read snapshot.
            total=db.execute('SELECT COUNT(*) FROM jobs j'+join+where,args).fetchone()[0]
            rows=db.execute('SELECT j.* FROM jobs j'+join+where+' ORDER BY j.created_at DESC,j.id DESC LIMIT ? OFFSET ?',[*args,limit,offset]).fetchall()
        return {'items':[self.decode(r) for r in rows],'total':total,'offset':offset,'limit':limit}

    def get(self, jid):
        with self.db() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
        return self.decode(row) if row else None

    def decode(self, row):
        d = dict(row)
        d['config'] = json.loads(d['config'])
        return d

    def create(self, config, resume_from=None, request_key=None, principal=None):
        jid = uuid.uuid4().hex
        with self.lock:
            if request_key:
                with self.db() as db:
                    prior = db.execute('SELECT * FROM jobs WHERE request_key=?', (request_key,)).fetchone()
                if prior:
                    decoded = self.decode(prior)
                    if decoded['config'] != config:
                        raise ValueError('相同提交标识不能用于不同配置')
                    return decoded
            self.require_accepting()
            with self.db() as db:
                pending=db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')").fetchone()[0]
                pending+=db.execute("SELECT COUNT(*) FROM model_tests WHERE status IN ('queued','running','cancelling')").fetchone()[0]
                ds=db.execute('SELECT * FROM datasets WHERE id=?',(config['dataset_id'],)).fetchone()
            if not ds:
                raise ValueError('数据集不存在')
            if pending>=self.max_pending:
                raise ValueError('任务队列已达上限，请等待任务完成')
            if shutil.disk_usage(self.root).free<self.min_free_bytes:
                raise ValueError('可用磁盘低于安全水位，拒绝创建任务')
            stage=config.get('stage','pretrain')
            with self.db() as db:
                fmt=db.execute('SELECT format FROM dataset_formats WHERE id=?',(config['dataset_id'],)).fetchone()
            if (fmt['format'] if fmt else 'pretrain') != stage:
                raise ValueError('数据格式与训练阶段不匹配，请选择对应的预训练、SFT 或 DPO 数据')
            base_payload={}
            if stage!='pretrain':
                if config.get('base_model_id'):
                    from server.model_registry import get_model
                    imported=get_model(self,config['base_model_id'],principal.project if principal else None)
                    base_path=Path(imported['path']);base_config=imported['config']
                    if config['base_checkpoint']!='final':raise ValueError('导入模型只能使用 final 版本')
                    expected_fingerprint=imported['fingerprint']
                else:
                    base_job=self.get(config.get('base_job_id'))
                    if not base_job:raise ValueError('基础模型任务不存在')
                    base_path=self.test_model_path(base_job,config['base_checkpoint']);base_config=base_job['config']
                    expected_fingerprint=None
                base_config=inherited_config(base_config)
                for field in MODEL_FIELDS:
                    if config.get(field,0)!=base_config.get(field,0):
                        raise ValueError('后训练的模型结构与上下文必须继承基础模型：'+field)
                base_payload={'base_model_path':str(base_path),'base_model_sha256':model_fingerprint(base_path)}
                if expected_fingerprint and expected_fingerprint!=base_payload['base_model_sha256']:
                    raise ValueError('导入模型文件摘要已改变，拒绝训练')
                if resume_from:
                    previous=json.loads((Path(resume_from).parents[1]/'config.json').read_text())
                    if previous.get('base_model_sha256')!=base_payload['base_model_sha256']:
                        raise ValueError('后训练基础模型已改变，不能恢复旧检查点')
            folder = self.jobdir(jid)
            folder.mkdir()
            # Model and data paths are chosen by the service, never by the client.
            with self.db() as db:
                ds = db.execute('SELECT * FROM datasets WHERE id=?', (config['dataset_id'],)).fetchone()
                if not ds:
                    raise ValueError('数据集不存在')
                payload = {**config, **base_payload, 'dataset_path': str(self.root/'datasets'/(ds['id']+'.jsonl')),
                           'dataset_sha256': ds['sha256'], 'output_dir': str(folder/'output'),
                           'metrics_path': str(folder/'metrics.jsonl'), 'resume_from': resume_from}
                (folder/'config.tmp').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
                (folder/'config.tmp').replace(folder/'config.json')
                db.execute('INSERT INTO jobs(id,status,config,created_at,resume_from,request_key) VALUES (?,?,?,?,?,?)', (jid,'queued',json.dumps(config),now(),resume_from,request_key))
                if principal:
                    db.execute('INSERT INTO resource_acl VALUES(?,?,?,?)',('job',jid,principal.project,principal.subject))
        return self.get(jid)

    def cancel(self, jid):
        with self.lock:
            job = self.get(jid)
            if not job:
                raise KeyError(jid)
            if job['status'] == 'queued':
                with self.db() as db:
                    db.execute("UPDATE jobs SET status='cancelled',ended_at=? WHERE id=?", (now(),jid))
            elif job['status'] == 'running':
                self.cancelled.add(jid)
                with self.db() as db:
                    db.execute("UPDATE jobs SET status='cancelling' WHERE id=?", (jid,))
                if self.proc and self.active_id == jid:
                    self.terminate(self.proc)
            return self.get(jid)

    def checkpoints(self, jid):
        out = self.jobdir(jid)/'output'
        return sorted([p for p in out.glob('checkpoint-*') if p.is_dir() and (p/'trainer_state.json').is_file() and (p/'optimizer.pt').is_file() and (p/'complete.json').is_file()], key=lambda p: int(p.name.split('-')[-1]))

    def command(self, job, cards):
        ids = job['config']['gpu_ids']
        gpu_map = {g['index']:g['uuid'] for g in cards}
        if any(i not in gpu_map for i in ids):
            raise ValueError('请求的 GPU 不可用，请检查 GPU 挂载和编号')
        env=self.child_environment()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_map[i] for i in ids)
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc_per_node={len(ids)}', '-m', 'server.train', '--config', str(self.jobdir(job['id'])/'config.json')]
        return cmd, env

    def child_environment(self):
        # Explicit allowlist: training does not inherit cloud credentials, API keys or proxy secrets.
        safe={'PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CUDA_HOME','CUDA_PATH','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES'}
        env = {k:v for k,v in os.environ.items() if k in safe}
        env['HF_HOME']=str(self.root/'cache')
        env['HF_HUB_OFFLINE']='1'
        env['TRANSFORMERS_OFFLINE']='1'
        env['PYTHONUNBUFFERED'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        env['OMP_NUM_THREADS'] = '4'
        # Do not pass control-plane auth into the training child.
        env.pop('TRAINLAB_API_TOKEN', None)
        return env

    def test_command(self,test,cards):
        parent=self.get(test['job_id'])
        self.test_model_path(parent,test['config']['checkpoint'])
        env=self.child_environment();env['CUDA_VISIBLE_DEVICES']=''
        if test['config']['device']=='cuda':
            gpu=next((g for g in cards if g['index']==test['config']['gpu_id']),None)
            if gpu is None:raise ValueError('测试 GPU 已不可用')
            env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        return [sys.executable,'-m','server.evaluate','--config',str(self.testdir(test['id'])/'config.json')],env

    def loop(self):
        try:self._loop()
        except Exception as exc:
            self.scheduler_error=type(exc).__name__+': '+str(exc)
            traceback.print_exc()
            if self.proc and self.proc.poll() is None:self.terminate(self.proc)
        finally:
            if self.execution_lease:self.execution_lease.close();self.execution_lease=None

    def _loop(self):
        while not self.stop_event.wait(.5):
            with self.lock:
                self.blocked_reason=None
                if self.control()['paused']:continue
                with self.db() as db:
                    train=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
                    test=db.execute("SELECT * FROM model_tests WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
                if not train and not test:continue
                is_test=bool(test and (not train or test['created_at']<train['created_at']))
                table='model_tests' if is_test else 'jobs'
                job=self.decode(test if is_test else train);jid=job['id']
                folder=self.testdir(jid) if is_test else self.jobdir(jid)
                log=folder/('test.log' if is_test else 'train.log')
                try:
                    cards=gpu_inventory()['gpus']
                    self.blocked_reason=self.launch_guard(job,is_test,cards)
                    if self.blocked_reason:
                        self.stop_event.wait(2)
                        continue
                    lease=(self.root/'execution.lock').open('a')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        lease.close();self.blocked_reason='仍有工作进程持有执行锁；等待退出，禁止重复启动'
                        self.stop_event.wait(2)
                        continue
                    self.execution_lease=lease
                    cmd,env=self.test_command(job,cards) if is_test else self.command(job,cards)
                    with log.open('ab') as output:
                        supervised=[sys.executable,'-m','server.supervisor','--parent-pid',str(os.getpid()),'--lease-fd',str(lease.fileno()),'--',*cmd]
                        self.proc=subprocess.Popen(supervised,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                    self.active_id=jid;self.active_kind=table
                    with self.db() as db:db.execute(f"UPDATE {table} SET status='running',started_at=? WHERE id=?",(now(),jid))
                except Exception as e:
                    if self.proc and self.proc.poll() is None:self.terminate(self.proc)
                    with self.db() as db:db.execute(f"UPDATE {table} SET status='failed',ended_at=?,error=? WHERE id=?",(now(),str(e),jid))
                    self.proc=None;self.active_id=None;self.active_kind=None
                    if self.execution_lease:self.execution_lease.close();self.execution_lease=None
                    continue
            started=time.monotonic();reason=None
            while self.proc.poll() is None:
                if self.stop_event.wait(.5):
                    with self.lock:
                        if self.proc.poll() is None:self.terminate(self.proc)
                    break
                if time.monotonic()-started>job['config'].get('max_runtime_seconds',21600):reason='超过任务运行时限'
                elif shutil.disk_usage(self.root).free<self.min_free_bytes:reason='可用磁盘低于安全水位'
                elif log.exists() and log.stat().st_size>self.max_log_bytes:reason='任务日志超过运行上限，请检查重复报错后从完整检查点恢复'
                if reason:
                    with self.lock:self.terminate(self.proc)
                    break
            exit_code=self.proc.wait()
            with self.lock:
                status='interrupted' if self.stop_event.is_set() else 'cancelled' if jid in self.cancelled else 'succeeded' if exit_code==0 else 'failed'
                error=reason or (None if status=='succeeded' else f'进程退出码 {exit_code}；详情见日志')
                if reason:status='failed'
                result=None
                if not is_test and status=='succeeded':
                    try:
                        output=folder/'output';summary=json.loads((output/'summary.json').read_text())
                        if not isinstance(summary,dict) or not isinstance(summary.get('validation'),dict) or not isinstance(summary['validation'].get('eval_loss'),(int,float)):raise ValueError('缺少验证 loss')
                        json.dumps(summary,allow_nan=False)
                        final=output/'final'
                        if not all((final/name).is_file() for name in ('config.json','tokenizer.json')) or not list(final.glob('*.safetensors')):
                            raise ValueError('缺少最终模型权重、配置或分词器')
                    except (OSError,ValueError) as exc:status='failed';error='训练进程已退出，但产物校验失败：'+str(exc)
                if is_test and status=='succeeded':
                    try:
                        file=folder/'result.json'
                        if file.stat().st_size>256*1024:raise ValueError('结果超过大小上限')
                        result=json.dumps(validate_result(json.loads(file.read_text()),job['config']),ensure_ascii=False,allow_nan=False)
                    except (OSError,ValueError) as e:status='failed';error='读取测试结果失败：'+str(e)
                if is_test and status=='failed' and log.exists():
                    with log.open('rb') as f:
                        f.seek(max(0,log.stat().st_size-4000));error=(error or '测试失败')+'\n'+f.read().decode('utf-8',errors='replace')
                with self.db() as db:
                    db.execute(f'UPDATE {table} SET status=?,ended_at=?,error=? WHERE id=?',(status,now(),error,jid))
                    if is_test:db.execute('UPDATE model_tests SET result=? WHERE id=?',(result,jid))
                self.cancelled.discard(jid);self.proc=None;self.active_id=None;self.active_kind=None
                if self.execution_lease:self.execution_lease.close();self.execution_lease=None
