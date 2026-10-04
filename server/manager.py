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
import psutil
from datetime import datetime, timezone
from contextlib import contextmanager
from server.model_tests import ModelTestQueue

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
        self.max_pending = int(os.environ.get('TRAINLAB_MAX_PENDING_JOBS','20'))
        self.min_free_bytes = int(os.environ.get('TRAINLAB_MIN_FREE_BYTES',str(5*1024**3)))
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, name TEXT, rows INTEGER, bytes INTEGER, sha256 TEXT, created_at TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, config TEXT, created_at TEXT, started_at TEXT, ended_at TEXT, error TEXT, resume_from TEXT, request_key TEXT)')
            if 'request_key' not in {r[1] for r in db.execute('PRAGMA table_info(jobs)')}:
                db.execute('ALTER TABLE jobs ADD COLUMN request_key TEXT')
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS jobs_request_key ON jobs(request_key)')
        self.init_tests()

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
            folder = self.jobdir(jid)
            folder.mkdir()
            # Model and data paths are chosen by the service, never by the client.
            with self.db() as db:
                ds = db.execute('SELECT * FROM datasets WHERE id=?', (config['dataset_id'],)).fetchone()
                if not ds:
                    raise ValueError('数据集不存在')
                payload = {**config, 'dataset_path': str(self.root/'datasets'/(ds['id']+'.jsonl')),
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
        while not self.stop_event.wait(.5):
            with self.lock:
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
                    cmd,env=self.test_command(job,cards) if is_test else self.command(job,cards)
                    with log.open('ab') as output:
                        self.proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    self.active_id=jid;self.active_kind=table
                    with self.db() as db:db.execute(f"UPDATE {table} SET status='running',started_at=? WHERE id=?",(now(),jid))
                except Exception as e:
                    if self.proc and self.proc.poll() is None:self.terminate(self.proc)
                    with self.db() as db:db.execute(f"UPDATE {table} SET status='failed',ended_at=?,error=? WHERE id=?",(now(),str(e),jid))
                    self.proc=None;self.active_id=None;self.active_kind=None
                    continue
            started=time.monotonic();reason=None
            while self.proc.poll() is None:
                if self.stop_event.wait(.5):
                    with self.lock:
                        if self.proc.poll() is None:self.terminate(self.proc)
                    break
                if time.monotonic()-started>job['config'].get('max_runtime_seconds',21600):reason='超过任务运行时限'
                elif shutil.disk_usage(self.root).free<self.min_free_bytes:reason='可用磁盘低于安全水位'
                if reason:
                    with self.lock:self.terminate(self.proc)
                    break
            exit_code=self.proc.wait()
            with self.lock:
                status='interrupted' if self.stop_event.is_set() else 'cancelled' if jid in self.cancelled else 'succeeded' if exit_code==0 else 'failed'
                error=reason or (None if status=='succeeded' else f'进程退出码 {exit_code}；详情见日志')
                if reason:status='failed'
                result=None
                if is_test and status=='succeeded':
                    try:
                        file=folder/'result.json'
                        if file.stat().st_size>256*1024:raise ValueError('结果超过大小上限')
                        result=json.dumps(json.loads(file.read_text()),ensure_ascii=False,allow_nan=False)
                    except (OSError,ValueError) as e:status='failed';error='读取测试结果失败：'+str(e)
                if is_test and status=='failed' and log.exists():
                    with log.open('rb') as f:
                        f.seek(max(0,log.stat().st_size-4000));error=(error or '测试失败')+'\n'+f.read().decode('utf-8',errors='replace')
                with self.db() as db:
                    db.execute(f'UPDATE {table} SET status=?,ended_at=?,error=? WHERE id=?',(status,now(),error,jid))
                    if is_test:db.execute('UPDATE model_tests SET result=? WHERE id=?',(result,jid))
                self.cancelled.discard(jid);self.proc=None;self.active_id=None;self.active_kind=None
