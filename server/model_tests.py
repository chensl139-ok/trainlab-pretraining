"""Persistent test queue, shared with the training scheduler and project ACLs."""
import hashlib
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone

ACTIVE=('queued','running','cancelling')
def timestamp():return datetime.now(timezone.utc).isoformat()

def validate_result(result,config):
    if not isinstance(result,dict) or result.get('mode')!=config['mode'] or result.get('checkpoint')!=config['checkpoint']:
        raise ValueError('测试结果与请求不匹配')
    fields=('prompt_tokens','generated_tokens','generation_seconds') if config['mode']=='generate' else ('loss','evaluated_tokens','blocks','sequence_length')
    for name in fields:
        value=result.get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('测试结果字段无效：'+name)
    if config['mode']=='generate':
        if not all(isinstance(result.get(key),str) for key in ('prompt','completion')):raise ValueError('缺少续写文本')
        rate=result.get('tokens_per_second')
        if rate is not None and (isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<0):raise ValueError('生成速率无效')
    else:
        if not all(result.get(key,0)>0 for key in ('evaluated_tokens','blocks','sequence_length')):raise ValueError('评估 token 数无效')
        ppl=result.get('perplexity')
        if ppl is not None and (isinstance(ppl,bool) or not isinstance(ppl,(int,float)) or not math.isfinite(ppl) or ppl<1):raise ValueError('困惑度无效')
    return result

class ModelTestQueue:
    def init_tests(self):
        (self.root/'tests').mkdir(exist_ok=True)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS model_tests (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, status TEXT, config TEXT, created_at TEXT, started_at TEXT, ended_at TEXT, error TEXT, result TEXT, request_key TEXT UNIQUE)')
            db.execute('CREATE INDEX IF NOT EXISTS model_tests_job ON model_tests(job_id,created_at)')

    def testdir(self,tid):return self.root/'tests'/tid

    def get_test(self,tid):
        with self.db() as db:row=db.execute('SELECT * FROM model_tests WHERE id=?',(tid,)).fetchone()
        return self.decode_test(row) if row else None

    def decode_test(self,row):
        out=self.decode(row);out['result']=json.loads(out['result']) if out['result'] else None
        return out

    def list_tests(self,jid):
        with self.db() as db:
            return [self.decode_test(row) for row in db.execute('SELECT * FROM model_tests WHERE job_id=? ORDER BY created_at DESC LIMIT 30',(jid,))]

    def test_model_path(self,job,checkpoint):
        if job['status'] in ACTIVE:raise ValueError('请等待训练停止后再测试，避免与检查点写入或清理冲突')
        if not re.fullmatch(r'final|checkpoint-[0-9]+',checkpoint):raise ValueError('检查点名称无效')
        if checkpoint=='final' and job['status']!='succeeded':raise ValueError('训练尚未成功完成，请选择完整检查点')
        root=self.jobdir(job['id'])/'output';path=root/checkpoint
        if not path.is_dir() or path.is_symlink() or root.is_symlink():raise ValueError('模型文件不存在')
        if checkpoint!='final' and not (path/'complete.json').is_file():raise ValueError('检查点未写入完成')
        if not all((path/name).is_file() for name in ('config.json','tokenizer.json')):raise ValueError('缺少模型配置或分词器')
        files=list(path.rglob('*'))
        if any(f.is_symlink() for f in files):raise ValueError('模型目录不能包含符号链接')
        if not any(f.name.endswith('.safetensors') for f in files):raise ValueError('缺少 safetensors 权重')
        return path

    def create_test(self,jid,config,request_key=None):
        with self.lock:
            if request_key:
                with self.db() as db:prior=db.execute('SELECT * FROM model_tests WHERE request_key=?',(request_key,)).fetchone()
                if prior:
                    old=self.decode_test(prior)
                    if old['config']!=config or old['job_id']!=jid:raise ValueError('相同提交标识不能用于不同测试')
                    return old
            self.require_accepting()
            job=self.get(jid)
            if not job:raise ValueError('训练任务不存在')
            path=self.test_model_path(job,config['checkpoint'])
            with self.db() as db:
                pending=db.execute("SELECT COUNT(*) FROM model_tests WHERE status IN ('queued','running','cancelling')").fetchone()[0]
                pending+=db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')").fetchone()[0]
                ds=db.execute('SELECT * FROM datasets WHERE id=?',(config.get('dataset_id'),)).fetchone()
                fmt=db.execute('SELECT format FROM dataset_formats WHERE id=?',(config.get('dataset_id'),)).fetchone()
            if pending>=self.max_pending:raise ValueError('任务队列已达上限，请等待完成')
            if shutil.disk_usage(self.root).free<self.min_free_bytes:raise ValueError('可用磁盘低于安全水位')
            if config['device']=='cpu':
                size=sum(f.stat().st_size for f in path.glob('*.safetensors'))
                if size>400*1024**2:raise ValueError('CPU 测试仅支持权重不超过 400 MiB 的小模型；请选择 GPU')
            train_config=json.loads((self.jobdir(jid)/'config.json').read_text())
            if config.get('dataset_id'):
                if not ds:raise ValueError('评估语料不存在')
                if (fmt['format'] if fmt else 'pretrain')!=train_config.get('stage','pretrain'):raise ValueError('评估语料格式必须与任务阶段一致')
                if ds['sha256']==train_config.get('dataset_sha256'):raise ValueError('另选评估语料不能与训练语料相同；可选择已有验证集')
            tid=uuid.uuid4().hex;folder=self.testdir(tid);folder.mkdir()
            payload={**config,'train_config':train_config,'model_path':str(path),'validation_path':str(self.jobdir(jid)/'output'/'prepared'/'validation.bin'),
                'validation_post_path':str(self.jobdir(jid)/'output'/'prepared'/'validation.jsonl'),
                'dataset_path':str(self.root/'datasets'/(ds['id']+'.jsonl')) if ds else None,'dataset_sha256':ds['sha256'] if ds else None,
                'result_path':str(folder/'result.json')}
            (folder/'config.json').write_text(json.dumps(payload,ensure_ascii=False))
            with self.db() as db:
                db.execute('INSERT INTO model_tests(id,job_id,status,config,created_at,request_key) VALUES(?,?,?,?,?,?)',(tid,jid,'queued',json.dumps(config),timestamp(),request_key))
            return self.get_test(tid)

    def cancel_test(self,tid):
        with self.lock:
            test=self.get_test(tid)
            if not test:raise ValueError('测试不存在')
            if test['status']=='queued':
                with self.db() as db:db.execute("UPDATE model_tests SET status='cancelled',ended_at=? WHERE id=?",(timestamp(),tid))
            elif test['status']=='running':
                self.cancelled.add(tid)
                with self.db() as db:db.execute("UPDATE model_tests SET status='cancelling' WHERE id=?",(tid,))
                if self.proc and self.active_id==tid:self.terminate(self.proc)
            return self.get_test(tid)
