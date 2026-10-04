import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import psutil
import pytest
from server.manager import Manager,ROOT
from server.schema import TrainConfig
from server.train import require_finite_metrics
from scripts.backup import backup
from test_security import system,upload,create
from test_model_tests import completed


def until(check,seconds=10):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if check():return
        time.sleep(.05)
    assert check(),'Timed out waiting for process state'


def test_maintenance_is_admin_only_durable_and_stops_new_work(system):
    c,k=system;m=c.app.state.manager;did=upload(c,k['alice']);jid=completed(c,k)
    queued=create(c,k['alice'],did,'e'*32).json()
    url='/api/scheduler/control';payload={'paused':True,'reason':'deployment'}
    assert c.post(url,json=payload).status_code==401
    for who in ('viewer','alice'):assert c.post(url,json=payload,headers=k[who]).status_code==403
    assert c.post(url,json=payload,headers=k['admin']).json()['paused'] is True
    assert Manager(m.root).control()['reason']=='deployment'
    assert create(c,k['alice'],did,'f'*32).status_code==422
    assert c.post(f'/api/jobs/{jid}/tests',json={'prompt':'hello'},headers=k['alice']).status_code==422
    assert m.get(queued['id'])['status']=='queued'
    assert c.get('/api/operations',headers=k['viewer']).status_code==403
    report=c.get('/api/operations',headers=k['admin']);assert report.status_code==200
    assert report.json()['database_integrity']=='ok'
    assert k['admin']['Authorization'][7:] not in report.text and 'prompt' not in report.text
    assert 'trainlab_scheduler_paused 1' in c.get('/api/metrics',headers=k['admin']).text
    assert c.post(url,json={'paused':False},headers=k['admin']).json()['paused'] is False
    assert create(c,k['alice'],did,'f'*32).status_code==200


def queue_manager(tmp_path):
    m=Manager(tmp_path)
    with m.db() as db:db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',('a'*32,'fixture',30,100,'hash','2020'))
    return m,m.create(TrainConfig(dataset_id='a'*32).model_dump())


def test_resource_gates_preserve_queue_and_pause_drains_active_work(tmp_path,monkeypatch):
    m,j=queue_manager(tmp_path)
    cards=[{'index':0,'uuid':'GPU-0','memory_used_mib':2000}]
    monkeypatch.setattr('server.manager.gpu_inventory',lambda:{'gpus':cards})
    monkeypatch.setattr(m,'command',lambda *a:([sys.executable,'-c','import time;time.sleep(20)'],os.environ.copy()))
    m.start()
    try:
        until(lambda:m.blocked_reason is not None)
        assert m.get(j['id'])['status']=='queued' and m.proc is None
        cards[0]['memory_used_mib']=0;m.min_free_bytes=10**30
        until(lambda:m.blocked_reason and '磁盘' in m.blocked_reason)
        assert m.get(j['id'])['status']=='queued'
        m.min_free_bytes=0
        until(lambda:m.get(j['id'])['status']=='running')
        m.set_control(True,'drain','admin');assert m.proc.poll() is None
        m.cancel(j['id']);until(lambda:m.get(j['id'])['status']=='cancelled')
        assert m.control()['paused'] and m.thread.is_alive()
    finally:m.close()


def test_execution_lease_blocks_duplicate_workers_and_offline_backup(tmp_path,monkeypatch):
    m,j=queue_manager(tmp_path)
    lease=(tmp_path/'execution.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with pytest.raises(RuntimeError,match='workers'):backup(tmp_path,tmp_path.parent/(tmp_path.name+'-snapshot'))
    monkeypatch.setattr('server.manager.gpu_inventory',lambda:{'gpus':[]})
    monkeypatch.setattr(m,'command',lambda *a:([sys.executable,'-c','raise SystemExit(3)'],os.environ.copy()))
    m.start()
    try:
        until(lambda:m.blocked_reason and '执行锁' in m.blocked_reason)
        assert m.get(j['id'])['status']=='queued'
        lease.close();until(lambda:m.get(j['id'])['status']=='failed')
        assert m.lease_available()
    finally:lease.close();m.close()


def test_log_cap_and_missing_artifacts_fail_instead_of_success(tmp_path,monkeypatch):
    m,j=queue_manager(tmp_path);m.max_log_bytes=100
    monkeypatch.setattr('server.manager.gpu_inventory',lambda:{'gpus':[]})
    monkeypatch.setattr(m,'command',lambda *a:([sys.executable,'-c','import time;print("x"*1000,flush=True);time.sleep(20)'],os.environ.copy()))
    m.start()
    try:
        until(lambda:m.get(j['id'])['status']=='failed')
        assert '日志超过' in m.get(j['id'])['error']
        monkeypatch.setattr(m,'command',lambda *a:([sys.executable,'-c','pass'],os.environ.copy()))
        second=m.create(TrainConfig(dataset_id='a'*32).model_dump())
        until(lambda:m.get(second['id'])['status']=='failed')
        assert '产物校验失败' in m.get(second['id'])['error']
    finally:m.close()


def test_supervisor_cleans_worker_when_api_process_is_killed(tmp_path):
    marker=tmp_path/'worker.pid';guardian=tmp_path/'supervisor.pid'
    worker='import os,time,pathlib;pathlib.Path('+repr(str(marker))+').write_text(str(os.getpid()));time.sleep(60)'
    parent_code='''import fcntl,os,pathlib,subprocess,sys,time
lease=open(sys.argv[1],'a');fcntl.flock(lease,fcntl.LOCK_EX)
p=subprocess.Popen([sys.executable,'-m','server.supervisor','--parent-pid',str(os.getpid()),'--lease-fd',str(lease.fileno()),'--',sys.executable,'-c',sys.argv[3]],pass_fds=(lease.fileno(),),start_new_session=True)
pathlib.Path(sys.argv[2]).write_text(str(p.pid))
time.sleep(60)
'''
    parent=subprocess.Popen([sys.executable,'-c',parent_code,str(tmp_path/'execution.lock'),str(guardian),worker],cwd=ROOT)
    try:
        until(marker.exists);pid=int(marker.read_text());assert psutil.pid_exists(pid)
        parent.kill();parent.wait(timeout=5)
        until(lambda:not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE)
        until(lambda:Manager(tmp_path).lease_available())
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=5)
        for file in (marker,guardian):
            if file.exists():
                try:os.kill(int(file.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass


def test_nonfinite_training_metrics_fail_loudly():
    require_finite_metrics({'loss':1.2,'grad_norm':.5,'step':2})
    for key in ('loss','eval_loss','grad_norm'):
        for bad in (float('nan'),float('inf'),float('-inf')):
            with pytest.raises(FloatingPointError,match=key):require_finite_metrics({key:bad})


def test_model_test_overview_respects_project_scope(system):
    c,k=system;jid=completed(c,k)
    assert c.post(f'/api/jobs/{jid}/tests',json={'prompt':'private prompt'},headers=k['alice']).status_code==200
    assert c.get('/api/overview',headers=k['alice']).json()['model_tests']=={'queued':1}
    assert c.get('/api/overview',headers=k['bob']).json()['model_tests']=={}
    assert 'trainlab_model_tests{status="queued"} 1' in c.get('/api/metrics',headers=k['admin']).text
