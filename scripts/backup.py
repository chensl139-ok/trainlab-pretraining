"""Offline snapshot/restore. Stop the API first; scheduler lock prevents live snapshots."""
import argparse
import fcntl
import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager, closing
from pathlib import Path

@contextmanager
def exclusive(state):
    with (state/'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Stop TrainLab before backup or restore') from None
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(snapshot):
    manifest=json.loads((snapshot/'manifest.json').read_text())
    actual={str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file() and p.relative_to(snapshot).as_posix()!='manifest.json'}
    if actual!=set(manifest['files']):raise ValueError('Snapshot file inventory mismatch')
    for name,sha in manifest['files'].items():
        p=snapshot/name
        if snapshot.resolve() not in p.resolve().parents or any(x.is_symlink() for x in (p,*p.parents)) or digest(p)!=sha:
            raise ValueError('Snapshot integrity check failed')
    with closing(sqlite3.connect((snapshot/'trainlab.sqlite3').resolve().as_uri()+'?immutable=1',uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity check failed')
    return manifest

def backup(state,destination):
    state=state.resolve();destination=destination.resolve()
    if state==destination or state in destination.parents:raise ValueError('Store backups outside the state directory')
    with exclusive(state):
        destination.mkdir(mode=0o700,parents=True,exist_ok=False)
        with closing(sqlite3.connect(state/'trainlab.sqlite3')) as src,closing(sqlite3.connect(destination/'trainlab.sqlite3')) as dst:
            src.backup(dst)
            dst.execute('PRAGMA journal_mode=DELETE')
        for name in ('datasets','jobs'):
            for path in (state/name).rglob('*'):
                if path.is_symlink():raise ValueError('Symlinks are not allowed in state backups')
            shutil.copytree(state/name,destination/name)
        files={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}
        (destination/'manifest.json').write_text(json.dumps({'format':1,'state_path':str(state),'files':files},indent=2))
        verify(destination)
    return len(files)

def restore(snapshot,state):
    manifest=verify(snapshot)
    if str(state.resolve())!=manifest['state_path']:
        raise ValueError('Restore to the same absolute state mount path; configs contain checkpoint paths')
    state.mkdir(mode=0o700,parents=True,exist_ok=True)
    with exclusive(state):
        if any(p.name!='scheduler.lock' for p in state.iterdir()):raise ValueError('Restore requires an empty state directory')
        for name in ('datasets','jobs'):shutil.copytree(snapshot/name,state/name)
        shutil.copy2(snapshot/'trainlab.sqlite3',state/'trainlab.sqlite3')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['backup','verify','restore'])
    p.add_argument('--state',type=Path,default=Path('/state'));p.add_argument('--snapshot',type=Path,required=True);a=p.parse_args()
    if a.action=='backup':print(json.dumps({'verified_files':backup(a.state,a.snapshot)}))
    elif a.action=='verify':print(json.dumps({'verified_files':len(verify(a.snapshot)['files'])}))
    else:restore(a.snapshot,a.state);print('Restore verified and complete; start service to reconcile interrupted jobs.')
