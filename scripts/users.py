"""Run on the server as an administrator. Raw credentials are written once, never logged."""
import argparse
import json
import os
from pathlib import Path
from server.manager import Manager
from server.security import AuthStore

p=argparse.ArgumentParser()
p.add_argument('--state',default=os.environ.get('TRAINLAB_STATE_DIR','state'))
sub=p.add_subparsers(dest='action',required=True)
issue=sub.add_parser('issue')
issue.add_argument('--subject',required=True)
issue.add_argument('--project',required=True)
issue.add_argument('--role',choices=['admin','operator','viewer'],required=True)
issue.add_argument('--days',type=int,default=30)
issue.add_argument('--credential-file',required=True)
revoke=sub.add_parser('revoke');revoke.add_argument('--key-id',required=True)
sub.add_parser('list')
a=p.parse_args();m=Manager(Path(a.state));auth=AuthStore(m)
if a.action=='issue':
    dest=Path(a.credential_file)
    dest.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as out:
            kid,secret=auth.issue(a.subject,a.role,a.project,a.days)
            out.write(secret+'\n')
            out.flush()
            os.fsync(out.fileno())
        print(json.dumps({'key_id':kid,'credential_file':str(dest),'expires_in_days':a.days}))
    except BaseException:
        if 'kid' in locals():
            with m.db() as db:db.execute('UPDATE api_keys SET revoked=1 WHERE id=?',(kid,))
        dest.unlink(missing_ok=True)
        raise
elif a.action=='revoke':
    with m.db() as db:
        count=db.execute('UPDATE api_keys SET revoked=1 WHERE id=?',(a.key_id,)).rowcount
    print(json.dumps({'revoked':count}))
else:
    with m.db() as db:
        print(json.dumps([dict(r) for r in db.execute('SELECT id,subject,role,project,expires,revoked FROM api_keys')],ensure_ascii=False,indent=2))
