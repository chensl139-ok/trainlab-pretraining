"""Individual high-entropy API keys; project ACL; immediate database revocation."""
import hashlib
import hmac
import secrets
import re
import time
from dataclasses import dataclass
from fastapi import HTTPException

@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    project: str
    key_id: str = 'legacy'

    def write(self):
        if self.role not in ('admin','operator'):
            raise HTTPException(403,'只读账号不能执行写操作')

class AuthStore:
    def __init__(self, manager):
        self.manager=manager
        with manager.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, digest TEXT NOT NULL, subject TEXT NOT NULL, role TEXT NOT NULL, project TEXT NOT NULL, expires REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS resource_acl (kind TEXT NOT NULL, id TEXT NOT NULL, project TEXT NOT NULL, owner TEXT NOT NULL, PRIMARY KEY(kind,id))')
            db.execute('CREATE TABLE IF NOT EXISTS audit_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, request_id TEXT NOT NULL, actor TEXT NOT NULL, project TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL, status INTEGER NOT NULL)')

    def issue(self,subject,role,project,days=30):
        if role not in ('admin','operator','viewer') or not re.fullmatch(r'[\w.@-]{1,80}',subject) or not re.fullmatch(r'[\w.@-]{1,80}',project) or not 1<=days<=365:
            raise ValueError('invalid subject, role, project or expiry')
        key_id=secrets.token_hex(12)
        raw='tl_'+key_id+'.'+secrets.token_urlsafe(48)
        with self.manager.db() as db:
            db.execute('INSERT INTO api_keys(id,digest,subject,role,project,expires) VALUES(?,?,?,?,?,?)',
                (key_id,hashlib.sha256(raw.encode()).hexdigest(),subject,role,project,time.time()+days*86400))
        return key_id,raw

    def authenticate(self, raw):
        if len(raw)>256 or not raw.startswith('tl_') or '.' not in raw:
            raise HTTPException(401,'访问凭据无效或已失效')
        key_id=raw[3:].split('.',1)[0]
        with self.manager.db() as db:
            row=db.execute('SELECT * FROM api_keys WHERE id=?',(key_id,)).fetchone()
        if not row or row['revoked'] or row['expires']<=time.time() or not hmac.compare_digest(row['digest'],hashlib.sha256(raw.encode()).hexdigest()):
            raise HTTPException(401,'访问凭据无效或已失效')
        return Principal(row['subject'],row['role'],row['project'],row['id'])

    def allowed(self,p,kind,rid):
        if p.role=='admin':
            return True
        with self.manager.db() as db:
            row=db.execute('SELECT project FROM resource_acl WHERE kind=? AND id=?',(kind,rid)).fetchone()
        return row is not None and row['project']==p.project

    def require(self,p,kind,rid):
        if not self.allowed(p,kind,rid):
            raise HTTPException(404,'资源不存在或无权访问')

    def grant(self,p,kind,rid,db=None):
        if db is not None:
            db.execute('INSERT INTO resource_acl VALUES(?,?,?,?)',(kind,rid,p.project,p.subject))
        else:
            with self.manager.db() as conn:
                self.grant(p,kind,rid,conn)
