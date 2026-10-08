"""Individual high-entropy API keys; project ACL; immediate database revocation."""
import hashlib
import hmac
import secrets
import re
import time
import ipaddress
from urllib.parse import urlsplit
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


class LocalAccess:
    """Single-owner access through explicit local hosts; no browser credentials."""
    def __init__(self, hosts='localhost,127.0.0.1,::1', networks='127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16'):
        self.hosts={h.strip().lower() for h in hosts.split(',') if h.strip()}
        if not self.hosts or any(h=='*' or '/' in h or '@' in h or not re.fullmatch(r'[a-z0-9.:-]+',h) for h in self.hosts):
            raise ValueError('TRAINLAB_LOCAL_HOSTS 必须是 localhost 或明确的本机/私有 IP，不能使用通配符或 URL')
        for host in self.hosts:
            if host=='localhost':continue
            try:address=ipaddress.ip_address(host)
            except ValueError:raise ValueError('个人模式仅支持 localhost 或本机/私有 IP；域名发布请使用 credentials 模式')
            if not address.is_private:raise ValueError('个人模式不允许公网 IP；请使用 SSH 隧道或 credentials 模式')
        self.networks=[ipaddress.ip_network(n.strip()) for n in networks.split(',') if n.strip()]
        if not self.networks or any(n.prefixlen==0 or not n.is_private for n in self.networks):
            raise ValueError('TRAINLAB_LOCAL_NETWORKS 仅支持明确的本机或私有网段')

    def require(self,request):
        try:peer=ipaddress.ip_address(request.client.host)
        except (ValueError,AttributeError):raise HTTPException(403,'无法确认访问来源，请通过本机或 SSH 隧道访问')
        if not any(peer in network for network in self.networks) or request.url.hostname.lower() not in self.hosts or len(request.headers.getlist('host'))!=1:
            raise HTTPException(403,'免凭据个人模式仅允许本机或配置的内网入口；请通过 SSH 隧道打开 localhost，或配置允许的内网主机')
        if request.headers.get('sec-fetch-site','') not in ('','none','same-origin'):
            raise HTTPException(403,'免凭据模式不允许其他网站发起访问')
        origins=request.headers.getlist('origin')
        if origins:
            def origin_parts(value):
                parsed=urlsplit(value)
                if parsed.scheme not in ('http','https') or parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment:raise ValueError('invalid origin')
                return (parsed.scheme,parsed.hostname,parsed.port or (443 if parsed.scheme=='https' else 80))
            try:
                if len(origins)!=1 or origin_parts(origins[0])!=(request.url.scheme,request.url.hostname,request.url.port or (443 if request.url.scheme=='https' else 80)):
                    raise HTTPException(403,'免凭据模式仅允许同源请求')
            except ValueError:raise HTTPException(403,'请求来源无效')
        return Principal('local-owner','admin','research','local')
