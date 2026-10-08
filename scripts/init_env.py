#!/usr/bin/env python3
"""Initialize single-owner local deployment settings."""
import os
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'.env'
try:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
except FileExistsError:
    print('.env 已存在，保留原配置。免凭据访问使用 TRAINLAB_AUTH_MODE=local；旧 TRAINLAB_API_TOKEN 须移除。')
else:
    with os.fdopen(fd,'w') as f:
        f.write('TRAINLAB_BIND=127.0.0.1\nTRAINLAB_PORT=8000\nTRAINLAB_AUTH_MODE=local\n')
    print('部署配置已生成：个人免凭据模式，默认通过本机或 SSH 隧道访问。')
