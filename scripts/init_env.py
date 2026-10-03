#!/usr/bin/env python3
"""Initialize deployment settings; credentials are issued individually by scripts.users."""
import os
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'.env'
try:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
except FileExistsError:
    print('.env 已存在；生产模式须移除旧 TRAINLAB_API_TOKEN，使用 scripts.users 签发个人凭据。')
else:
    with os.fdopen(fd,'w') as f:
        f.write('TRAINLAB_BIND=127.0.0.1\nTRAINLAB_PORT=8000\n')
    print('部署配置已生成。下一步使用 scripts.users 签发个人凭据。')
