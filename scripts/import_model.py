"""Import a complete ModelScope local download. Run inside the training image."""
import argparse
import json
import os
from pathlib import Path
from server.manager import Manager
from server.model_registry import import_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',default=os.environ.get('TRAINLAB_STATE_DIR','/state'))
    p.add_argument('--path',type=Path,required=True)
    p.add_argument('--project',required=True)
    p.add_argument('--source',required=True,help='ModelScope model id, e.g. Qwen/Qwen3-0.6B')
    p.add_argument('--revision',required=True,help='The revision used for the download; contents are also SHA256 hashed')
    p.add_argument('--context',type=int,default=1024)
    a=p.parse_args();m=Manager(Path(a.state))
    print(json.dumps(import_model(m,a.path,a.project,a.source,a.revision,a.context),ensure_ascii=False))

if __name__=='__main__':main()
