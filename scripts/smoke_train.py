"""Offline CPU check of tokenizer -> random GPT -> loss -> save -> checkpoint resume."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from server.schema import TrainConfig
root=Path(__file__).resolve().parent.parent
corpus=root/'examples'/'sample-corpus.jsonl'
import tempfile
(root/'.runtime').mkdir(exist_ok=True)
work=Path(tempfile.mkdtemp(prefix='cpu-smoke-',dir=root/'.runtime'))
work.mkdir(parents=True,exist_ok=True)
c=TrainConfig(dataset_id='a'*32,layers=2,hidden_size=128,heads=4,seq_length=64,vocab_size=512,
    micro_batch=2,grad_accum=1,max_steps=10,eval_steps=5,save_steps=5,precision='fp32').model_dump()
c.update(dataset_path=str(corpus),dataset_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
    output_dir=str(work/'first'),metrics_path=str(work/'first-metrics.jsonl'),resume_from=None)
env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
for stage in ['first','resumed']:
    if stage=='resumed':
        c.update(resume_from=str(work/'first'/'checkpoint-5'),output_dir=str(work/'resumed'),metrics_path=str(work/'resumed-metrics.jsonl'))
    config=work/(stage+'.json')
    config.write_text(json.dumps(c))
    subprocess.run([sys.executable,'-m','server.train','--config',str(config),'--cpu-smoke'],cwd=root,env=env,check=True)
    result=json.loads((Path(c['output_dir'])/'summary.json').read_text())
    assert result['training']['train_loss']>0
    assert (Path(c['output_dir'])/'final'/'model.safetensors').exists()
from safetensors.torch import load_file
first=load_file(work/'first'/'final'/'model.safetensors')
resumed=load_file(work/'resumed'/'final'/'model.safetensors')
assert first.keys()==resumed.keys()
assert all(first[k].equal(resumed[k]) for k in first), 'Resume changed final weights'
print('CPU smoke passed: tokenizer, scratch training, validation, save, resume; final weights identical.')
print('Evidence directory:',work)
