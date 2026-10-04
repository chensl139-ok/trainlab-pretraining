"""Offline CPU check of tokenizer -> random GPT -> loss -> save -> checkpoint resume."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--architecture',choices=['gpt2','qwen3','qwen3_5'],default='gpt2');args=parser.parse_args()
from server.schema import TrainConfig
root=Path(__file__).resolve().parent.parent
corpus=root/'examples'/'sample-corpus.jsonl'
import tempfile
(root/'.runtime').mkdir(exist_ok=True)
work=Path(tempfile.mkdtemp(prefix='cpu-smoke-',dir=root/'.runtime'))
work.mkdir(parents=True,exist_ok=True)
c=TrainConfig(dataset_id='a'*32,architecture=args.architecture,layers=4 if args.architecture=='qwen3_5' else 2,hidden_size=128,heads=4,seq_length=64,vocab_size=512,
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
    from server.architectures import parameter_estimate
    meta=json.loads((Path(c['output_dir'])/'data_manifest.json').read_text())
    assert result['parameters']==parameter_estimate({**c,'vocab_size':meta['vocab_size']})
    assert result['training']['train_loss']>0
    assert (Path(c['output_dir'])/'final'/'model.safetensors').exists()
from safetensors.torch import load_file
first=load_file(work/'first'/'final'/'model.safetensors')
resumed=load_file(work/'resumed'/'final'/'model.safetensors')
assert first.keys()==resumed.keys()
assert all(first[k].equal(resumed[k]) for k in first), 'Resume changed final weights'
print('CPU smoke passed: tokenizer, scratch training, validation, save, resume; final weights identical.')
print('Evidence directory:',work)

# Load both a complete checkpoint and the final model using the offline test path.
from server.evaluate import run as evaluate
from server.schema import ModelTestConfig
for mode,checkpoint in [('generate','checkpoint-5'),('generate','final'),('score','final')]:
    test=ModelTestConfig(mode=mode,checkpoint=checkpoint,prompt='The model',max_new_tokens=8,max_blocks=2).model_dump()
    test.update(train_config=c,model_path=str(work/'first'/checkpoint),validation_path=str(work/'first'/'prepared'/'validation.bin'),result_path=str(work/(mode+'-'+checkpoint+'.json')))
    result=evaluate(test)
    assert result['architecture']==args.architecture
    if mode=='generate':assert 0<result['generated_tokens']<=8
    else:assert result['evaluated_tokens']>0 and result['loss']>0
print('Model reload, generation, and held-out validation scoring passed:',args.architecture)
