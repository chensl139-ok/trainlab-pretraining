"""Real offline SFT -> DPO training, deterministic resume and masked evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from server.posttraining import model_fingerprint


def run(base, config, work, env):
    from safetensors.torch import load_file
    from server.evaluate import run as evaluate
    from server.schema import ModelTestConfig
    # Exercise the import validator with a small local fixture; this is not an official downloaded model.
    from server.manager import Manager
    from server.model_registry import import_model,get_model
    manager=Manager(work/'registry-fixture')
    imported=import_model(manager,base,'smoke','Qwen/Qwen3-0.6B-Base','local-test-fixture',context=config['seq_length'])
    registered=get_model(manager,imported['id'],'smoke')
    base=Path(registered['path'])
    config={**config,**registered['config']}
    for stage in ('sft','dpo'):
        corpus=work/(stage+'.jsonl')
        records=[{'prompt':f'Number {i}?',**({'response':str(i)} if stage=='sft' else {'chosen':str(i),'rejected':'wrong'})} for i in range(240)]
        corpus.write_text(''.join(json.dumps(r)+'\n' for r in records))
        original=model_fingerprint(base)
        c={**config,'stage':stage,'base_job_id':'b'*32,'base_checkpoint':'final','dpo_beta':.1,
           'dataset_path':str(corpus),'dataset_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
           'base_model_path':str(base),'base_model_sha256':original,'learning_rate':.0001,'grad_accum':2}
        for part in ('first','resumed'):
            output=work/(stage+'-'+part)
            c.update(output_dir=str(output),metrics_path=str(work/(stage+'-'+part+'-metrics.jsonl')),
                     resume_from=str(work/(stage+'-first')/'checkpoint-5') if part=='resumed' else None)
            file=work/(stage+'-'+part+'.json');file.write_text(json.dumps(c))
            subprocess.run([sys.executable,'-m','server.train','--config',str(file),'--cpu-smoke'],env=env,check=True)
            summary=json.loads((output/'summary.json').read_text())
            assert summary['stage']==stage and summary['validation']['eval_loss']>0
            if stage=='dpo':assert summary['perplexity'] is None
        first=load_file(work/(stage+'-first')/'final'/'model.safetensors')
        resumed=load_file(work/(stage+'-resumed')/'final'/'model.safetensors')
        assert all(first[k].equal(resumed[k]) for k in first),'Post-training resume changed final weights'
        before=load_file(base/'model.safetensors')
        assert any(not first[k].equal(before[k]) for k in first),'Weights did not update'
        assert model_fingerprint(base)==original,'Post-training mutated the base model'
        for mode in ('generate','score'):
            test=ModelTestConfig(mode=mode,prompt='Number 3?',max_new_tokens=4,max_blocks=2).model_dump()
            test.update(train_config=c,model_path=str(work/(stage+'-first')/'final'),
                validation_post_path=str(work/(stage+'-first')/'prepared'/'validation.jsonl'),result_path=str(work/(stage+'-'+mode+'.json')))
            result=evaluate(test)
            if mode=='score':assert result['loss']>0 and result['score_unit']=='examples'
            else:assert result['generated_tokens']>0
        base=work/(stage+'-first')/'final'
        print('POST TRAINING PASSED:',stage,'weights changed, base unchanged, exact resume, generation and response-only scoring',flush=True)
