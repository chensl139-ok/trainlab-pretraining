"""Bounded offline evaluation of platform-produced, safetensors-only models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def score_blocks(config,tokenizer,length):
    import numpy as np
    if not config.get('dataset_path'):
        path=Path(config['validation_path'])
        if not path.is_file():raise ValueError('原验证集文件缺失，请选择另一份评估语料')
        data=np.memmap(path,dtype='<u4',mode='r')
        for n in range(min(len(data)//length,config['max_blocks'])):
            yield data[n*length:(n+1)*length].astype('int64').tolist()
        return
    path=Path(config['dataset_path'])
    if digest(path)!=config['dataset_sha256']:raise ValueError('评估语料摘要不一致')
    pending=[];seen=set();count=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            text=json.loads(line)['text'].strip();h=hashlib.sha256(text.encode()).digest()
            if not text or h in seen:continue
            seen.add(h)
            pending.extend(tokenizer.encode(text,add_special_tokens=False,verbose=False)+[tokenizer.eos_token_id])
            pos=0
            while len(pending)-pos>=length:
                yield pending[pos:pos+length];pos+=length;count+=1
                if count>=config['max_blocks']:return
            pending=pending[pos:]


def run(config):
    import torch
    from transformers import PreTrainedTokenizerFast,set_seed
    from server.architectures import model_class
    started=time.monotonic();torch.set_num_threads(2);set_seed(config['seed'])
    train=config['train_config'];architecture=train.get('architecture','gpt2');length=train['seq_length']
    path=Path(config['model_path'])
    if any(p.is_symlink() for p in [path,*path.rglob('*')]):raise ValueError('模型目录不能包含符号链接')
    tokenizer=PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True)
    device=config['device']
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('指定 GPU 不可用，未回退到 CPU')
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model=model_class(architecture).from_pretrained(path,local_files_only=True,use_safetensors=True,dtype=dtype).to(device).eval()
    if model.config.model_type != {'gpt2':'gpt2','qwen3':'qwen3','qwen3_5':'qwen3_5_text'}[architecture]:raise ValueError('模型架构与训练记录不匹配')
    fingerprint=hashlib.sha256((digest(path/'config.json')+digest(path/'tokenizer.json')).encode()).hexdigest()
    result={'mode':config['mode'],'architecture':architecture,'checkpoint':config['checkpoint'],'device':device,'dtype':str(dtype),
            'parameters':model.num_parameters(),'seed':config['seed'],'tokenizer_sha256':digest(path/'tokenizer.json'),'config_tokenizer_sha256':fingerprint,
            'torch_version':torch.__version__}
    with torch.inference_mode():
        if config['mode']=='generate':
            ids=tokenizer.encode(config['prompt'],add_special_tokens=False)
            if not ids or len(ids)+config['max_new_tokens']>length:
                raise ValueError(f'提示词 tokens ({len(ids)}) + 新增 tokens ({config["max_new_tokens"]}) 超过训练上下文 ({length})，请缩短输入或生成长度')
            x=torch.tensor([ids],device=device)
            options={'max_new_tokens':config['max_new_tokens'],'do_sample':config['temperature']>0,'use_cache':True,'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id}
            if options['do_sample']:options.update(temperature=config['temperature'],top_p=config['top_p'])
            if device=='cuda':torch.cuda.synchronize()
            begin=time.monotonic();out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),**options)
            if device=='cuda':torch.cuda.synchronize()
            seconds=time.monotonic()-begin;new=out[0,len(ids):].tolist()
            result.update(prompt=config['prompt'],completion=tokenizer.decode(new,skip_special_tokens=True),prompt_tokens=len(ids),generated_tokens=len(new),generation_seconds=seconds,tokens_per_second=len(new)/seconds if seconds else None,
                finish_reason='eos' if new and new[-1]==tokenizer.eos_token_id else 'length',note='基础语言模型续写；未进行指令微调，不代表聊天或推理能力。')
        else:
            total=0.;tokens=0;blocks=0
            for ids in score_blocks(config,tokenizer,length):
                x=torch.tensor([ids],device=device);loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=x,use_cache=False).loss.item()
                if not math.isfinite(loss):raise ValueError('评估出现非有限 loss')
                count=len(ids)-1;total+=loss*count;tokens+=count;blocks+=1
            if not tokens:raise ValueError('评估语料不足一个完整上下文块，请增加文档或选择原验证集')
            loss=total/tokens;source=config.get('dataset_path') or config['validation_path']
            result.update(loss=loss,perplexity=math.exp(loss) if loss<50 else None,evaluated_tokens=tokens,blocks=blocks,sequence_length=length,
                source='selected_dataset' if config.get('dataset_path') else 'training_validation',dataset_sha256=digest(source),max_blocks=config['max_blocks'],
                note='按顺序评估有界样本；丢弃末尾不完整块，不计算跨块第一个 token。仅同分词器、同语料摘要和采样范围可比较；验证集不是独立测试集，另选语料也需自行排除训练重叠。')
    result['elapsed_seconds']=time.monotonic()-started
    target=Path(config['result_path']);temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False));temp.replace(target)
    print(json.dumps({'event':'model_test_complete','mode':result['mode'],'architecture':architecture,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    run(json.loads(Path(args.config).read_text()))
