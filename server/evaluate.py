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



def post_rows(config, tokenizer, length):
    from server.posttraining import PostDataset, record_format, encode_pair
    stage=config['train_config']['stage']
    if not config.get('dataset_path'):
        data=PostDataset(config['validation_post_path'])
        for i in range(min(len(data),config['max_blocks'])):yield data[i]
        return
    path=Path(config['dataset_path'])
    if digest(path)!=config['dataset_sha256']:raise ValueError('评估语料摘要不一致')
    seen=set();count=0
    with path.open() as f:
        for line in f:
            if not line.strip():continue
            item=json.loads(line)
            if record_format(item)!=stage:raise ValueError('评估语料格式与模型训练阶段不匹配')
            key=json.dumps(item,sort_keys=True)
            if key in seen:continue
            seen.add(key)
            yield encode_pair(tokenizer,item['prompt'],item['response'],length) if stage=='sft' else {k:encode_pair(tokenizer,item['prompt'],item[k],length) for k in ('chosen','rejected')}
            count+=1
            if count>=config['max_blocks']:return


def report_progress(config, phase, **details):
    """Atomic, bounded status for the API; never contains prompts or corpus text."""
    target=Path(config['result_path']).with_name('progress.json')
    data={'phase':phase,**details}
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(data,allow_nan=False));temp.replace(target)
    print(json.dumps({'event':'model_test_progress',**data}),flush=True)


def generation_ids(config, tokenizer):
    from server.posttraining import prompt_text
    train=config['train_config'];length=train['seq_length']
    text=prompt_text(config['prompt'],tokenizer) if train.get('stage','pretrain')!='pretrain' else config['prompt']
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)+config['max_new_tokens']>length:
        raise ValueError(f'提示词（含模板）{len(ids)} tokens + 生成 {config["max_new_tokens"]} tokens 超过上下文 {length}；请将生成长度降至 {max(0,length-len(ids))} 或缩短输入')
    return ids


def run(config):
    import torch
    from transformers import PreTrainedTokenizerFast,set_seed
    from server.architectures import model_class
    started=time.monotonic();report_progress(config,'validating');torch.set_num_threads(2);set_seed(config['seed'])
    train=config['train_config'];architecture=train.get('architecture','gpt2');length=train['seq_length']
    path=Path(config['model_path'])
    if any(p.is_symlink() for p in [path,*path.rglob('*')]):raise ValueError('模型目录不能包含符号链接')
    tokenizer=PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True)
    # Validate token budget and bounded evaluation samples before allocating model memory.
    post=train.get('stage','pretrain')!='pretrain'
    if config['mode']=='generate':
        ids=generation_ids(config,tokenizer)
        report_progress(config,'loading',prompt_tokens=len(ids),context_tokens=length)
    else:
        rows=list(post_rows(config,tokenizer,length) if post else score_blocks(config,tokenizer,length))
        if not rows:raise ValueError('没有可评估的完整样本，请增加语料或选择原验证集')
        report_progress(config,'loading',total=len(rows),completed=0)
    device=config['device']
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('指定 GPU 不可用，未回退到 CPU')
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model=model_class(architecture).from_pretrained(path,local_files_only=True,use_safetensors=True,dtype=dtype).to(device).eval()
    if model.config.model_type != {'gpt2':'gpt2','qwen3':'qwen3','qwen3_5':'qwen3_5_text'}[architecture]:raise ValueError('模型架构与训练记录不匹配')
    fingerprint=hashlib.sha256(''.join(str(f.relative_to(path))+digest(f) for f in sorted(path.rglob('*')) if f.is_file() and f.suffix in ('.json','.jinja')).encode()).hexdigest()
    result={'mode':config['mode'],'architecture':architecture,'checkpoint':config['checkpoint'],'device':device,'dtype':str(dtype),
            'parameters':model.num_parameters(),'seed':config['seed'],'tokenizer_sha256':digest(path/'tokenizer.json'),'config_tokenizer_sha256':fingerprint,
            'torch_version':torch.__version__}
    with torch.inference_mode():
        if config['mode']=='generate':
            report_progress(config,'generating',prompt_tokens=len(ids),context_tokens=length)
            x=torch.tensor([ids],device=device)
            options={'max_new_tokens':config['max_new_tokens'],'do_sample':config['temperature']>0,'use_cache':True,'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id}
            if options['do_sample']:options.update(temperature=config['temperature'],top_p=config['top_p'])
            if device=='cuda':torch.cuda.synchronize()
            begin=time.monotonic();out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),**options)
            if device=='cuda':torch.cuda.synchronize()
            seconds=time.monotonic()-begin;new=out[0,len(ids):].tolist()
            result.update(prompt=config['prompt'],completion=tokenizer.decode(new,skip_special_tokens=True),prompt_tokens=len(ids),generated_tokens=len(new),generation_seconds=seconds,tokens_per_second=len(new)/seconds if seconds else None,
                finish_reason='eos' if new and new[-1]==tokenizer.eos_token_id else 'length',note='使用训练时相同的指令模板进行单轮回答；质量需独立业务评测。' if post else '基础语言模型续写；未进行指令微调，不代表聊天或推理能力。')
        elif train.get('stage','pretrain')!='pretrain':
            from server.posttraining import collator,response_logps
            paired=train['stage']=='dpo';total=0.;tokens=0;blocks=0;correct=0
            report_progress(config,'scoring',completed=0,total=len(rows))
            for row in rows:
                batch={k:v.to(device) for k,v in collator(tokenizer.pad_token_id,paired)([row]).items()}
                labels=batch.pop('labels')
                logps=response_logps(model(**batch,use_cache=False).logits,labels)
                # SFT targets or DPO chosen targets only; never exponentiate DPO training loss.
                count=int((labels[0,1:]!=-100).sum());total-=float(logps[0]);tokens+=count;blocks+=1
                if not torch.isfinite(logps).all():raise ValueError('评估出现非有限 log 概率')
                if paired:correct+=int(logps[0]>logps[1])
                report_progress(config,'scoring',completed=blocks,total=len(rows))
            if not tokens:raise ValueError('没有可评估的回答 tokens')
            loss=total/tokens;source=config.get('dataset_path') or config['validation_post_path']
            result.update(loss=loss,perplexity=math.exp(loss) if loss<50 else None,evaluated_tokens=tokens,blocks=blocks,sequence_length=length,
                source='selected_dataset' if config.get('dataset_path') else 'training_validation',dataset_sha256=digest(source),max_blocks=config['max_blocks'],
                score_unit='examples',loss_scope='chosen_response' if paired else 'response',
                preference_accuracy=correct/blocks if paired else None,
                note='仅回答及 EOS 的 token 加权交叉熵；不是 DPO 训练损失。偏好命中率按回答总 log 概率比较，受长度影响，不代表人工胜率。' if paired else '仅回答及 EOS 参与评分，排除提示词与 padding。仅同语料、模板、分词器和样本范围可比较。')
        else:
            total=0.;tokens=0;blocks=0
            report_progress(config,'scoring',completed=0,total=len(rows))
            for ids in rows:
                x=torch.tensor([ids],device=device);loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=x,use_cache=False).loss.item()
                if not math.isfinite(loss):raise ValueError('评估出现非有限 loss')
                count=len(ids)-1;total+=loss*count;tokens+=count;blocks+=1
                report_progress(config,'scoring',completed=blocks,total=len(rows))
            if not tokens:raise ValueError('评估语料不足一个完整上下文块，请增加文档或选择原验证集')
            loss=total/tokens;source=config.get('dataset_path') or config['validation_path']
            result.update(loss=loss,perplexity=math.exp(loss) if loss<50 else None,evaluated_tokens=tokens,blocks=blocks,sequence_length=length,
                source='selected_dataset' if config.get('dataset_path') else 'training_validation',dataset_sha256=digest(source),max_blocks=config['max_blocks'],
                note='按顺序评估有界样本；丢弃末尾不完整块，不计算跨块第一个 token。仅同分词器、同语料摘要和采样范围可比较；验证集不是独立测试集，另选语料也需自行排除训练重叠。')
    report_progress(config,'saving')
    result['elapsed_seconds']=time.monotonic()-started
    target=Path(config['result_path']);temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False));temp.replace(target)
    report_progress(config,'complete')
    print(json.dumps({'event':'model_test_complete','mode':result['mode'],'architecture':architecture,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    try:run(json.loads(Path(args.config).read_text()))
    except Exception as exc:
        if 'out of memory' in str(exc).lower():raise RuntimeError('测试设备内存不足：请选择显存更大的 GPU 或较小的模型版本；续写可减少生成 tokens。平台不会自动切换设备。') from exc
        raise
