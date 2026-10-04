"""From-scratch causal text training. Launch with torchrun; CPU smoke is CLI-only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


def require_finite_metrics(logs):
    for key,value in logs.items():
        if isinstance(value,(int,float)) and not math.isfinite(value):
            raise FloatingPointError(f'训练指标 {key} 出现非有限值，停止运行；请检查精度、学习率及数据后恢复完整检查点')


def documents(path, seed):
    seen = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line)['text'].strip()
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            bucket = int(hashlib.sha256((str(seed)+digest).encode()).hexdigest()[:8],16)%100
            yield text, 'validation' if bucket<10 else 'train'


def prepare(config, output):
    import numpy as np
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    path = config['dataset_path']
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    if h.hexdigest()!=config['dataset_sha256']:
        raise ValueError('数据文件摘要与提交时不一致，拒绝继续训练')
    prepared = output/'prepared'
    prepared.mkdir(parents=True,exist_ok=True)
    counts={'train':0,'validation':0}
    for _, split in documents(path,config['seed']):
        counts[split]+=1
    if min(counts.values())<2:
        raise ValueError('按文档去重和哈希划分后，训练/验证各至少需要两篇不同文档；请增加独立文档')
    if config.get('resume_from'):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config['resume_from'],local_files_only=True)
    else:
        raw = Tokenizer(models.BPE(unk_token='<|unk|>'))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=config['vocab_size'],min_frequency=2,
                    special_tokens=['<|pad|>','<|eos|>','<|unk|>'],
                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        raw.train_from_iterator((t for t,s in documents(path,config['seed']) if s=='train'),trainer=trainer)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<|pad|>',eos_token='<|eos|>',unk_token='<|unk|>',bos_token='<|eos|>',model_max_length=config['seq_length'])
    tokenizer.save_pretrained(output/'tokenizer')
    tokens={'train':0,'validation':0}
    with open(prepared/'train.bin','wb') as ft, open(prepared/'validation.bin','wb') as fv:
        for text, split in documents(path,config['seed']):
            ids=tokenizer.encode(text,add_special_tokens=False,verbose=False)+[tokenizer.eos_token_id]
            np.asarray(ids,dtype='<u4').tofile(ft if split=='train' else fv)
            tokens[split]+=len(ids)
    n=config['seq_length']
    blocks={k:v//n for k,v in tokens.items()}
    if blocks['train']<len(config['gpu_ids'])*config['micro_batch'] or blocks['validation']<1:
        raise ValueError(f'语料不足以构造训练/验证序列：{blocks}；增加文档或降低序列长度、GPU 数及 micro batch')
    meta={'documents_after_exact_dedup':counts,'tokens':tokens,'blocks':blocks,'vocab_size':len(tokenizer),
          'dataset_sha256':h.hexdigest(),'split':'seeded SHA256 by document, approx 90/10',
          'packing':'EOS between documents; contiguous non-overlapping blocks; trailing partial block dropped',
          'dropped_tokens':{k:v%n for k,v in tokens.items()}}
    (output/'data_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta


def run(config, cpu_smoke=False):
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, TrainerCallback, default_data_collator, set_seed
    rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    if not cpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用；真实训练入口不会退回 CPU')
        torch.cuda.set_device(local_rank)
        if config['precision']=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('当前设备不支持 BF16；请选择 FP32 或检查驱动')
        if world>1:
            torch.distributed.init_process_group(backend='nccl')
    elif world!=1:
        raise ValueError('CPU smoke must use one process')
    output=Path(config['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        if config.get('stage','pretrain')=='pretrain':
            meta=prepare(config,output)
        else:
            from server.posttraining import prepare_post
            meta=prepare_post(config,output)
        print(json.dumps({'event':'data_prepared',**meta},ensure_ascii=False),flush=True)
    if world>1:
        torch.distributed.barrier()
    tokenizer=PreTrainedTokenizerFast.from_pretrained(output/'tokenizer',local_files_only=True)
    set_seed(config['seed'])
    from server.architectures import model_config,model_class
    architecture=config.get('architecture','gpt2')
    # Explicit architecture class + new config creates RANDOM weights; no download.
    stage=config.get('stage','pretrain')
    reference=None
    if stage=='pretrain':
        model=model_class(architecture)(model_config(config,len(tokenizer),tokenizer.pad_token_id,tokenizer.eos_token_id))
    else:
        model=model_class(architecture).from_pretrained(config['base_model_path'],local_files_only=True,use_safetensors=True)
        model.config.use_cache=False
        if stage=='dpo':
            reference=model_class(architecture).from_pretrained(config['base_model_path'],local_files_only=True,use_safetensors=True)
            for candidate in (model,reference):
                for module in candidate.modules():
                    if isinstance(module,torch.nn.Dropout):module.p=0.
            reference.requires_grad_(False).eval()

    class PackedDataset(Dataset):
        def __init__(self,path,length):
            self.data=np.memmap(path,dtype='<u4',mode='r')
            self.length=length
        def __len__(self):
            return len(self.data)//self.length
        def __getitem__(self,i):
            ids=torch.tensor(self.data[i*self.length:(i+1)*self.length].astype(np.int64))
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':ids.clone()}

    class Metrics(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if state.is_world_process_zero and logs:
                require_finite_metrics(logs)
                safe=logs
                event={'step':state.global_step,'time':time.time(),**safe}
                with open(config['metrics_path'],'a') as f:
                    f.write(json.dumps(event,allow_nan=False)+'\n')
        def on_save(self,args,state,control,**kwargs):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if state.is_world_process_zero:
                # Manager offers only checkpoints whose Trainer save completed.
                (Path(args.output_dir)/f'checkpoint-{state.global_step}'/'complete.json').write_text('{"complete":true}')

    args=TrainingArguments(output_dir=str(output),max_steps=config['max_steps'],
        per_device_train_batch_size=config['micro_batch'],per_device_eval_batch_size=config['micro_batch'],
        gradient_accumulation_steps=config['grad_accum'],learning_rate=config['learning_rate'],
        warmup_steps=math.ceil(config['warmup_ratio']*config['max_steps']),weight_decay=config['weight_decay'],
        lr_scheduler_type='cosine',optim='adamw_torch',max_grad_norm=1.0,
        eval_strategy='steps',eval_steps=config['eval_steps'],logging_steps=1,logging_nan_inf_filter=False,
        save_strategy='steps',save_steps=config['save_steps'],save_total_limit=2,
        bf16=config['precision']=='bf16' and not cpu_smoke,
        use_cpu=cpu_smoke,tf32=False,seed=config['seed'],data_seed=config['seed'],
        gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False,report_to=[],dataloader_num_workers=0,
        disable_tqdm=True,remove_unused_columns=False)
    trainer_type=Trainer;extra={};collate=default_data_collator
    if stage=='pretrain':
        train_data=PackedDataset(output/'prepared'/'train.bin',config['seq_length'])
        val_data=PackedDataset(output/'prepared'/'validation.bin',config['seq_length'])
    else:
        from server.posttraining import PostDataset,collator,dpo_trainer_class
        train_data=PostDataset(output/'prepared'/'train.jsonl')
        val_data=PostDataset(output/'prepared'/'validation.jsonl')
        collate=collator(tokenizer.pad_token_id,stage=='dpo')
        if stage=='dpo':
            trainer_type=dpo_trainer_class();extra={'reference':reference,'beta':config['dpo_beta']}
    trainer=trainer_type(model=model,args=args,train_dataset=train_data,eval_dataset=val_data,
        processing_class=tokenizer,data_collator=collate,callbacks=[Metrics()],**extra)
    if rank==0:
        print(json.dumps({'event':'model_initialized','parameters':model.num_parameters(),
            'architecture':architecture,'world_size':world,'effective_batch':config['micro_batch']*config['grad_accum']*world,
            'torch_version':torch.__version__,'cuda':torch.version.cuda,'cpu_smoke':cpu_smoke}),flush=True)
    result=trainer.train(resume_from_checkpoint=config.get('resume_from'))
    evaluation=trainer.evaluate()
    trainer.save_model(str(output/'final'))
    if rank==0:
        tokenizer.save_pretrained(output/'final')
        loss=evaluation.get('eval_loss')
        summary={'mode':'CPU smoke test' if cpu_smoke else 'GPU '+stage, 'stage':stage,
                 'loss_kind':'DPO preference loss' if stage=='dpo' else 'response cross entropy' if stage=='sft' else 'next token cross entropy',
                 'parameters':model.num_parameters(),'architecture':architecture,'world_size':world,
                 'training':result.metrics,'validation':evaluation,
                 'perplexity':math.exp(loss) if stage!='dpo' and loss is not None and math.isfinite(loss) and loss<50 else None,
                 'note':'DPO loss 不是语言困惑度；需独立偏好/业务评测。' if stage=='dpo' else '同一分词器、损失掩码和验证集内比较；本任务没有独立测试集。',
                 'config':{k:v for k,v in config.items() if k not in ['metrics_path','dataset_path','output_dir']}}
        temp=output/'summary.tmp'
        temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        temp.replace(output/'summary.json')
    if world>1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--cpu-smoke',action='store_true')
    args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.cpu_smoke)
