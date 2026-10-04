"""Administrator-imported, project-scoped local ModelScope snapshots."""
import json
import re
import shutil
import uuid
from pathlib import Path
from server.posttraining import model_fingerprint


def init_registry(manager):
    (manager.root/'models').mkdir(exist_ok=True)
    with manager.db() as db:
        db.execute('CREATE TABLE IF NOT EXISTS base_models (id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL, source TEXT NOT NULL, revision TEXT NOT NULL, fingerprint TEXT NOT NULL, config TEXT NOT NULL)')


def get_model(manager, mid, project=None):
    with manager.db() as db:
        row=db.execute('SELECT * FROM base_models WHERE id=?',(mid,)).fetchone()
    if not row or (project is not None and row['project']!=project):
        raise ValueError('导入模型不存在或不属于当前项目')
    value=dict(row);value['config']=json.loads(value['config'])
    value['path']=str(manager.root/'models'/mid)
    return value


def import_model(manager, path, project, source, revision, context=1024):
    import fcntl
    with (manager.root/'import.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _import_model(manager,path,project,source,revision,context)


def _import_model(manager, path, project, source, revision, context):
    from safetensors import safe_open
    from transformers import PreTrainedTokenizerFast
    from server.schema import TrainConfig
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',project):raise ValueError('项目名无效')
    if not re.fullmatch(r'Qwen/Qwen3-(0\.6B|1\.7B)(-Base)?',source):
        raise ValueError('当前适配 ModelScope Qwen3 0.6B / 1.7B 稠密文本模型；不接受 MoE、量化或远程代码')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',revision):raise ValueError('请记录实际下载 revision')
    path=Path(path)
    if not path.is_dir() or path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('模型目录不存在或包含符号链接；请使用 ModelScope --local_dir 完整下载')
    raw=json.loads((path/'config.json').read_text())
    if raw.get('model_type')!='qwen3' or raw.get('quantization_config') or raw.get('auto_map'):
        raise ValueError('仅支持原始 Qwen3 稠密 safetensors 模型，无量化、无自定义代码')
    weights=list(path.glob('*.safetensors'))
    if not weights:raise ValueError('缺少 safetensors 权重')
    params=0;names=set();locations={}
    for file in weights:
        with safe_open(file,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key in names:raise ValueError('权重分片存在重复张量')
                names.add(key);locations[key]=file.name
                if f.get_slice(key).get_dtype() not in ('F32','BF16','F16'):raise ValueError('训练权重必须为 FP32/BF16/FP16，不能是量化或整数张量')
                shape=f.get_slice(key).get_shape();count=1
                for dim in shape:count*=dim
                params+=count
    if not 0<params<=2_000_000_000:raise ValueError('当前全参数后训练只开放不超过 2B 的导入模型')
    index=path/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text())['weight_map']
        if mapping!=locations:
            raise ValueError('权重索引与文件不完整或含非法路径')
    elif len(weights)!=1:raise ValueError('多分片模型必须有完整权重索引')
    if not 64<=context<=min(2048,raw['max_position_embeddings']):raise ValueError('训练上下文需为 64–2048 且不超过基础模型上限')
    c={'architecture':'qwen3','layers':raw['num_hidden_layers'],'hidden_size':raw['hidden_size'],
       'heads':raw['num_attention_heads'],'kv_heads':raw['num_key_value_heads'],'intermediate_size':raw['intermediate_size'],
       'seq_length':context,'vocab_size':raw['vocab_size'],'parameter_count':params}
    TrainConfig(dataset_id='a'*32,**c)
    if not isinstance(raw.get('head_dim',raw['hidden_size']//raw['num_attention_heads']),int) or not 16<=raw.get('head_dim',raw['hidden_size']//raw['num_attention_heads'])<=256:raise ValueError('不支持的 head_dim')
    # Validate tensor names and shapes against the explicit model class without materializing weights.
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    with torch.device('meta'):
        model=Qwen3ForCausalLM(Qwen3Config.from_dict(raw))
    expected={k:tuple(v.shape) for k,v in model.state_dict().items()}
    if raw.get('tie_word_embeddings') and 'lm_head.weight' not in names:expected.pop('lm_head.weight',None)
    if set(expected)!=names:raise ValueError('权重张量列表与模型结构不匹配，拒绝不完整模型')
    for file in weights:
        with safe_open(file,framework='pt',device='cpu') as f:
            if any(tuple(f.get_slice(k).get_shape())!=expected[k] for k in f.keys()):raise ValueError('模型权重形状不匹配')
    tokenizer=PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None or len(tokenizer)>raw['vocab_size']:
        raise ValueError('分词器缺失特殊 token 或词表超过模型 embedding')
    # Copy only runtime artifacts. Never bring downloaded Python or pickle code into state.
    artifacts=[p for p in path.iterdir() if p.is_file() and (p.suffix in ('.json','.safetensors','.jinja') or p.name in ('merges.txt','vocab.json'))]
    total=sum(p.stat().st_size for p in artifacts)
    if shutil.disk_usage(manager.root).free<manager.min_free_bytes+total:raise ValueError('磁盘空间不足以保存模型副本')
    mid=uuid.uuid4().hex;temp=manager.root/'models'/('.import-'+mid);target=manager.root/'models'/mid
    try:
        temp.mkdir()
        for artifact in artifacts:shutil.copy2(artifact,temp/artifact.name)
        fingerprint=model_fingerprint(temp)
        temp.rename(target)
        with manager.db() as db:
            db.execute('INSERT INTO base_models VALUES(?,?,?,?,?,?,?)',(mid,project,source,source,revision,fingerprint,json.dumps(c)))
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);shutil.rmtree(target,ignore_errors=True);raise
    return {'id':mid,'project':project,'source':source,'revision':revision,'sha256':fingerprint,'parameters':params}
