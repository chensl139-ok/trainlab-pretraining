"""Explicit, offline model implementations; never load user-supplied remote code."""
ARCHITECTURES = [
    {'id':'qwen3_5','name':'Qwen3.5 · 混合注意力','description':'纯文本稠密模型；每 4 层包含 3 层 Gated DeltaNet 和 1 层全注意力。缩小配置从零训练。','minimum_layers':4},
    {'id':'qwen3','name':'Qwen3 · 稠密 Transformer','description':'GQA、QK-Norm、RoPE、SwiGLU；适合作为现代架构对照。','minimum_layers':2},
    {'id':'gpt2','name':'GPT-2 · 兼容已有实验','description':'学习式位置编码、标准多头注意力；已有任务默认保持此架构。','minimum_layers':2},
]

def parameter_estimate(c):
    if c.get('parameter_count'):return c['parameter_count']
    h,l,v=c['hidden_size'],c['layers'],c['vocab_size']
    architecture=c.get('architecture','gpt2')
    if architecture=='gpt2':
        return v*h+c['seq_length']*h+l*(12*h*h+13*h)+2*h
    d=h//c['heads'];kv=c.get('kv_heads',2)*d;i=c.get('intermediate_size',0) or 3*h
    common=3*h*i+2*h
    if architecture=='qwen3':
        return 2*v*h+l*(2*h*h+2*h*kv+2*d+common)+h
    full=3*h*h+2*h*kv+2*d+common
    linear=2*h*kv+3*h*h+2*h*c['heads']+4*(2*kv+h)+2*c['heads']+d+common
    return 2*v*h+(l//4)*full+(l-l//4)*linear+h

def model_config(c,vocab_size,pad_token_id,eos_token_id):
    from transformers import GPT2Config,Qwen3Config,Qwen3_5TextConfig
    architecture=c.get('architecture','gpt2')
    common=dict(vocab_size=vocab_size,pad_token_id=pad_token_id,bos_token_id=eos_token_id,eos_token_id=eos_token_id,use_cache=False)
    if architecture=='gpt2':
        return GPT2Config(**common,n_positions=c['seq_length'],n_embd=c['hidden_size'],n_layer=c['layers'],n_head=c['heads'],resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
    h=c['hidden_size'];d=h//c['heads']
    common.update(hidden_size=h,intermediate_size=c.get('intermediate_size',0) or 3*h,num_hidden_layers=c['layers'],num_attention_heads=c['heads'],num_key_value_heads=c.get('kv_heads',2),head_dim=d,max_position_embeddings=c['seq_length'],tie_word_embeddings=False,attention_dropout=0.)
    if architecture=='qwen3':
        return Qwen3Config(**common,rope_parameters={'rope_type':'default','rope_theta':10000.},layer_types=['full_attention']*c['layers'])
    if architecture=='qwen3_5':
        half=d//2;part=half//3
        return Qwen3_5TextConfig(**common,linear_key_head_dim=d,linear_value_head_dim=d,linear_num_key_heads=c.get('kv_heads',2),linear_num_value_heads=c['heads'],linear_conv_kernel_dim=4,
            layer_types=['full_attention' if (n+1)%4==0 else 'linear_attention' for n in range(c['layers'])],
            rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[part,part,half-2*part]})
    raise ValueError('不支持的模型架构')

def model_class(architecture):
    from transformers import GPT2LMHeadModel,Qwen3ForCausalLM,Qwen3_5ForCausalLM
    classes={'gpt2':GPT2LMHeadModel,'qwen3':Qwen3ForCausalLM,'qwen3_5':Qwen3_5ForCausalLM}
    if architecture not in classes:raise ValueError('不支持的模型架构')
    return classes[architecture]
