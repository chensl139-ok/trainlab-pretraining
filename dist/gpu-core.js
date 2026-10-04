/* Shared, pure UI calculations; independently testable without a GPU or a DOM. */
(function(root){
  'use strict';
  const defaults={name:'首次文本预训练',dataset_id:'',gpu_ids:[],architecture:'qwen3',kv_heads:2,intermediate_size:0,layers:6,hidden_size:384,heads:6,seq_length:256,vocab_size:4096,micro_batch:2,grad_accum:8,max_steps:100,learning_rate:.0003,warmup_ratio:.03,weight_decay:.1,eval_steps:25,save_steps:25,seed:42,precision:'bf16',gradient_checkpointing:true,max_runtime_seconds:21600};
  const bounds={kv_heads:[1,32],intermediate_size:[0,16384],layers:[2,32],hidden_size:[128,2048],heads:[2,32],seq_length:[64,2048],vocab_size:[512,65536],micro_batch:[1,16],grad_accum:[1,256],max_steps:[10,1000000],learning_rate:[.000001,.01],warmup_ratio:[0,.5],weight_decay:[0,1],eval_steps:[1,10000],save_steps:[1,10000],seed:[0,2147483647],max_runtime_seconds:[60,604800]};
  const fractions=new Set(['learning_rate','warmup_ratio','weight_decay']);
  function validate(c){
    const errors=[];
    if(typeof c.name!=='string'||!c.name.trim()||c.name.length>80)errors.push('任务名称需为 1–80 个字符');
    if(!/^[a-f0-9]{32}$/.test(c.dataset_id||''))errors.push('请选择已上传的数据集');
    if(!Array.isArray(c.gpu_ids)||!c.gpu_ids.length||c.gpu_ids.length>8||new Set(c.gpu_ids).size!==c.gpu_ids.length||c.gpu_ids.some(x=>!Number.isInteger(x)||x<0||x>63))errors.push('请选择 1–8 张不同的 GPU');
    for(const [key,[lo,hi]] of Object.entries(bounds))if(!Number.isFinite(c[key])||c[key]<lo||c[key]>hi||(!fractions.has(key)&&!Number.isInteger(c[key])))errors.push(`参数 ${key} 应为 ${lo}–${hi} 范围内的${fractions.has(key)?'数值':'整数'}`);
    if(c.hidden_size%c.heads!==0)errors.push('隐藏维度必须能被注意力头数整除');
    if(!['gpt2','qwen3','qwen3_5'].includes(c.architecture))errors.push('请选择受支持的架构');
    if(c.architecture!=='gpt2'){if(c.heads%c.kv_heads||c.hidden_size/c.heads%2||c.hidden_size/c.heads<16)errors.push('头数需能被 KV 头数整除，且每头维度为至少 16 的偶数');if(c.intermediate_size&&c.intermediate_size<c.hidden_size)errors.push('MLP 中间维度需大于等于隐藏维度，或设为 0 自动计算');if(c.architecture==='qwen3_5'&&c.layers<4)errors.push('Qwen3.5 至少需要 4 层');}
    if(c.eval_steps>c.max_steps||c.save_steps>c.max_steps)errors.push('评估和保存间隔不能超过总更新步数');
    if(!['bf16','fp32'].includes(c.precision))errors.push('请选择 BF16 或 FP32');
    return errors;
  }
  function estimate(c){const h=c.hidden_size,l=c.layers,v=c.vocab_size,b=c.micro_batch*c.grad_accum*c.gpu_ids.length;let p;
    if((c.architecture||'gpt2')==='gpt2')p=v*h+c.seq_length*h+l*(12*h*h+13*h)+2*h;
    else{const d=h/c.heads,kv=c.kv_heads*d,i=c.intermediate_size||3*h,common=3*h*i+2*h;if(c.architecture==='qwen3')p=2*v*h+l*(2*h*h+2*h*kv+2*d+common)+h;
    else{const full=3*h*h+2*h*kv+2*d+common,linear=2*h*kv+3*h*h+2*h*c.heads+4*(2*kv+h)+2*c.heads+d+common;p=2*v*h+Math.floor(l/4)*full+(l-Math.floor(l/4))*linear+h;}}
    return {parameters:p,batch:b,tokensPerStep:b*c.seq_length,totalTokens:b*c.seq_length*c.max_steps};}
  function draft(c){return Object.fromEntries(Object.keys(defaults).filter(k=>!['dataset_id','gpu_ids'].includes(k)).map(k=>[k,c[k]]));}
  function restore(raw){const result={...defaults,gpu_ids:[]};if(raw&&typeof raw==='object'&&!raw.architecture)result.architecture='gpt2';if(!raw||typeof raw!=='object'||Array.isArray(raw))return result;for(const key of Object.keys(draft(defaults))){if(typeof raw[key]===typeof defaults[key]&&(typeof raw[key]!=='number'||Number.isFinite(raw[key])))result[key]=raw[key];}return result;}
  function csv(metrics){const keys=['step','loss','eval_loss','learning_rate','grad_norm','time'];return keys.join(',')+'\n'+metrics.map(m=>keys.map(k=>Number.isFinite(m[k])?m[k]:'').join(',')).join('\n');}
  const architectureNames={gpt2:'GPT-2',qwen3:'Qwen3',qwen3_5:'Qwen3.5'};
  const api={architectureNames,defaults,bounds,validate,estimate,draft,restore,csv};root.TrainLabUI=api;if(typeof module!=='undefined')module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
