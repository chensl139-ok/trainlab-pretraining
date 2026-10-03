const test=require('node:test');
const assert=require('node:assert/strict');
const ui=require('../dist/gpu-core.js');
const valid=()=>({...ui.defaults,dataset_id:'a'.repeat(32),gpu_ids:[0]});
test('valid configuration and 8-card batch estimate',()=>{const c=valid();assert.deepEqual(ui.validate(c),[]);c.gpu_ids=[0,1,2,3,4,5,6,7];const e=ui.estimate(c);assert.equal(e.batch,128);assert.equal(e.tokensPerStep,32768);assert.equal(e.parameters,12318720);});
test('block invalid architecture, intervals and GPU choices',()=>{for(const patch of [{heads:5},{eval_steps:101},{gpu_ids:[0,0]},{gpu_ids:[]},{seq_length:NaN},{max_steps:3.5},{precision:'fp16'},{max_runtime_seconds:0},{name:' '}])assert.ok(ui.validate({...valid(),...patch}).length,JSON.stringify(patch));});
test('drafts exclude identity, credentials, dataset and GPU binding',()=>{const d=ui.draft({...valid(),token:'secret',subject:'admin'});for(const key of ['token','subject','dataset_id','gpu_ids'])assert.equal(key in d,false);const restored=ui.restore({...d,token:'secret',dataset_id:'a'.repeat(32),gpu_ids:[7],layers:'broken'});assert.equal(restored.layers,6);assert.equal(restored.dataset_id,'');assert.deepEqual(restored.gpu_ids,[]);assert.equal('token' in restored,false);});
test('reuse supports complete custom training parameters',()=>{const c={...valid(),layers:8,hidden_size:512,heads:8,seed:7,gradient_checkpointing:false,eval_steps:20,save_steps:50,warmup_ratio:.1,weight_decay:.02};const restored=ui.restore(ui.draft(c));for(const key of Object.keys(ui.draft(c)))assert.equal(restored[key],c[key]);});
test('CSV contains numeric metrics only and cannot introduce formulas',()=>{assert.equal(ui.csv([{step:1,loss:2.5,eval_loss:'=cmd',time:Infinity}]),'step,loss,eval_loss,learning_rate,grad_norm,time\n1,2.5,,,,');});
