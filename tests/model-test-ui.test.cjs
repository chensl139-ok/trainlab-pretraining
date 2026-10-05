const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
function workbench(options={}){
 const elements=new Map();
 const selects={'test-checkpoint':['final','checkpoint-5'],'test-device':['cpu','cuda:0'],'test-dataset':['','available']};
 for(const [id,values] of Object.entries({...selects,...options})){
  let current=values[0];elements.set('#'+id,{get value(){return current},set value(v){current=values.includes(v)?v:''}});
 }
 const context=vm.createContext({$:id=>{if(!elements.has(id))elements.set(id,{value:'',onchange(){},scrollIntoView(){}});return elements.get(id)},testBindingError:'',detailData:null,paintTestControls(){},toast:()=>{}});
 const source=fs.readFileSync('dist/gpu.js','utf8').split('function reuseTest(t){')[1].split('// Running tests')[0];
 vm.runInContext('function reuseTest(t){'+source,context);
 return {context,elements,run:code=>vm.runInContext(code,context)};
}
const config={mode:'score',checkpoint:'checkpoint-5',prompt:'',max_new_tokens:32,temperature:0,top_p:.9,seed:42,max_runtime_seconds:180,max_blocks:2,dataset_id:'available',device:'cuda',gpu_id:0};
test('reuse restores the original checkpoint, GPU and evaluation corpus',()=>{
 const b=workbench();b.context.item={config};b.run('reuseTest(item)');
 assert.equal(b.elements.get('#test-checkpoint').value,'checkpoint-5');assert.equal(b.elements.get('#test-device').value,'cuda:0');assert.equal(b.elements.get('#test-dataset').value,'available');assert.equal(b.run('testBindingError'),'');
});
test('missing original GPU or corpus requires explicit reselection instead of CPU or validation fallback',()=>{
 for(const options of [{'test-device':['cpu']},{'test-dataset':['']},{'test-checkpoint':['final']}]){
  const b=workbench(options);b.context.item={config};b.run('reuseTest(item)');assert.match(b.run('testBindingError'),/不可用/);
 }
});
