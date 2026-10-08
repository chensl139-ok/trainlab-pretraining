const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
// Exercise the actual connection/API lifecycle with controlled transport and DOM.
function browser(fetch,storageFailure=false){
  const store=new Map(),elements=new Map();
  const document={querySelector(selector){if(!elements.has(selector))elements.set(selector,{disabled:false,value:'',dataset:{},hidden:false,close(){},replaceChildren(){},classList:{add(){},remove(){}}});return elements.get(selector);},querySelectorAll(){return [];}};
  const context=vm.createContext({document,window:{TrainLabUI:{}},fetch,AbortController,AbortSignal,setTimeout:()=>0,clearTimeout(){},sessionStorage:{getItem:k=>store.get(k),setItem(k,v){if(storageFailure)throw new Error('Storage denied');store.set(k,v);},removeItem:k=>store.delete(k)}});
  const source=fs.readFileSync('dist/gpu.js','utf8').split('\nfunction paintSystem')[0];
  vm.runInContext(source+'\nfunction paintSystem(){}\nasync function refresh(){}\nfunction changeTab(){}',context);
  return {store,elements,run:code=>vm.runInContext(code,context)};
}
const response=(status=200)=>({status,ok:status===200,json:async()=>({identity:{subject:'test'}}),headers:{get:()=>null}});
test('successful login survives reload and explicit logout removes it',async()=>{
 const b=browser(async()=>response());await b.run("connect('test-credential')");assert.equal(b.run('savedCredential()'),'test-credential');b.run('disconnected()');assert.equal(b.store.size,0);assert.equal(b.run('connected'),false);
});
test('401 on restored credentials clears saved login and requires authentication',async()=>{
 const b=browser(async()=>response(401));b.run("saveCredential('revoked')");await b.run("connect(savedCredential(),true)");assert.equal(b.store.size,0);assert.equal(b.run('connected'),false);assert.equal(b.elements.get('#login').hidden,false);
});
test('network failure retains saved credential for retry without treating it as authenticated',async()=>{
 const b=browser(async()=>{throw new Error('network down');});b.run("saveCredential('retry-later')");await b.run("connect(savedCredential(),true)");assert.equal(b.run('savedCredential()'),'retry-later');assert.equal(b.run('connected'),false);
});
test('late authentication response cannot reconnect after logout',async()=>{
 let finish;const b=browser(()=>new Promise(resolve=>finish=resolve));const pending=b.run("connect('stale-credential')");b.run('disconnected()');finish(response());await pending;assert.equal(b.store.size,0);assert.equal(b.run('connected'),false);
});
test('blocked browser storage still permits login with an explicit fallback notice',async()=>{
 const b=browser(async()=>response(),true);await b.run("connect('memory-only')");assert.equal(b.run('connected'),true);assert.equal(b.store.size,0);assert.match(b.elements.get('#toast').textContent,/浏览器禁止会话存储/);
});

test('test view survives reload, is isolated by identity, and clears on logout',async()=>{
 const b=browser(async()=>response());await b.run("connect('credential')");
 b.run("selectedJob='a'.repeat(32);selectedTest='b'.repeat(32);activeTab='testing';saveView();selectedJob=null;selectedTest=null;activeTab='metrics';restoreView({subject:'test'})");
 assert.equal(b.run('selectedJob'),'a'.repeat(32));assert.equal(b.run('activeTab'),'testing');assert.equal(b.run('restoredTest.id'),'b'.repeat(32));
 b.run("selectedJob=null;restoreView({subject:'another-user'})");assert.equal(b.run('selectedJob'),null);
 b.run("restoreView({subject:'test'});disconnected()");assert.equal(b.store.size,0);
});

test('local mode enters automatically without sending or retaining credentials',async()=>{
 const calls=[];const b=browser(async(path,options)=>{calls.push({path,options});return {...response(),json:async()=>path==='/api/health'?{service:'trainlab-pretraining',auth_required:false,auth_mode:'local'}:{identity:{subject:'local-owner',project:'research',role:'admin'}}};});
 b.run("saveCredential('obsolete-personal-key')");await b.run('check()');
 assert.equal(b.run('connected'),true);assert.equal(b.run('token'),'');assert.equal(b.run('savedCredential()'),'');
 assert.equal(b.elements.get('#login').hidden,true);assert.equal(b.elements.get('#logout').hidden,true);
 assert.equal(calls.find(c=>c.path==='/api/system').options.headers.Authorization,undefined);
});
test('local mode works with storage blocked and reports access failures without a login form',async()=>{
 const b=browser(async(path)=>({...response(path==='/api/system'?403:200),json:async()=>path==='/api/health'?{service:'trainlab-pretraining',auth_required:false}:{detail:'local access denied'}}),true);
 await b.run('check()');assert.equal(b.run('connected'),false);assert.equal(b.elements.get('#login').hidden,true);assert.equal(b.elements.get('#offline').hidden,false);assert.match(b.elements.get('#error-text').textContent,/local access denied/);
 const good=browser(async(path)=>({...response(),json:async()=>path==='/api/health'?{service:'trainlab-pretraining',auth_required:false}:{identity:{subject:'local-owner'}}}),true);
 await good.run('check()');assert.equal(good.run('connected'),true);assert.equal(good.store.size,0);assert.equal(good.elements.has('#toast'),false);
});
