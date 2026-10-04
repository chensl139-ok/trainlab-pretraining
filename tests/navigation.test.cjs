const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {parse,views}=require('../dist/navigation.js');
test('navigation routes normalize legacy and malformed hashes safely',()=>{
  for(const view of views)assert.deepEqual(parse('#'+view),{view,chapter:0,hash:'#'+view});
  for(const value of ['', '#missing', '#learn/<script>', '#learn/-1'])assert.equal(parse(value).hash,'#training');
  assert.equal(parse('#training/4').hash,'#training');
});
test('knowledge chapters support stable deep links and bounded chapter indexes',()=>{
  assert.deepEqual(parse('#learn/4'),{view:'learn',chapter:4,hash:'#learn/4'});
  assert.equal(parse('#learn/0').hash,'#learn');
  assert.equal(parse('#learn/999').chapter,5);
});
test('both workspaces expose identical navigation destinations and accessibility controls',()=>{
  const pages=['index.html','gpu.html'].map(f=>fs.readFileSync(new URL('../dist/'+f,'file://'+__filename),'utf8'));
  const sidebars=pages.map(s=>s.match(/<aside[\s\S]*?<\/aside>/)[0]);
  assert.equal(sidebars[0],sidebars[1]);
  assert.equal((sidebars[0].match(/data-nav=/g)||[]).length,6);
  for(const page of pages){assert.match(page,/aria-controls="site-sidebar"/);assert.match(page,/src="navigation.js"/);assert.match(page,/href="navigation.css"/);}
  assert.equal((sidebars[0].match(/target="_blank" rel="noopener"/g)||[]).length,2);
});
