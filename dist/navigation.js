/* Shared route semantics and accessible navigation for both workspaces. */
(function(root){
  'use strict';
  const views=['training','experiments','data','llm','learn'];
  function parse(hash){
    const match=/^#?(training|experiments|data|llm|learn)(?:\/(\d+))?$/.exec(hash||'');
    const view=match?match[1]:'training';
    const chapter=view==='learn'&&match?.[2]!==undefined?Math.min(5,Number(match[2])):0;
    return {view,chapter,hash:'#'+view+(chapter?'/'+chapter:'')};
  }
  if(typeof module!=='undefined')module.exports={parse,views};
  if(typeof document==='undefined')return;
  const sidebar=document.querySelector('#site-sidebar'),toggle=document.querySelector('.nav-toggle'),shell=document.querySelector('.shell'),backdrop=document.querySelector('.nav-backdrop'),close=sidebar.querySelector('.nav-close'),mobile=matchMedia('(max-width:800px)');
  let opened=false;
  function drawer(show,returnFocus=true){
    opened=show&&mobile.matches;
    document.body.classList.toggle('nav-open',opened);toggle.setAttribute('aria-expanded',String(opened));
    backdrop.hidden=!opened;shell.inert=opened;sidebar.inert=mobile.matches&&!opened;
    if(opened){sidebar.setAttribute('role','dialog');sidebar.setAttribute('aria-modal','true');requestAnimationFrame(()=>{if(opened)close.focus();});}
    else{sidebar.removeAttribute('role');sidebar.removeAttribute('aria-modal');if(returnFocus&&mobile.matches)toggle.focus();}
  }
  function select(view){sidebar.querySelectorAll('[data-nav]').forEach(a=>{const active=a.dataset.nav===view;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});}
  root.TrainLabNav={parse,select};
  const gpu=location.pathname.endsWith('/gpu.html');
  if(!gpu){sidebar.querySelectorAll('a[href^="/#"]').forEach(a=>a.setAttribute('href',a.hash));}
  select(gpu?'gpu':parse(location.hash).view);
  toggle.onclick=()=>drawer(!opened);close.onclick=()=>drawer(false);backdrop.onclick=()=>drawer(false);
  sidebar.addEventListener('click',event=>{
    const a=event.target.closest('a');if(!a||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
    // An active navigation item is a stable location, not a reset action.
    if(a.getAttribute('aria-current')==='page')event.preventDefault();
    if(opened)drawer(false);
  });
  document.addEventListener('keydown',event=>{
    if(!opened)return;
    if(event.key==='Escape'){event.preventDefault();drawer(false);}
    if(event.key==='Tab'){
      const items=[...sidebar.querySelectorAll('a,button')].filter(el=>el.getClientRects().length),first=items[0],last=items.at(-1);
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
    }
  });
  mobile.addEventListener('change',()=>drawer(false,false));drawer(false,false);
})(typeof window==='undefined'?globalThis:window);
