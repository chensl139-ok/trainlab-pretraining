'use strict';
const $=s=>document.querySelector(s),all=s=>[...document.querySelectorAll(s)],UI=window.TrainLabUI;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={queued:'排队中',running:'训练中',cancelling:'正在停止',cancelled:'已取消',failed:'失败',succeeded:'已完成',interrupted:'已中断'};
const number=x=>Number.isFinite(x)?x.toLocaleString('zh-CN'):'—';
const size=x=>!Number.isFinite(x)?'—':x>=1024**3?(x/1024**3).toFixed(1)+' GiB':x>=1024**2?(x/1024**2).toFixed(1)+' MiB':(x/1024).toFixed(1)+' KiB';
const date=x=>x?new Date(x).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}):'尚未开始';
let token='',connected=false,canWrite=false,session=0,system=null,datasets=[],selectedJob=null,detailData=null;
let refreshing=false,submitting=false,mutating=false,uploading=false,uploadRequest=null,selectedFile=null;
let offset=0,total=0,pageSize=12,listRevision=0,detailRevision=0,activeTab='metrics',pendingSubmission=null,dialogAction=null,dialogBusy=false,refreshQueued=false;
let testSubmitting=false,testRevision=0,selectedTest=null,testItems=[],pendingTest=null,testError='';
const requests=new Set();let toastTimer,searchTimer;
const credentialKey='trainlab-tab-credential-v1';
function savedCredential(){try{return sessionStorage.getItem(credentialKey)||'';}catch{return '';}}
function saveCredential(value){try{sessionStorage.setItem(credentialKey,value);return true;}catch{return false;}}
function clearCredential(){try{sessionStorage.removeItem(credentialKey);}catch{}}
function toast(message){clearTimeout(toastTimer);$('#toast').textContent=message;$('#toast').classList.add('show');toastTimer=setTimeout(()=>$('#toast').classList.remove('show'),4200);}
function showError(error){if(error?.stale)return;$('#error-text').textContent=typeof error==='string'?error:error.message;$('#error-banner').hidden=false;}
function clearError(){$('#error-banner').hidden=true;}
function disconnected(message='已断开连接，服务端任务继续运行'){
  clearCredential();session++;token='';connected=false;canWrite=false;system=null;datasets=[];selectedJob=null;detailData=null;selectedFile=null;pendingSubmission=null;
  requests.forEach(c=>c.abort());requests.clear();if(uploadRequest)uploadRequest.abort();uploadRequest=null;uploading=false;
  $('#console').hidden=true;$('#login').hidden=false;$('#logout').hidden=true;$('#refresh').hidden=true;$('#token').value='';$('#connection').textContent='未连接';$('#notice').textContent=message;
  $('#action-dialog').close();$('#job-detail').hidden=true;$('#detail-empty').hidden=false;$('#corpus-file').value='';$('#jobs').replaceChildren();$('#job-log').textContent='';$('#job-config').textContent='';$('#metric-rows').replaceChildren();$('#artifact-list').replaceChildren();delete $('#artifact-list').dataset.signature;$('#test-result').hidden=true;$('#test-result').replaceChildren();delete $('#test-result').dataset.signature;$('#test-history').replaceChildren();$('#test-prompt').value='';delete $('#test-checkpoint').dataset.job;testItems=[];selectedTest=null;pendingTest=null;$('#dataset_id').innerHTML='<option value="">连接后选择语料</option>';$('#gpu-select').replaceChildren();$('#dialog-body').replaceChildren();$('#upload-progress').hidden=true;$('#cancel-upload').hidden=true;$('#file-name').textContent='UTF-8 编码 · .jsonl';
}
function staleError(){const e=new Error('连接已变更');e.stale=true;return e;}
async function api(path,options={}){
  const version=session,controller=new AbortController();requests.add(controller);const timeout=setTimeout(()=>controller.abort(),options.timeoutMs||20000);
  try{
    const response=await fetch('/api'+path,{...options,signal:controller.signal,headers:{Authorization:'Bearer '+token,...options.headers}});
    if(version!==session)throw staleError();
    if(response.status===401){disconnected('凭据已过期或被吊销，请重新连接');throw new Error('访问凭据无效，请重新连接');}
    if(options.raw){if(!response.ok)throw new Error('下载失败，请刷新后重试');return response;}
    let result;try{result=await response.json();}catch{throw new Error('未收到有效 API 响应，请确认服务器地址与服务状态');}
    if(version!==session)throw staleError();
    if(!response.ok){const detail=result.detail;const msg=Array.isArray(detail)?detail.map(x=>x.msg).join('；'):detail||'请求失败';throw new Error(msg+(response.headers.get('X-Request-ID')?' · 请求 '+response.headers.get('X-Request-ID').slice(0,12):''));}
    return result;
  }catch(error){if(version!==session&&error.message!=='访问凭据无效，请重新连接')throw staleError();if(error.name==='AbortError')throw new Error('服务器响应超时，操作结果可能尚未返回。请刷新核对；重复提交会使用同一提交标识。');if(error instanceof TypeError)throw new Error('网络连接失败，请检查服务器或 SSH 隧道');throw error;}
  finally{clearTimeout(timeout);requests.delete(controller);}
}
async function check(){
  $('#retry-connection').disabled=true;$('#offline').hidden=true;$('#notice').textContent='正在检查训练服务…';
  try{const r=await fetch('/api/health',{signal:AbortSignal.timeout(8000)}),health=await r.json();if(!r.ok||health.service!=='trainlab-pretraining')throw new Error();$('#login').hidden=false;$('#connection').textContent='服务在线 · 待认证';$('#notice').textContent='训练服务可连接。输入个人凭据后查看机器资源与实验。';const saved=savedCredential();if(saved)await connect(saved,true);}
  catch{$('#login').hidden=true;$('#offline').hidden=false;$('#connection').textContent='未连接 GPU 机器';$('#notice').textContent='当前未连接训练后端。请部署到 Linux 机器，或检查服务器与 SSH 隧道。';}
  finally{$('#retry-connection').disabled=false;}
}
async function connect(credential,restoring=false){
  if($('#login-button').disabled)return;session++;const version=session;token=credential;$('#login-button').disabled=true;$('#login-button').textContent=restoring?'恢复连接中…':'连接中…';clearError();
  if(restoring)$('#notice').textContent='正在恢复当前标签页的登录状态…';
  try{const data=await api('/system');if(version!==session)return;connected=true;const remembered=saveCredential(token);$('#token').value='';$('#login').hidden=true;$('#offline').hidden=true;$('#console').hidden=false;$('#logout').hidden=false;$('#refresh').hidden=false;paintSystem(data,true);offset=0;await refresh();if(!remembered&&connected)toast('浏览器禁止会话存储，本次连接有效，刷新后需要重新输入凭据');}
  catch(error){if(!error.stale){if(version===session){token='';connected=false;}showError(error);if(restoring&&savedCredential())$('#notice').textContent='暂时无法恢复连接，已保留本标签页凭据；请检查网络后刷新重试。';}}
  finally{$('#login-button').disabled=false;$('#login-button').textContent='连接训练服务 →';}
}
$('#login').onsubmit=event=>{event.preventDefault();connect($('#token').value.trim());};
function paintSystem(data,initial=false){
  system=data;canWrite=['admin','operator'].includes(data.identity?.role);const identity=data.identity;
  $('#connection').textContent=identity?`${identity.subject} · ${identity.role==='viewer'?'只读':identity.project}`:'已连接';
  $('#notice').classList.remove('stale');$('#notice').textContent=`${data.control?.paused?'维护模式':data.blocked_reason||data.execution_blocked?'等待资源 / 检查':data.scheduler_alive?'调度器运行中':'调度器未就绪'} · ${data.control?.paused?'已停止新任务接收':!canWrite?'只读访问':data.gpus?.length?'可提交任务':'等待 GPU 就绪'} · 单机串行 DDP · 数据保留在服务器`;
  const gpus=data.gpus||[];$('#gpu-description').textContent=gpus.length?`${gpus.length} 张 GPU 在线`:'未检测到 GPU';
  $('#gpu-cards').innerHTML=gpus.length?gpus.map(g=>`<article class="gpu-card"><div class="gpu-top"><strong>GPU ${g.index}</strong><span>${number(g.utilization_percent)}% 利用率</span></div><p title="${esc(g.name)}">${esc(g.name)}</p><div class="vram">${(g.memory_used_mib/1024).toFixed(1)} <small>/ ${(g.memory_total_mib/1024).toFixed(1)} GiB</small></div><div class="progress-track"><div style="width:${Math.min(100,g.memory_used_mib/g.memory_total_mib*100)}%"></div></div><p>驱动 ${esc(g.driver)}</p></article>`).join(''):'<div class="gpu-empty"><span aria-hidden="true">▥</span><div><b>尚无可用的 NVIDIA GPU</b><p>可以准备语料和配置；提交训练前请检查驱动与容器 GPU 挂载。</p></div></div>';
  const previous=all('[name=gpu]:checked').map(x=>+x.value),newIds=gpus.map(g=>g.index).join(',');
  if(initial||$('#gpu-select').dataset.ids!==newIds){$('#gpu-select').dataset.ids=newIds;$('#gpu-select').innerHTML=gpus.length?gpus.map((g,i)=>`<label><input type="checkbox" name="gpu" value="${g.index}" ${(initial?i===0:previous.includes(g.index))?'checked':''}> ${g.index}</label>`).join(''):'<span class="mini-note">等待可用 GPU</span>';all('[name=gpu]').forEach(x=>x.onchange=updateForm);}
  $('#stat-disk').textContent=size(data.disk?.free);$('#stat-disk-note').textContent=data.disk?.free<data.limits?.min_free_bytes?'低于运行水位，请先释放空间':`最低保留 ${size(data.limits?.min_free_bytes)}`;
  paintOperations(data);updateForm();
}
function paintOperations(data){
  const paused=!!data.control?.paused,blocked=!!(data.blocked_reason||data.scheduler_error||data.execution_blocked),alive=!!data.scheduler_alive;
  $('#environment-badge').textContent=data.environment==='production'?'生产配置':'开发预览';
  $('#environment-badge').title='运行配置标识，不代表 GPU 已通过验收';
  $('#operations').classList.toggle('paused',paused);$('#operations').classList.toggle('blocked',blocked||!alive);
  $('#scheduler-state').textContent=!alive?'调度器异常':paused?'维护模式':blocked?'等待资源 / 检查':'运行中';
  $('#scheduler-detail').textContent=data.scheduler_error||data.blocked_reason||(data.execution_blocked?'工作进程仍持有执行锁，禁止启动新任务；等待退出或检查容器状态。':paused?'新训练和测试已停止接收，队列暂停派发。'+(data.control.reason?' 原因：'+data.control.reason:''):'任务按提交顺序串行执行。启动前检查磁盘和 GPU 占用；异常不会自动重跑。');
  $('#worker-kind').textContent=data.worker_kind==='jobs'?'训练任务正在运行':data.worker_kind==='model_tests'?'模型测试正在运行':'当前空闲';
  $('#runtime-guards').textContent=`磁盘保留 ${size(data.limits?.min_free_bytes)} · 日志上限 ${size(data.limits?.max_log_bytes)}`;
  $('#maintenance-toggle').hidden=$('#diagnostics').hidden=data.identity?.role!=='admin';
  $('#maintenance-toggle').textContent=paused?'退出维护模式':'进入维护模式';
  $('#operations-note').textContent=paused?'备份前请等待当前任务结束并停止服务。维护模式会跨重启保留，恢复派发需手动退出。':'GPU 显存占用超过 '+number(data.limits?.gpu_idle_mib)+' MiB 时等待释放；此门槛不等同于模型显存容量估算。';
}
$('#maintenance-toggle').onclick=()=>{
  const paused=!system?.control?.paused;
  openDialog(paused?'进入维护模式？':'恢复任务调度？',paused?'<p>当前训练或测试继续运行；排队任务保留但不启动。新训练、恢复训练和模型测试将被拒绝。设置在重启后保留。</p><label class="field"><span>维护说明</span><input id="maintenance-reason" maxlength="200" value="部署升级 / 维护检查"></label>':'<p>排队任务将按顺序继续启动，并重新允许提交训练和测试。请确认维护已完成。</p>',paused?'进入维护模式':'恢复调度',async()=>{
    const control=await api('/scheduler/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused,reason:paused?$('#maintenance-reason').value:''})});
    system.control=control;paintOperations(system);await refresh();toast(paused?'已进入维护模式':'任务调度已恢复');
  });
};
$('#diagnostics').onclick=async()=>{
  const version=session;$('#diagnostics').disabled=true;
  try{const report=await api('/operations');if(version!==session)return;
    openDialog('运行诊断',`<p>生成于 ${date(report.generated_at)}。仅检查当前服务状态，不代表目标 GPU 验收已完成。</p><div class="diagnostics-grid"><div>数据库完整性<strong>${esc(report.database_integrity)}</strong></div><div>调度器<strong>${report.scheduler_alive?'在线':'异常'} · ${report.control.paused?'维护中':'可派发'}</strong></div><div>可用磁盘<strong>${size(report.disk.free)}</strong></div><div>主机可用内存<strong>${size(report.host_memory.available)}</strong></div></div><p class="mini-note">报告不包含凭据、提示词和语料内容；包含设备与版本信息。</p><details><summary>查看完整诊断字段</summary><pre class="diagnostic-json">${esc(JSON.stringify(report,null,2))}</pre></details>`,'下载诊断 JSON',async()=>{downloadBlob(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}),'trainlab-diagnostics.json');});$('#dialog-kicker').textContent='服务自检';$('#dialog-back').textContent='关闭';
  }catch(error){showError(error);}finally{$('#diagnostics').disabled=false;}
};
function readConfig(){const c={...UI.defaults};for(const key of Object.keys(UI.defaults)){if(key==='gpu_ids'){c[key]=all('[name=gpu]:checked').map(x=>+x.value);continue;}if(key==='max_runtime_seconds'){c[key]=Number($('#runtime_minutes').value)*60;continue;}const el=$('#'+key);c[key]=typeof UI.defaults[key]==='number'?Number(el.value):typeof UI.defaults[key]==='boolean'?el.checked:el.value;}c.name=c.name.trim();return c;}
function inferPreset(c){return Object.entries({tiny:[6,384,6],small:[12,768,12],medium:[24,1024,16]}).find(([,v])=>v[0]===c.layers&&v[1]===c.hidden_size&&v[2]===c.heads)?.[0]||'custom';}
function applyConfig(c){c={...UI.defaults,...c,architecture:c.architecture||'gpt2'};for(const key of Object.keys(UI.defaults)){if(key==='gpu_ids'){all('[name=gpu]').forEach(el=>el.checked=(c.gpu_ids||[]).includes(+el.value));continue;}if(key==='max_runtime_seconds'){$('#runtime_minutes').value=c[key]/60;continue;}const el=$('#'+key);if(typeof UI.defaults[key]==='boolean')el.checked=c[key];else el.value=c[key]??UI.defaults[key];}$('#preset').value=inferPreset(c);$('#advanced').open=inferPreset(c)==='custom';datasetMeta();updateForm();}
function errorsFor(c){const errors=UI.validate(c);if(!datasets.some(d=>d.id===c.dataset_id)&&c.dataset_id)errors.push('当前账号无权访问所选语料，请重新选择');if(c.gpu_ids.some(id=>!system?.gpus.some(g=>g.index===id)))errors.push('所选 GPU 已离线，请重新选择');if(system?.control?.paused)errors.push('维护模式下暂不接收新任务');if(!system?.scheduler_alive)errors.push('调度器未就绪，请检查服务');if(system?.disk?.free<system?.limits?.min_free_bytes)errors.push('磁盘低于运行水位，请释放空间');return errors;}
function updateForm(){
  const architecture=$('#architecture').value;$('#kv-field').hidden=$('#mlp-field').hidden=architecture==='gpt2';$('#architecture-note').textContent=({qwen3_5:'Qwen3.5 纯文本稠密架构 · Gated DeltaNet + 全注意力混合。随机初始化、自训 BPE；当前为参考内核，先小规模验证。',qwen3:'Qwen3 稠密架构 · GQA + QK-Norm + RoPE + SwiGLU。随机初始化、自训 BPE，不加载官方预训练权重。',gpt2:'GPT-2 架构 · 标准多头注意力 + 学习式位置编码。保持已有实验和检查点兼容。'})[architecture];
  const c=readConfig(),e=UI.estimate(c),errors=errorsFor(c);$('#budget').innerHTML=`<div class="budget-values"><span>参数量（估计）<b>${Number.isFinite(e.parameters)?(e.parameters/1e6).toFixed(1)+'M':'—'}</b></span><span>有效 Batch<b>${number(e.batch)}</b></span></div>每次更新 ${number(e.tokensPerStep)} tokens<br>计划处理 ${Number.isFinite(e.totalTokens)?(e.totalTokens/1e6).toFixed(2)+'M':'—'} tokens（含重复遍历）<small>按词表上限估计；此值不保证显存足够。</small><small>启动磁盘预留约 ${size(e.parameters*48+(datasets.find(d=>d.id===c.dataset_id)?.bytes||0)*8+(system?.limits?.min_free_bytes||0)+(system?.limits?.max_log_bytes||0))}（检查点、语料与水位估算）</small>`;
  $('#form-errors').textContent=!canWrite&&connected?'当前账号为只读；仍可查看和导出实验。':errors.slice(0,3).join('；');
  $('#submit-job').disabled=!connected||!canWrite||submitting||errors.length>0;$('#submit-job').textContent=submitting?'正在提交…':'检查配置并提交 →';
  $('#open-catalog').disabled=!connected||!canWrite;$('#upload').disabled=!connected||!canWrite||uploading||!selectedFile;$('#upload').textContent=uploading?'上传校验中…':'上传并校验';$('#select-all-gpus').disabled=!system?.gpus.length;$('#preview-dataset').disabled=!datasets.some(d=>d.id===$('#dataset_id').value);
}
$('#job-form').addEventListener('input',()=>{if(['layers','hidden_size','heads'].includes(document.activeElement?.id))$('#preset').value='custom';updateForm();});
$('#preset').onchange=()=>{const preset={tiny:[6,384,6],small:[12,768,12],medium:[24,1024,16]}[$('#preset').value];if(preset){['layers','hidden_size','heads'].forEach((id,i)=>$('#'+id).value=preset[i]);}else $('#advanced').open=true;updateForm();};
$('#select-all-gpus').onclick=()=>{const inputs=all('[name=gpu]'),checked=inputs.every(x=>x.checked);inputs.forEach(x=>x.checked=!checked);updateForm();};
function datasetMeta(){const d=datasets.find(d=>d.id===$('#dataset_id').value);$('#dataset-meta').textContent=d?`${number(d.rows)} 篇 · ${size(d.bytes)}`:'尚未选择数据集';$('#preview-dataset').disabled=!d;}
$('#dataset_id').onchange=()=>{datasetMeta();updateForm();};
function paintDatasets(){const previous=$('#dataset_id').value;$('#dataset_id').innerHTML='<option value="">选择已上传语料</option>'+datasets.map(d=>`<option value="${esc(d.id)}">${esc(d.name)} · ${number(d.rows)} 篇</option>`).join('');$('#dataset_id').value=datasets.some(d=>d.id===previous)?previous:datasets[0]?.id||'';datasetMeta();updateForm();}
function chooseFile(file){if(uploading)return;selectedFile=file||null;$('#file-name').textContent=file?`${file.name} · ${size(file.size)}`:'UTF-8 编码 · .jsonl';$('#upload-state').textContent=file?'文件已选择，上传后在服务器校验。':'示例仅验证流程，不代表真实语言能力。';updateForm();}
$('#corpus-file').onchange=()=>chooseFile($('#corpus-file').files[0]);
for(const type of ['dragover','dragenter'])$('#drop-zone').addEventListener(type,e=>{e.preventDefault();$('#drop-zone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('#drop-zone').addEventListener(type,e=>{e.preventDefault();$('#drop-zone').classList.remove('dragging');if(type==='drop')chooseFile(e.dataTransfer.files[0]);});
$('#upload').onclick=async()=>{
  if(uploading||!canWrite||!selectedFile)return;if(selectedFile.size>100*1024**2||!selectedFile.size){showError('请选择非空且不超过 100 MiB 的语料文件');return;}
  const version=session;uploading=true;clearError();updateForm();$('#cancel-upload').hidden=false;$('#upload-progress').hidden=false;$('#upload-progress').value=0;$('#upload-state').textContent='正在上传…';
  try{
    const data=await new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();uploadRequest=xhr;xhr.open('POST','/api/datasets?name='+encodeURIComponent(selectedFile.name));xhr.setRequestHeader('Authorization','Bearer '+token);xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.timeout=180000;
      xhr.upload.onprogress=e=>{if(version!==session)return;if(e.lengthComputable){const percent=Math.round(e.loaded/e.total*100);$('#upload-progress').value=percent;$('#upload-state').textContent=percent===100?'传输完成，正在校验语料…':`正在上传 ${percent}%`;}};
      xhr.onload=()=>{if(version!==session)return reject(staleError());let result;try{result=JSON.parse(xhr.responseText);}catch{return reject(new Error('上传服务响应异常'));}if(xhr.status===401){disconnected('凭据已失效，请重新连接');return reject(new Error('访问凭据无效'));}xhr.status>=200&&xhr.status<300?resolve(result):reject(new Error(typeof result.detail==='string'?result.detail:'语料校验失败'));};
      xhr.onerror=()=>reject(new Error('上传连接失败，请刷新核对语料是否已保存'));xhr.ontimeout=()=>reject(new Error('上传超时，请刷新核对是否已保存后重试'));xhr.onabort=()=>reject(new Error('上传已取消；若服务器已保存，可在数据集列表核对'));xhr.send(selectedFile);
    });
    if(version!==session)throw staleError();datasets=await api('/datasets');paintDatasets();$('#dataset_id').value=data.id;datasetMeta();$('#upload-state').textContent=`已保存 ${number(data.rows)} 篇 · ${size(data.bytes)}`;selectedFile=null;$('#corpus-file').value='';$('#file-name').textContent='继续选择或拖入下一份语料';toast('语料已保存，可预览并配置训练');await refresh();
  }catch(error){if(version===session){$('#upload-state').textContent=error.message;showError(error);}}
  finally{if(version===session){uploading=false;uploadRequest=null;$('#cancel-upload').hidden=true;$('#upload-progress').hidden=true;updateForm();}}
};
$('#cancel-upload').onclick=()=>uploadRequest?.abort();
function openDialog(title,body,label,action){dialogAction=action;$('#dialog-title').textContent=title;$('#dialog-kicker').textContent=action?'操作确认':'数据检查';$('#dialog-body').innerHTML=body+'<p id="dialog-error" class="job-error" role="alert" hidden></p>';$('#dialog-confirm').textContent=label;$('#dialog-confirm').hidden=!action;$('#dialog-back').textContent=action?'返回修改':'关闭';$('#action-dialog').classList.toggle('catalog-dialog',title==='常用文本数据集');$('#action-dialog').showModal();$('#dialog-back').focus({preventScroll:true});$('#action-dialog').scrollTop=0;}
$('#dialog-back').onclick=()=>$('#action-dialog').close();
$('#action-dialog').addEventListener('cancel',e=>{if(dialogBusy)e.preventDefault();});
$('#action-dialog').addEventListener('close',()=>{dialogAction=null;});
$('#dialog-confirm').onclick=async()=>{if(!dialogAction||dialogBusy)return;const action=dialogAction;dialogBusy=true;$('#dialog-confirm').disabled=true;$('#dialog-back').disabled=true;$('.dialog-close').disabled=true;try{if($('#dialog-error'))$('#dialog-error').hidden=true;await action();$('#action-dialog').close();}catch(error){if(!error.stale&&$('#action-dialog').open){$('#dialog-error').textContent=error.message;$('#dialog-error').hidden=false;}else showError(error);}finally{dialogBusy=false;$('#dialog-confirm').disabled=false;$('#dialog-back').disabled=false;$('.dialog-close').disabled=false;}};
$('#preview-dataset').onclick=async()=>{const id=$('#dataset_id').value;$('#preview-dataset').disabled=true;try{const data=await api('/datasets/'+id+'/preview');const d=data.dataset;openDialog('语料预览',`<b>${esc(d.name)}</b><p>${number(d.rows)} 篇 · ${size(d.bytes)} · ${date(d.created_at)}</p><div class="digest">SHA256 · ${esc(d.sha256)}</div>${data.provenance?`<div class="source-note"><b>${esc(data.provenance.source.name)}</b> · ${esc(data.provenance.split)}<br>${esc(data.provenance.source.license)}<br>${esc(data.provenance.selection)}<br><button type="button" id="export-source" class="text-button">下载来源记录 JSON</button></div>`:String()}${data.samples.map((s,i)=>`<div class="sample-heading">样本 ${i+1}${s.truncated?' · 已截断至 800 字符':''}</div><pre class="sample-text">${esc(s.text)}</pre>`).join('')}<p class="mini-note">最多显示前三篇；预览不替代数据质量与污染检查。</p>`,'',null);if($('#export-source'))$('#export-source').onclick=async()=>{try{const source=await api('/datasets/'+id+'/source');downloadBlob(new Blob([JSON.stringify(source,null,2)],{type:'application/json'}),'source-'+id.slice(0,8)+'.json');}catch(error){$('#dialog-error').textContent=error.message;$('#dialog-error').hidden=false;}};}catch(error){showError(error);}finally{datasetMeta();}};
$('#save-draft').onclick=()=>{try{localStorage.setItem('trainlab-gpu-draft-v1',JSON.stringify(UI.draft(readConfig())));$('#draft-state').textContent='参数草稿已保存在此浏览器；不含凭据和语料内容。';toast('草稿已保存');}catch{showError('浏览器无法保存草稿，请检查存储权限');}};
$('#restore-draft').onclick=()=>{try{const raw=localStorage.getItem('trainlab-gpu-draft-v1');if(!raw)return toast('此浏览器还没有保存的草稿');const current=readConfig();applyConfig({...UI.restore(JSON.parse(raw)),dataset_id:current.dataset_id,gpu_ids:current.gpu_ids});$('#draft-state').textContent='草稿已载入，已保留当前语料与 GPU 选择，请检查参数。';}catch{showError('草稿格式无效，请重置参数后重新保存');}};
$('#reset-config').onclick=()=>{openDialog('重置训练参数？','<p>恢复入门模型的默认参数，保留当前数据集和 GPU 选择。已保存的本机草稿不受影响。</p>','确认重置',async()=>{const c=readConfig();applyConfig({...UI.defaults,dataset_id:c.dataset_id,gpu_ids:c.gpu_ids});});};
$('#job-form').onsubmit=event=>{
  event.preventDefault();if(submitting||!canWrite)return;const c=readConfig(),errors=errorsFor(c);if(errors.length)return showError(errors.join('；'));const e=UI.estimate(c),dataset=datasets.find(d=>d.id===c.dataset_id);
  openDialog('确认预训练任务',`<div class="review-grid"><div><small>任务名称</small><b>${esc(c.name)}</b></div><div><small>语料</small><b>${esc(dataset.name)}</b></div><div><small>模型 / 参数（估计）</small><b>${UI.architectureNames[c.architecture]} · ${c.layers} 层 / ${(e.parameters/1e6).toFixed(1)}M</b></div><div><small>GPU / 精度</small><b>${c.gpu_ids.join(', ')} · ${c.precision.toUpperCase()}</b></div><div><small>有效 batch / 更新步数</small><b>${e.batch} / ${number(c.max_steps)}</b></div><div><small>最长运行时间</small><b>${c.max_runtime_seconds/60} 分钟</b></div></div><p class="review-note">从随机权重开始，任务将进入服务器队列。关闭页面不影响训练；此配置不保证显存足够。</p>`,'确认提交',()=>submit(c));
};
async function submit(c){if(submitting)return;const errors=errorsFor(c);if(errors.length)throw new Error(errors.join('；'));const version=session,payload=JSON.stringify(c);if(!pendingSubmission||pendingSubmission.payload!==payload)pendingSubmission={payload,key:[...crypto.getRandomValues(new Uint8Array(16))].map(x=>x.toString(16).padStart(2,'0')).join('')};submitting=true;updateForm();try{const job=await api('/jobs',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':pendingSubmission.key},body:payload});pendingSubmission=null;selectedJob=job.id;offset=0;$('#job-search').value='';$('#job-filter').value='';toast('任务已入队');await refresh();}finally{if(version===session){submitting=false;updateForm();}else submitting=false;}}
async function loadJobs(){const revision=++listRevision,version=session;const query=new URLSearchParams({q:$('#job-search').value.trim(),status:$('#job-filter').value,offset,limit:pageSize});const page=await api('/jobs/page?'+query);if(revision!==listRevision||version!==session)return;
  total=page.total;if(offset>=total&&offset>0){offset=Math.max(0,Math.floor(Math.max(0,total-1)/pageSize)*pageSize);return loadJobs();}
  $('#job-count').textContent=total+' 个任务';$('#page-info').textContent=total?`${offset+1}–${Math.min(offset+pageSize,total)} / ${total}`:'0 个任务';$('#prev-page').disabled=offset===0;$('#next-page').disabled=offset+pageSize>=total;
  if(!selectedJob&&page.items.length)selectedJob=page.items[0].id;
  $('#jobs').innerHTML=page.items.length?page.items.map(j=>`<button class="job-row ${selectedJob===j.id?'selected':''}" data-job="${j.id}" aria-pressed="${selectedJob===j.id}"><div><strong>${esc(j.config.name)}</strong><small>${UI.architectureNames[j.config.architecture||'gpt2']} · ${j.config.layers} 层 · ${j.config.hidden_size} 维 · ${j.config.gpu_ids.length?j.config.gpu_ids.length+' 卡':'CPU 验证'} · ${date(j.created_at)}</small></div><span class="badge ${esc(j.status)}">${names[j.status]||esc(j.status)}</span></button>`).join(''):`<div class="empty"><b>${query.get('q')||query.get('status')?'没有匹配的任务':'开始你的第一个训练实验'}</b>${query.get('q')||query.get('status')?'尝试其他关键词或任务状态。':'上传语料并完成配置，训练记录会保存在这里。'}</div>`;
  all('[data-job]').forEach(button=>button.onclick=()=>{selectedJob=button.dataset.job;detailData=null;all('[data-job]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',b===button?'true':'false');});$('#detail-title').textContent='正在加载任务…';$('#job-detail').hidden=true;$('#detail-empty').hidden=false;loadDetail().catch(showError);});
}
async function refresh(){
  if(!connected)return;if(refreshing){refreshQueued=true;return;}refreshing=true;const version=session;$('#refresh').disabled=true;
  try{const [s,ds,overview]=await Promise.all([api('/system'),api('/datasets'),api('/overview')]);if(version!==session)return;paintSystem(s);datasets=ds;paintDatasets();const tests=overview.model_tests||{};$('#test-queue-summary').textContent=`${(tests.running||0)+(tests.cancelling||0)} 个运行 · ${tests.queued||0} 个排队 · ${(tests.failed||0)+(tests.interrupted||0)} 个需检查`;$('#stat-active').textContent=number((overview.jobs.running||0)+(overview.jobs.cancelling||0));$('#stat-queue').textContent=`${overview.jobs.queued||0} 个排队中`;$('#stat-complete').textContent=number(overview.jobs.succeeded||0);$('#stat-total').textContent=`共 ${overview.total_jobs} 个实验 · ${(overview.jobs.failed||0)+(overview.jobs.interrupted||0)} 个需检查`;$('#stat-datasets').textContent=number(overview.dataset_count);$('#stat-data-size').textContent=size(overview.dataset_bytes);await loadJobs();await loadDetail();if(version!==session)return;$('#last-sync').textContent='同步于 '+new Date().toLocaleTimeString('zh-CN',{hour12:false});}
  catch(error){if(version===session){$('#notice').classList.add('stale');$('#notice').textContent='同步中断 · 当前显示最近成功获取的数据；请检查网络并点击刷新';showError(error);}}
  finally{refreshing=false;$('#refresh').disabled=false;if(refreshQueued){refreshQueued=false;if(connected)refresh();}}
}
async function loadDetail(){
  if(!selectedJob)return;const jid=selectedJob,revision=++detailRevision;const [j,log]=await Promise.all([api('/jobs/'+jid),api('/jobs/'+jid+'/logs')]);if(jid!==selectedJob||revision!==detailRevision)return;
  detailData=j;$('#detail-empty').hidden=true;$('#job-detail').hidden=false;$('#detail-title').textContent=j.config.name;$('#job-status').textContent=names[j.status]||j.status;$('#job-status').className='badge '+j.status;$('#detail-id').textContent='ID · '+j.id.slice(0,12);$('#detail-time').textContent=(j.summary?.mode==='CPU smoke test'?'CPU 流程验证 · ':'')+'创建 '+date(j.created_at);$('#checkpoint-count').textContent=`${j.checkpoints.length} 个完整检查点`;
  $('#cancel-job').hidden=!canWrite||!['queued','running'].includes(j.status);$('#resume-job').hidden=!canWrite||!['failed','cancelled','interrupted'].includes(j.status)||!j.checkpoints.length;$('#cancel-job').disabled=mutating;$('#resume-job').disabled=mutating;$('#reuse-config').disabled=!canWrite;
  $('#job-error').hidden=!j.error;$('#job-error').textContent=j.error||'';const m=j.metrics||[],loss=[...m].reverse().find(x=>Number.isFinite(x.loss)),val=[...m].reverse().find(x=>Number.isFinite(x.eval_loss)),step=m.reduce((v,x)=>Math.max(v,Number.isFinite(x.step)?x.step:0),0);
  $('#metric-loss').textContent=loss?loss.loss.toFixed(4):'—';$('#metric-val').textContent=val?val.eval_loss.toFixed(4):'—';$('#metric-step').textContent=number(step);$('#job-progress').value=Math.min(100,100*step/j.config.max_steps);$('#progress-label').textContent=`${step} / ${j.config.max_steps}`;$('#chart-empty').hidden=!!(loss||val);$('#perplexity').textContent=j.summary?.perplexity?`验证 Perplexity ${j.summary.perplexity.toFixed(2)} · 仅在相同分词器与验证集内比较`:'尚无最终评估结果；训练 loss 不代表独立测试集能力。';
  $('#metric-rows').innerHTML=m.slice(-40).reverse().map(x=>`<tr><td>${number(x.step)}</td><td>${Number.isFinite(x.loss)?x.loss.toFixed(4):'—'}</td><td>${Number.isFinite(x.eval_loss)?x.eval_loss.toFixed(4):'—'}</td><td>${Number.isFinite(x.learning_rate)?x.learning_rate.toExponential(2):'—'}</td></tr>`).join('');$('#export-metrics').disabled=!m.length;
  const logEl=$('#job-log'),text=log.text||'暂无日志。任务可能还在队列中。';if(logEl.textContent!==text){const top=logEl.scrollTop;logEl.textContent=text;logEl.scrollTop=$('#follow-log').checked?logEl.scrollHeight:top;}
  $('#job-config').textContent=JSON.stringify(j.config,null,2);paintArtifacts(j);drawChart();paintTestControls(j);if(activeTab==='testing')await loadTests();
}
function paintArtifacts(j){const items=j.artifacts.filter(a=>!a.path.startsWith('prepared/'));const signature=JSON.stringify([j.id,items]);if($('#artifact-list').dataset.signature===signature)return;$('#artifact-list').dataset.signature=signature;$('#artifact-list').innerHTML=items.length?items.map(a=>`<button data-artifact="${esc(a.path)}"><span>↓ ${esc(a.path)}</span><small>${size(a.bytes)}</small></button>`).join(''):'<div class="empty">暂无可下载文件，等待训练写入产物。</div>';all('[data-artifact]').forEach(button=>button.onclick=()=>downloadArtifact(j.id,items.find(a=>a.path===button.dataset.artifact),button));}
function downloadBlob(blob,filename){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;a.hidden=true;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);toast('已发起下载，请查看浏览器下载记录');}
async function downloadArtifact(jid,item,button){if(!item)return;if(item.bytes>256*1024**2){showError('文件超过浏览器 256 MiB 下载上限，请从服务器 output 目录复制，或使用带鉴权的下载 API');return;}button.disabled=true;const version=session;try{const r=await api('/jobs/'+jid+'/artifact/'+item.path.split('/').map(encodeURIComponent).join('/'),{raw:true}),blob=await r.blob();if(version!==session)return;downloadBlob(blob,item.path.split('/').pop());toast('已发起文件下载，请保持模型目录结构');}catch(error){showError(error);}finally{button.disabled=false;}}
function changeTab(tab){activeTab=tab;all('[data-tab]').forEach(b=>{const yes=b.dataset.tab===tab;b.setAttribute('aria-selected',yes?'true':'false');b.tabIndex=yes?0:-1;$('#tab-'+b.dataset.tab).hidden=!yes;});if(tab==='testing'){if(detailData)paintTestControls(detailData);loadTests().catch(showError);}if(tab==='metrics')drawChart();if(tab==='logs'&&$('#follow-log').checked)$('#job-log').scrollTop=$('#job-log').scrollHeight;}
all('[data-tab]').forEach(button=>{button.onclick=()=>changeTab(button.dataset.tab);button.onkeydown=e=>{const tabs=all('[data-tab]'),i=tabs.indexOf(button);let n;if(e.key==='ArrowRight')n=(i+1)%tabs.length;if(e.key==='ArrowLeft')n=(i+tabs.length-1)%tabs.length;if(e.key==='Home')n=0;if(e.key==='End')n=tabs.length-1;if(n!==undefined){e.preventDefault();tabs[n].focus();changeTab(tabs[n].dataset.tab);}};});
$('#reuse-config').onclick=()=>{if(!detailData)return;const c=detailData.config;applyConfig({...c,name:(c.name+' · 副本').slice(0,80)});$('#job-form').scrollIntoView({behavior:'smooth',block:'start'});$('#name').focus({preventScroll:true});toast('已复用完整配置，请确认当前语料与 GPU 可用后提交');};
function askMutation(action){if(!detailData||mutating)return;const j=detailData;openDialog(action==='cancel'?'停止这个训练任务？':'从检查点恢复训练？',`<p><b>${esc(j.config.name)}</b></p><p>${action==='cancel'?'运行中的进程将终止，未完成的检查点不会用于恢复；排队任务将被取消。':'将新建任务，保留原始数据、架构和训练参数，并从最近完整检查点恢复。'}</p>`,action==='cancel'?'确认停止':'确认恢复',async()=>{mutating=true;$('#cancel-job').disabled=true;$('#resume-job').disabled=true;try{const result=await api('/jobs/'+j.id+'/'+action,{method:'POST'});selectedJob=result.id;await refresh();toast(action==='cancel'?'停止请求已处理':'恢复任务已入队');}finally{mutating=false;$('#cancel-job').disabled=false;$('#resume-job').disabled=false;}});}
$('#cancel-job').onclick=()=>askMutation('cancel');$('#resume-job').onclick=()=>askMutation('resume');
$('#export-config').onclick=()=>{if(detailData)downloadBlob(new Blob([JSON.stringify(detailData.config,null,2)],{type:'application/json'}),'trainlab-'+detailData.id.slice(0,8)+'.json');};
$('#export-metrics').onclick=()=>{if(detailData)downloadBlob(new Blob(['\ufeff'+UI.csv(detailData.metrics)],{type:'text/csv;charset=utf-8'}),'metrics-'+detailData.id.slice(0,8)+'.csv');};
$('#download-log').onclick=()=>{if(detailData)downloadBlob(new Blob([$('#job-log').textContent],{type:'text/plain;charset=utf-8'}),'log-tail-'+detailData.id.slice(0,8)+'.txt');};
function drawChart(){if(!detailData||activeTab!=='metrics')return;const canvas=$('#server-chart'),rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;if(w<1)return;const dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const metrics=detailData.metrics||[],values=metrics.flatMap(x=>[x.loss,x.eval_loss]).filter(Number.isFinite),max=Math.max(1,...values)*1.1,left=34,top=16,pw=w-46,ph=h-40;ctx.font='10px sans-serif';ctx.fillStyle='#a0a8b5';for(let i=0;i<5;i++){const y=top+ph*i/4;ctx.strokeStyle='#eef0f5';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-8,y);ctx.stroke();ctx.fillText((max*(1-i/4)).toFixed(1),0,y+3);ctx.fillText(Math.round(detailData.config.max_steps*i/4),left+pw*i/4-8,h-2);}for(const [key,color]of[['loss','#947bdb'],['eval_loss','#56b198']]){const points=metrics.filter(x=>Number.isFinite(x[key])&&Number.isFinite(x.step));ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;points.forEach((x,i)=>{const px=left+pw*x.step/detailData.config.max_steps,py=top+ph*(1-x[key]/max);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();for(const x of points){ctx.fillStyle=color;ctx.beginPath();ctx.arc(left+pw*x.step/detailData.config.max_steps,top+ph*(1-x[key]/max),2,0,Math.PI*2);ctx.fill();}}}
function search(){offset=0;loadJobs().catch(showError);}
$('#job-search').oninput=()=>{clearTimeout(searchTimer);listRevision++;searchTimer=setTimeout(search,250);};$('#job-filter').onchange=search;$('#prev-page').onclick=()=>{offset=Math.max(0,offset-pageSize);loadJobs().catch(showError);};$('#next-page').onclick=()=>{offset+=pageSize;loadJobs().catch(showError);};
$('#dismiss-error').onclick=clearError;$('#logout').onclick=()=>{disconnected();clearError();};$('#retry-connection').onclick=check;$('#refresh').onclick=()=>{clearError();refresh();};
$('#auto-refresh').onchange=()=>{if($('#auto-refresh').checked)refresh();else $('#last-sync').textContent='自动刷新已暂停';};
setInterval(()=>{if(!document.hidden&&$('#auto-refresh').checked)refresh();},5000);document.addEventListener('visibilitychange',()=>{if(!document.hidden&&$('#auto-refresh').checked)refresh();});window.addEventListener('resize',drawChart);check();

$('#open-catalog').onclick=async()=>{
  if(!canWrite)return;const version=session;$('#open-catalog').disabled=true;
  try{
    const sources=await api('/dataset-catalog');
    openDialog('常用文本数据集',`<p>导入后即可选择训练。每行保留一篇完整文档，仅取小样本，不下载整库。</p><div class="catalog-grid">${sources.map(s=>`<label class="catalog-card"><input type="radio" name="catalog-source" value="${esc(s.key)}" ${s.key==='ms-mini-pretrain'?'checked':''}><b>${esc(s.name)}</b><small>${esc(s.provider)} · ${esc(s.language)} · ${s.network?'国内数据源':'离线可用'}</small><p>${esc(s.description)}</p><small>${esc(s.license)}</small><a href="${esc(s.url)}" target="_blank" rel="noopener">${s.network?'官方数据卡':'查看合成示例'} ↗</a></label>`).join('')}</div><label class="field"><span>导入规模</span><select id="catalog-count"><option value="50">50 篇 · 快速试跑</option><option value="200">200 篇 · 小型实验</option><option value="500">500 篇 · 扩大样本</option></select></label><details class="catalog-rules"><summary>导入规则、许可与网络要求</summary><p class="mini-note">按源顺序取样并精确去重，最多 20 MiB；合成示例最多 240 篇。从 ModelScope 预训练 JSONL 文件开头顺序读取完整行，由平台重新划分训练/验证集，不代表官方基准成绩。使用与分发须遵循原数据许可。</p><p class="mini-note">联网导入通常需要数秒，最多约一分钟；关闭页面不会撤销服务器导入。相同项目、来源和规模再次导入会复用已保存数据。</p></details>`,'导入并选用',async()=>{
      const key=$('[name="catalog-source"]:checked').value,documents=+$('#catalog-count').value;
      $('#dialog-confirm').textContent='正在获取完整文档…';
      try{const result=await api('/dataset-catalog/'+key+'/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({documents}),timeoutMs:90000});if(version!==session)return;datasets=await api('/datasets');paintDatasets();$('#dataset_id').value=result.id;datasetMeta();updateForm();$('#upload-state').textContent=`${result.reused?'已选用已有语料':'导入成功'} · ${result.rows} 篇 · ${size(result.bytes)}`;toast('语料已选中，可预览或配置训练');await refresh();}
      finally{$('#dialog-confirm').textContent='导入并选用';}
    });
  }catch(error){showError(error);}finally{if(version===session)updateForm();}
};

function paintTestControls(j){
  const version=$('#test-checkpoint'),models=j.testable_models||[],same=version.dataset.job===j.id,old=version.value,signature=JSON.stringify(models);
  if(!same||version.dataset.models!==signature){version.innerHTML=models.length?models.map(x=>`<option value="${esc(x)}">${x==='final'?'最终模型':esc(x)}</option>`).join(''):'<option value="">等待完整模型</option>';if(same&&models.includes(old))version.value=old;version.dataset.models=signature;version.dataset.job=j.id;}
  if(!same){testError='';delete $('#test-history').dataset.signature;$('#test-result-empty').hidden=false;delete $('#test-result').dataset.signature;selectedTest=null;testItems=[];pendingTest=null;$('#test-result').hidden=true;$('#test-history').innerHTML='<div class="empty">正在读取测试记录…</div>';}
  const device=$('#test-device'),devices=JSON.stringify(system?.gpus?.map(g=>g.index)||[]);
  if(device.dataset.cards!==devices){const before=device.value;device.innerHTML='<option value="cpu">CPU · 小模型测试</option>'+(system?.gpus||[]).map(g=>`<option value="cuda:${g.index}">GPU ${g.index} · ${esc(g.name)}</option>`).join('');device.value=[...device.options].some(x=>x.value===before)?before:'cpu';if(!device.dataset.cards&&system?.gpus?.length)device.value='cuda:'+system.gpus[0].index;device.dataset.cards=devices;}
  const ds=$('#test-dataset'),dsSignature=JSON.stringify([j.id,datasets.map(d=>[d.id,d.name])]);
  if(ds.dataset.signature!==dsSignature){const previous=ds.value;ds.innerHTML='<option value="">原训练任务的验证集</option>'+datasets.filter(d=>d.id!==j.config.dataset_id).map(d=>`<option value="${d.id}">${esc(d.name)}</option>`).join('');ds.value=[...ds.options].some(x=>x.value===previous)?previous:'';ds.dataset.signature=dsSignature;}
  $('#test-availability').textContent=models.length?`${UI.architectureNames[j.config.architecture||'gpt2']} · ${j.config.layers} 层 · 上下文 ${j.config.seq_length} tokens · ${models.length} 个模型版本`:'当前任务还没有可测试的完整模型。';
  $('#test-new-tokens').max=Math.min(256,j.config.seq_length-1);
  $('#test-context-hint').textContent=`当前上下文 ${j.config.seq_length} tokens，包含输入和生成内容。基础模型用于续写，不等同于聊天模型。`;
  $('#test-submit-state').textContent=testError||(!models.length?'等待训练停止并写出完整模型或检查点后再测试。':system?.control?.paused?'维护模式下暂不接收新测试。':!canWrite?'当前账号为只读。':'');
  $('#submit-test').disabled=!canWrite||!models.length||testSubmitting||!!system?.control?.paused;$('#submit-test').textContent=testSubmitting?'正在提交…':$('#test-mode').value==='generate'?'运行续写测试':'运行语料评估';
}
$('#test-mode').onchange=()=>{
  const score=$('#test-mode').value==='score';
  $('#generation-fields').hidden=$('#generation-advanced').hidden=score;$('#score-fields').hidden=!score;
  all('#generation-fields input,#generation-fields textarea,#generation-advanced input').forEach(el=>el.disabled=score);
  all('#score-fields input,#score-fields select').forEach(el=>el.disabled=!score);
  all('[name=test-mode-choice]').forEach(el=>el.checked=el.value===$('#test-mode').value);
  $('#test-prompt').required=!score;$('#test-top-p').disabled=score||+$('#test-temperature').value===0;
  $('#test-prompt').setCustomValidity('');testError='';if(detailData)paintTestControls(detailData);
};
all('[name=test-mode-choice]').forEach(el=>el.onchange=()=>{$('#test-mode').value=el.value;$('#test-mode').onchange();});
$('#test-temperature').oninput=()=>{$('#test-top-p').disabled=$('#test-mode').value==='score'||+$('#test-temperature').value===0;};
$('#test-top-p').disabled=true;
$('#model-test-form').addEventListener('invalid',event=>{const details=event.target.closest('details');if(details)details.open=true;},true);
$('#model-test-form').addEventListener('input',()=>{testError='';$('#test-prompt').setCustomValidity('');if(detailData)paintTestControls(detailData);});
$('#model-test-form').onsubmit=async e=>{
  e.preventDefault();if(!detailData||!canWrite||testSubmitting)return;
  if($('#test-mode').value==='generate'&&!$('#test-prompt').value.trim()){$('#test-prompt').setCustomValidity('请输入非空的续写内容');$('#test-prompt').reportValidity();return;}
  testError='';
  const jid=detailData.id,version=session,device=$('#test-device').value,score=$('#test-mode').value==='score',sampling=!score&&+$('#test-temperature').value>0;
  const c={mode:score?'score':'generate',checkpoint:$('#test-checkpoint').value,prompt:score?'':$('#test-prompt').value,max_new_tokens:score?32:+$('#test-new-tokens').value,temperature:score?0:+$('#test-temperature').value,top_p:sampling?+$('#test-top-p').value:0.9,seed:+$('#test-seed').value,device:device.startsWith('cuda:')?'cuda':'cpu',gpu_id:device.startsWith('cuda:')?+device.split(':')[1]:0,dataset_id:score?($('#test-dataset').value||null):null,max_blocks:score?+$('#test-blocks').value:32,max_runtime_seconds:+$('#test-timeout').value};
  const payload=JSON.stringify(c),signature=jid+payload;
  if(!pendingTest||pendingTest.signature!==signature)pendingTest={signature,key:[...crypto.getRandomValues(new Uint8Array(16))].map(x=>x.toString(16).padStart(2,'0')).join('')};
  testSubmitting=true;paintTestControls(detailData);
  try{const result=await api('/jobs/'+jid+'/tests',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':pendingTest.key},body:payload});if(version!==session||jid!==selectedJob)return;pendingTest=null;selectedTest=result.id;await loadTests();if(version!==session||jid!==selectedJob)return;$('#test-result').scrollIntoView({block:'nearest'});toast('测试已入队，结果会自动刷新');}
  catch(error){if(version===session&&jid===selectedJob&&!error.stale)testError=error.message;}finally{testSubmitting=false;if(detailData)paintTestControls(detailData);}
};
async function loadTests(){
  if(!connected||!selectedJob)return;const jid=selectedJob,revision=++testRevision;const items=await api('/jobs/'+jid+'/tests');if(jid!==selectedJob||revision!==testRevision)return;
  testItems=items;if(!items.some(x=>x.id===selectedTest))selectedTest=items[0]?.id||null;
  $('#test-history-count').textContent=`${items.length} 条`;const history=$('#test-history'),historySignature=JSON.stringify([jid,items,canWrite]);
  if(history.dataset.signature!==historySignature){history.dataset.signature=historySignature;
  history.innerHTML=items.length?items.map(t=>`<div class="test-record"><button data-test="${t.id}" class="${selectedTest===t.id?'selected':''}"><strong>${t.config.mode==='generate'?'文本续写':'语料评估'} · ${esc(t.config.checkpoint)} <span class="badge ${esc(t.status)}">${names[t.status]||esc(t.status)}</span></strong><small>${date(t.created_at)} · ${esc(t.config.device.toUpperCase())} · ${t.result?.loss!=null?'Loss '+t.result.loss.toFixed(4):t.config.mode==='generate'?esc(t.config.prompt.slice(0,50)):'等待评估结果'}</small></button>${canWrite&&['queued','running','cancelling'].includes(t.status)?`<button class="button compact" data-stop-test="${t.id}" ${t.status==='cancelling'?'disabled':''}>停止</button>`:''}</div>`).join(''):'<div class="empty">尚未运行模型测试</div>';
  all('[data-test]').forEach(b=>b.onclick=()=>{selectedTest=b.dataset.test;all('[data-test]').forEach(x=>x.classList.toggle('selected',x===b));paintTestResult();$('#test-result').scrollIntoView({block:'nearest'});$('#test-result').focus({preventScroll:true});});
  all('[data-stop-test]').forEach(b=>b.onclick=()=>{openDialog('停止模型测试？','<p>停止本次续写或评估，已训练好的模型保持不变。</p>','确认停止',async()=>{await api('/jobs/'+jid+'/tests/'+b.dataset.stopTest+'/cancel',{method:'POST'});if(jid===selectedJob)await loadTests();});});}paintTestResult();
}
function paintTestResult(){
  const t=testItems.find(x=>x.id===selectedTest),box=$('#test-result');box.hidden=!t;$('#test-result-empty').hidden=!!t;if(!t){delete box.dataset.signature;return;}const signature=JSON.stringify(t);if(box.dataset.signature===signature)return;box.dataset.signature=signature;const r=t.result;
  let body=r?(r.mode==='generate'?`<h4>输入提示词</h4><pre>${esc(r.prompt)}</pre><h4>模型续写</h4><pre class="result-text">${esc(r.completion||'模型输出了结束符或仅含特殊符号，没有可展示的文本。')}</pre><p class="mini-note">输入 ${r.prompt_tokens} tokens · 新增 ${r.generated_tokens} tokens · ${r.generation_seconds.toFixed(2)} 秒 · ${r.tokens_per_second?.toFixed(1)||'—'} tokens/s</p>`:`<div class="job-metrics"><div><small>评估 Loss</small><b>${r.loss.toFixed(4)}</b></div><div><small>Perplexity</small><b>${r.perplexity?.toFixed(2)||'超出范围'}</b></div><div><small>有效 tokens</small><b>${number(r.evaluated_tokens)}</b></div></div><p class="mini-note">${r.blocks} 个完整块 · 每块 ${r.sequence_length} tokens · ${r.source==='training_validation'?'原验证集':'另选评估语料'}</p><details><summary>复现与比较条件</summary><p>语料 SHA256：<code>${esc(r.dataset_sha256)}</code></p><p>分词器 SHA256：<code>${esc(r.tokenizer_sha256)}</code></p></details>`):`<p>${t.status==='queued'?'正在排队，前面的训练或测试结束后开始。':t.status==='running'?'正在加载模型并执行测试…':names[t.status]||esc(t.status)}</p>`;
  if(t.error){const last=t.error.trim().split('\n').at(-1);body+=`<p class="job-error">${esc(last)}</p><details class="test-error-details"><summary>查看错误详情</summary><pre>${esc(t.error)}</pre></details>`;}
  box.innerHTML=`<div class="test-result-meta"><div>${t.config.mode==='generate'?'文本续写':'语料评估'} · ${t.config.checkpoint==='final'?'最终模型':esc(t.config.checkpoint)}<small>${date(t.created_at)} · ${esc(t.config.device.toUpperCase())}</small></div><span class="badge ${esc(t.status)}">${t.status==='running'?'测试中':names[t.status]||esc(t.status)}</span></div>${body}${r?`<p class="mini-note">${esc(r.note)}</p>`:''}<div class="result-actions"><button id="reuse-test" class="text-button">复用参数再测试</button>${r?'<button id="export-test" class="text-button">下载结果 JSON</button>':''}</div>`;
  $('#reuse-test').onclick=()=>{const c=t.config;$('#test-mode').value=c.mode;$('#test-prompt').value=c.prompt;$('#test-new-tokens').value=c.max_new_tokens;$('#test-temperature').value=c.temperature;$('#test-top-p').value=c.top_p;$('#test-seed').value=c.seed;$('#test-timeout').value=c.max_runtime_seconds;$('#test-blocks').value=c.max_blocks;$('#test-dataset').value=c.dataset_id||'';const target=c.device==='cuda'?'cuda:'+c.gpu_id:'cpu';$('#test-device').value=[...$('#test-device').options].some(o=>o.value===target)?target:'cpu';$('#test-mode').onchange();$('#model-test-form').scrollIntoView({block:'start'});toast('已复用测试参数，请核对模型版本和运行设备');};
  if(r)$('#export-test').onclick=()=>downloadBlob(new Blob([JSON.stringify(t,null,2)],{type:'application/json'}),'model-test-'+t.id.slice(0,8)+'.json');
}
$('#refresh-tests').onclick=()=>loadTests().catch(showError);
