'use strict';
/* 会话/结果归后端所有。浏览器只持有当前选择、未提交草稿、附件选择和阅读设置。
   请求归属固定 cid；切换会话后迟到的响应不能改写新会话。写入 key 在网络失败后保留。 */
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const states = {preparing:'正在理解要求',submitting:'正在提交任务',executing:'执行中',replied:'已回复',finished:'已结束',failed:'提交失败',interrupted:'准备已中断',canceled:'已取消',queued:'排队中',running:'执行中',completed:'验收通过',waiting_user:'等待主管',blocked:'暂时受阻',partial:'本轮已结束',unverified:'尚未通过验收',unknown:'状态待核实',conflict:'状态待核实'};
const errors = {conversation_delete_confirmation_required:'请确认删除会话。',workspace_busy:'这个工作区还有任务正在准备或执行，请结束后再删除。',workspace_default_required:'默认工作区用于保留历史会话，不能删除。',workspace_delete_confirmation_required:'请先确认删除工作区。',model_stream_canceled:'任务已停止，已收到的思考片段保留。',model_stream_missing_finish:'未收到完整结束标记，原始片段已保留，本轮未继续执行。',model_stream_incomplete:'流式返回被截断或过滤，原稿保留，本轮未继续执行。',model_stream_content_invalid:'供应商返回的流式正文格式不符合协议，已停止本轮请求。',model_stream_after_finish:'供应商在结束标记后继续返回正文，已停止本轮请求。',context_compaction_canceled:'任务已停止，摘要原文保留，未继续派发请求。',context_summary_previous_failure:'相同上下文的摘要此前失败，原文保留，未自动重复调用。',context_summary_failed:'上下文摘要失败，原文已保留，未继续派发执行请求。',context_summary_insufficient:'摘要仍过长，原文已保留，未自动重试。',context_protected_input_too_large:'当前必需输入已占满模型窗口，历史摘要无法腾出足够空间。',conversation_history_resource_limit:'会话原文超过本机传输资源保护范围，原记录保留。',model_change_busy:'还有任务在运行，请结束后再切换模型。',model_discovery_failed:'读取供应商模型列表失败，请检查原供应商配置或稍后重试。',provider_not_configured:'供应商尚未配置密钥。',reasoning_effort_unsupported:'当前模型未声明支持这个思考档位，请刷新模型列表。',model_native_tools_unsupported:'当前连接未配置此模型的原生工具能力，无法保存选择；原模型保持不变。请先验证并配置对应模型的工具调用能力。',execution_native_tools_unsupported:'当前模型连接未通过工具调用能力检查，任务尚未开始执行。请核对供应商协议与模型能力配置。',execution_capability_unknown:'当前模型连接的工具调用能力尚未确认，任务尚未开始执行。请核对供应商协议与模型能力配置。',model_not_discovered:'请先刷新供应商模型列表，再选择模型。',task_model_changed:'这条旧任务使用的模型已改变，请恢复原模型设置或发送新任务。',task_workspace_changed:'当前工作区与这条旧任务的原约束不同。请发送一条新消息，按当前工作区重新执行；原记录保留。',contract_preparation_rejected:'Agent 未能形成覆盖本次要求的有效任务条件，尚未开始执行。可重新编辑并发送。',agent_unavailable:'Agent 当前未连接或未启用。会话和草稿会继续保留。',agent_disabled:'Agent 当前已停用，请先在控制中枢启用。',conversation_busy:'这个会话还有任务在执行，可先停止，或新建会话。',workspace_directory_missing:'找不到该本地目录，请填写已有目录的绝对路径。',workspace_name_exists:'已有同名工作区，请换一个名称。',attachment_text_required:'目前支持 UTF-8 文本附件，每个最多 12,000 字。图片、PDF 和二进制文件暂不能直接上传。',conversation_context_full:'当前会话内容已达到本轮输入上限，请新建会话继续。',execution_input_oversized:'本轮消息与资料过长，请减少附件或新建会话。',resume_effects_unconfirmed:'当前 Agent 未允许安全恢复此任务，尚未重新执行。',resume_via_host_required:'该任务需要通过原宿主入口恢复。',task_version_conflict:'任务状态已变化，请刷新后再操作。',artifact_changed_or_oversized:'文件已发生变化或超过 32 MB，不能作为原产物下载。',artifact_unavailable:'原产物已不存在或缺少可验证的文件回执。',input_preparation_failed:'要求准备失败，尚未取得执行回执。请检查 Agent 配置后重新发送。',preparation_interrupted:'服务在准备要求时重启，这条消息没有自动重放。',preparation_canceled:'已取消准备；迟到的回复不会提交执行。',input_queue_full:'待处理的输入较多，请稍后再试。',host_authorization_denied:'当前 Agent 权限不允许此操作。',session_token_required:'连接已更新，请重试刚才的操作。',task_not_found:'任务已不在运行账本中，保留此前归档记录。'};
// 恢复错误来自宿主执行回执；说明阻塞原因，不把继续任务误写成重新执行命令。
Object.assign(errors,{conversation_waiting_answer:'会话还有待回答任务，请先回答或取消任务再删除。',question_not_current:'这个问题已失效或已有新的问题，请查看最新任务状态。',question_canceled:'任务已取消，迟到的回答不会恢复执行。',question_answer_conflict:'这个问题已有另一份回答，不能用重复提交覆盖。',question_answer_required:'请先填写回答。',recovery_effect_unknown:'旧工具调用的执行结果尚无法核实，继续任务已暂停。原记录保留，需要先核对执行回执。',recovery_process_requires_poll:'旧命令尚无结束回执，请先核对其运行状态，再继续任务。',recovery_process_identity_unavailable:'无法核对旧命令的执行身份，继续任务已暂停；原记录保留。',recovery_effect_unconfirmed:'原文件或执行回执已变化，继续任务已暂停，请先核对变化。',recovery_budget_insufficient:'任务累计预算已用完，未继续执行；原记录保留。',recovery_required_material_missing:'恢复所需的原始资料不完整，请补齐资料后继续。',resume_permission_expansion:'当前权限超出了旧任务提交时的范围，请使用原权限继续或发送新任务。'});

let conversationDeleteTarget='',conversationDeleting=false;
const deletedConversations=new Set();
let token='', workspaces=[], conversations=[], current=null, activeWorkspaceId='', selection=0, online=false, agent={};
let modelCatalog={available:false,providers:[],selected:{}}, modelSaving=false, modelRevision=0, modelDraft=null, modelStatus=null;
let creatingConversation=false;
let sending=false, actionBusy=false, workspaceFormMode='create', attachmentTarget=null, currentPreview=null, toastTimer;
const writes=new Map();
let messagesDeferred=false,reasoningDrag=null,selectionFrame=0;
const usageDisplay=window.AngelosUsage.create({api,changed:()=>{renderUsage();renderMessages()}});
// 日志面板先建实例，再交给 activity 面板的回调使用；两边只共享展示函数，不共享任务事实。
const logDisplay=window.AngelosLog?.create({
  root:document.getElementById('log-panel'),
  api,
  esc:v=>String(v??'').replace(/[&<>"']/g,c=>({"&":'&amp;',"<":'&lt;',">":'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  selectionInside:node=>{
    const selection=document.getSelection();
    if(!selection||selection.isCollapsed||!node)return false;
    for(let i=0;i<selection.rangeCount;i++)if(selection.getRangeAt(i).intersectsNode(node))return true;
    return false;
  },
  onChange:()=>{}
});
const activityDisplay=window.AngelosActivity?.create({api,changed:()=>renderMessages(),layout:()=>resizeComposer(),log:logDisplay||null});
let usageScope='session',usageConversation='';
function renderUsage(){
  // 复用低频快照，不因打开浮层或切换统计范围增加请求；浮层不参与页面排版。
  if(usageConversation!==current?.id){usageConversation=current?.id;usageScope='session';$('usage-details').open=false;}
  const ctx=lastTurn()?.context_usage;
  const contextText=ctx?.window_tokens&&typeof ctx.ratio==='number'?'上下文 '+Math.round(ctx.ratio*100)+'%':'上下文未知';
  $('usage-context').textContent=contextText;
  $('usage-context').title=ctx?.window_tokens?'本轮窗口 '+ctx.window_tokens.toLocaleString('zh-CN')+' token，使用量为估算':'尚无本轮窗口占用数据；容量未知时无法计算90%阈值';
  $('usage-session-hit').textContent=usageDisplay.statusHTML();
  $('usage-session-hit').hidden=!$('usage-session-hit').textContent;
  if($('usage-instance-hit'))$('usage-instance-hit').textContent=usageDisplay.instanceStatusHTML?.()||'累计命中待采样';
  for(const button of document.querySelectorAll('[data-usage-scope]'))button.setAttribute('aria-pressed',String(button.dataset.usageScope===usageScope));
  if(!$('usage-popover').matches(':popover-open'))return;
  const panel=usageDisplay.panel(usageScope,lastTurn()?.id),selection=window.getSelection();
  for(const [id,html] of [['usage-summary',panel.summary],['usage-advanced',panel.advanced]]){
    const el=$(id);
    // 选中文本时暂缓替换，保留复制内容与阅读位置；下次渲染再补齐快照。
    if(selection&&!selection.isCollapsed&&(el.contains(selection.anchorNode)||el.contains(selection.focusNode)))continue;
    if(el.innerHTML!==html){const body=$('usage-popover').querySelector('.usage-popover-body'),top=body.scrollTop;el.innerHTML=html;body.scrollTop=top;}
  }
}
$('usage-popover').addEventListener('toggle',renderUsage);
for(const button of document.querySelectorAll('[data-usage-scope]'))button.addEventListener('click',()=>{usageScope=button.dataset.usageScope;$('usage-details').open=false;renderUsage()});
function readLocal(key,fallback){try{return JSON.parse(localStorage.getItem('angelos:'+key))??fallback}catch{return fallback}}
function saveLocal(key,value){try{localStorage.setItem('angelos:'+key,JSON.stringify(value))}catch{}}
let preferences=readLocal('preferences',{size:'standard',send:'enter',compact:false,details:false});
const workspace=()=>workspaces.find(w=>w.id===activeWorkspaceId)||workspaces[0];
const time=v=>new Date(v*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
const turnState=t=>t.view?.status||t.stage;
// 分层结果只描述分析交付；证据是否核验另列，避免把正文交付写成全部事实通过。
const acceptanceLayers=t=>t.result?.verification?.layers||t.view?.verification?.layers||{};
// partial是执行状态，不能单凭它断言用户目标只完成一部分。
// 仅将宿主列出的缺项ID关联到具体要求；不从回答措辞猜测完成度。
function turnRemaining(t){
  if(turnState(t)!=='partial')return [];
  const requirements=t.result?.contract?.requirements||[];
  return [...new Set((t.result?.remaining||[]).map(id=>requirements.find(r=>r.id===id)?.description).filter(s=>typeof s==='string'&&s.trim()).map(s=>s.trim()))];
}
function turnLabel(t){const state=turnState(t),layers=acceptanceLayers(t);if(state==='partial')return turnRemaining(t).length?'部分完成':'本轮已结束';if(layers.policy==='conversation_reply')return '已回复';if(layers.policy==='agent_reply_delivery')return state==='completed'?'答复已交付':(t.view?.reason==='reply_delivered'?'答复已交付 · 未通过整项验收':(states[state]||state));return layers.policy==='ordinary_readonly_analysis'&&state==='completed'?'分析已交付':(states[state]||state)}
function remainingHTML(t){const rows=turnRemaining(t);return rows.length?`<div class="decision-record" aria-label="尚未完成事项"><strong>尚未完成</strong><ul>${rows.map(text=>`<li>${esc(text)}</li>`).join('')}</ul></div>`:'';}


const active=t=>['preparing','submitting','executing'].includes(t.stage);
const lastTurn=()=>current?.turns.at(-1);
const messageError=code=>errors[code]||('操作未完成：'+code);
function notify(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4500)}
function showError(error){$('connection-error').replaceChildren();const span=document.createElement('span');span.textContent=error.message||String(error);$('connection-error').append(span);const button=document.createElement('button');button.type='button';button.textContent='重新连接';button.addEventListener('click',()=>refresh(true));$('connection-error').append(button);$('connection-error').hidden=false}
function clearError(){$('connection-error').hidden=true}
function resizeComposer(){
  const input=$('message-input'),scroll=$('conversation-scroll');
  const nearEnd=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<48,top=scroll.scrollTop,inputTop=input.scrollTop;
  // 先释放旧高度，才能在删字/换草稿时缩回；超过视口上限才在输入框内滚动。
  // 只调整尺寸，不替换编辑器，保留输入焦点、选区和正在阅读的会话位置。
  input.style.height='auto';
  input.style.height=input.scrollHeight+'px';
  const overflowing=input.scrollHeight>input.clientHeight+1;
  input.style.overflowY=overflowing?'auto':'hidden';
  if(overflowing)input.scrollTop=inputTop;
  scroll.scrollTop=nearEnd?scroll.scrollHeight:top;
}
async function api(path, data, timeoutMs=15000){
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),timeoutMs);
  try{
    const response=await fetch('/angelos/api'+path,{method:data===undefined?'GET':'POST',credentials:'same-origin',cache:'no-store',signal:controller.signal,
      headers:data===undefined?{}:{'Content-Type':'application/json','X-Angelos-Token':token},body:data===undefined?undefined:JSON.stringify(data)});
    const value=await response.json();
    // 保留机器错误码，文件定位仅对明确的不存在做补查，不能误吞权限或网络错误。
    if(!response.ok||value.ok===false){if(value.reason==='session_token_required')await bootstrap();const error=new Error(messageError(value.reason||'request_failed'));error.reason=value.reason||'request_failed';throw error}
    return value;
  }catch(error){if(error.name==='AbortError')throw new Error(path==='/native-picker'?'文件选择超时，请重新打开选择窗口。':'连接超时。消息草稿已保留，重试会使用同一个提交编号。');if(error instanceof TypeError)throw new Error('无法连接本机 Agent，正在保留当前页面与草稿。');throw error}
  finally{clearTimeout(timeout)}
}
// 原生窗口等待可长于普通请求；不重试弹窗，取消不视为错误。
async function chooseNativePath(kind){
  if(!agent.nativePicker)throw new Error('当前服务未启用系统文件选择器，请先手动输入路径。');
  try{return await api('/native-picker',{kind,workspaceId:activeWorkspaceId||''},160000)}
  catch(error){const messages={native_picker_busy:'已有选择窗口打开，请在桌面完成或取消。',native_picker_unavailable:'系统文件选择器不可用，请手动输入路径。',native_picker_workspace_required:'请先为工作区选择本地目录。',native_picker_timeout:'选择窗口已超时关闭，请重新选择。',native_picker_directory_missing:'工作区目录已不存在，请先更新目录。',native_picker_failed:'系统选择窗口未能完成，请手动输入路径。'};throw new Error(messages[error.message.replace(/^操作未完成：/,'')]||error.message)}
}
let workspaceFormRevision=0;
$('browse-workspace').addEventListener('click',async()=>{
  const revision=workspaceFormRevision,wid=activeWorkspaceId,old=$('workspace-path').value;
  $('browse-workspace').disabled=true;$('workspace-picker-status').textContent='请在系统窗口选择目录…';
  try{
    const result=await chooseNativePath('directory');
    // 迟到结果不能覆盖另一表单、另一个工作区或等待期间的手工输入。
    if(revision!==workspaceFormRevision||wid!==activeWorkspaceId||$('workspace-form').hidden||!$('workspace-dialog').open||$('workspace-path').value!==old)return;
    if(!result.cancelled)$('workspace-path').value=result.path;
    $('workspace-picker-status').textContent=result.cancelled?'已取消选择。':'目录已填入，点击保存后生效。';
  }catch(error){if(revision===workspaceFormRevision&&wid===activeWorkspaceId)$('workspace-picker-status').textContent=error.message}
  finally{$('browse-workspace').disabled=false}
});
function safely(fn){return (...args)=>{try{Promise.resolve(fn(...args)).catch(showError)}catch(error){showError(error)}}}
function drafts(c){return readLocal('draft:'+c.id,{text:c.draft||'',pending:[],request:null})}
function saveDraft(c){saveLocal('draft:'+c.id,{text:c.draft,pending:c.pending,request:c.request});}
function queueDraft(c){
  const cid=c.id,text=c.draft;
  const previous=writes.get(cid)||Promise.resolve();
  const pending=previous.catch(()=>{}).then(()=>api('/conversations/'+cid,{draft:text}));
  writes.set(cid,pending);pending.catch(showError);
}
function adopt(c, attachments){
  const local=current?.id===c.id?{text:current.draft,pending:current.pending,request:current.request}:drafts(c);
  current={...c,draft:local.text,pending:local.pending,request:local.request,attachments:attachments||current?.attachments||[]};
  activeWorkspaceId=c.workspaceId;upsert(c);
}
function upsert(c){if(deletedConversations.has(c.id))return;const i=conversations.findIndex(x=>x.id===c.id);const item={...c,waiting:c.turns?c.turns.filter(t=>t.question).map(t=>({...t.question,turnId:t.id})):c.waiting,activity:c.turns?conversationActivity(c.turns):c.activity,search:c.turns?.map(t=>t.text).join('\n')||c.search};if(i<0)conversations.push(item);else conversations[i]=item;}
function renderSidebar(){
  const w=workspace();if(!w)return;
  $('sidebar-workspace').textContent=w.name;$('sidebar-path').textContent=w.path||'未关联目录';
  const options=workspaces.map(row=>`<option value="${esc(row.id)}">${esc(row.name)}</option>`).join('');
  for(const id of ['workspace-select','header-workspace-select']){const el=$(id);if(el.dataset.options!==options){el.innerHTML=options;el.dataset.options=options}if(document.activeElement!==el)el.value=w.id;el.disabled=creatingConversation}
  if($('status-workspace')){$('status-workspace').textContent=w.name;$('status-workspace').title=w.path||'未关联目录';}
  $('sidebar-path').title=w.path||'未关联目录';$('new-target').textContent=w.name;
  $('new-conversation').title='在「'+w.name+'」中新建独立对话';

  renderRunningSessions();
  renderWaitingSessions();
  const all=conversations.filter(c=>c.workspaceId===activeWorkspaceId&&!c.activity&&!c.waiting?.length).sort((a,b)=>b.updated-a.updated);
  const query=$('conversation-search').value.trim().toLocaleLowerCase();const shown=all.filter(c=>(c.title+' '+c.search).toLocaleLowerCase().includes(query));
  $('conversation-count').textContent=all.length;
  const markup=shown.length?shown.map(c=>`<div class="conversation-row"><button class="conversation-item" data-conversation="${esc(c.id)}" aria-current="${c.id===current?.id}" title="${esc(c.title)} · 最后使用 ${esc(time(c.updated))}"><strong>${esc(c.title)}</strong>${conversationBadge(c)}${drafts(c).text?'<span class="conversation-draft">草稿</span>':''}<span class="conversation-time" role="tooltip">最后使用 ${esc(time(c.updated))}</span></button><button class="conversation-delete" data-delete-conversation="${esc(c.id)}" aria-label="删除会话 ${esc(c.title)}">删除</button></div>`).join(''):`<p class="list-empty">${query?'没有匹配的历史会话':conversations.some(c=>c.workspaceId===activeWorkspaceId&&(c.activity||c.waiting?.length))?'当前任务见上方运行或等待列表':'还没有会话，点击「新会话」开始。'}</p>`;
  if($('conversation-list').dataset.rendered!==markup){$('conversation-list').innerHTML=markup;$('conversation-list').dataset.rendered=markup}
}
function renderHeader(){
  if(!current)return;const w=workspace();
  $('header-title').textContent=current.title;$('header-workspace').textContent=w.name;$('rename-conversation').hidden=false;
  const ctx=lastTurn()?.context_usage;
  const remaining=typeof ctx?.remaining_tokens==='number'?' · 预计可用 '+ctx.remaining_tokens.toLocaleString('zh-CN')+' token':'';
  const usage=ctx?(ctx.window_tokens?' · 上下文约 '+Math.round(ctx.ratio*100)+'%'+remaining+(ctx.compacted?' · 已摘要':''):' · 上下文窗口未知'):'';
  $('conversation-title').textContent=current.title;$('conversation-caption').textContent=w.name+' · '+current.turns.length+' 轮对话'+usage;
  $('conversation-status').textContent=lastTurn()?turnLabel(lastTurn()):'未开始';
  $('composer-workspace').textContent=w.name;$('composer-scope').textContent=current.turns.length?'继续当前会话':'新会话 · 独立上下文';
  $('empty-workspace-name').textContent=w.name;$('empty-workspace-path').textContent=w.path||'未关联目录，可直接对话';
  renderModelSelectors();
}
function artifactURL(t,index){return '/angelos/api/conversations/'+current.id+'/artifacts/'+encodeURIComponent(t.id)+'/'+index}
function artifactsHTML(t){
  const artifacts=t.result?.artifacts||[],latest=new Map();
  artifacts.forEach((f,index)=>latest.set(f.path,{f,index}));
  return [...latest.values()].map(({f,index})=>`<div class="artifact-row"><span class="file-icon">FILE</span><button class="file-preview" data-preview="${esc(t.id)}" data-index="${index}"><strong>${esc((f.path||'产物').split(/[\\/]/).pop())}</strong><small>本轮最后修改 · ${esc(f.bytes??'未知')} 字节 · 打开时校验</small></button><a class="text-button" href="${artifactURL(t,index)}" download>下载 ↧</a></div>`).join('');
}
function toolHTML(t){
  const events=t.result?.evidence||t.view?.done||[];
  if(!events.length&&!t.view)return '';
  const details={执行状态:turnLabel(t),核验:acceptanceLayers(t).policy==='agent_reply_delivery'?'答复已交付，执行事实见工具回执；整项目标未核验':acceptanceLayers(t).policy==='conversation_reply'?'普通回复，不涉及行动验收':t.view?.verification?.verified?'硬条件已通过':'尚未通过',模型诊断:t.model_outputs||[],分层核验:acceptanceLayers(t),原因:t.view?.reason||'',剩余条件:t.result?.remaining||[],上下文:t.context_usage||{},预算:t.view?.budget||{}};
  return `<button class="tool-panel-link" data-show-tools="${esc(t.id)}">查看工具调用 · ${events.length} 条回执 ↗</button><details class="tool-detail" data-detail="${esc(t.id)}" ${preferences.details?'open':''}><summary>执行详情</summary><div class="tool-output">${esc(JSON.stringify(details,null,2))}</div></details>`;
}
function reasoningHTML(t){
  const rows=activityDisplay?.reasoning(t)||t.reasoning||[];if(!rows.length)return '';
  const streaming=rows.some(r=>r.streaming),interrupted=rows.some(r=>['interrupted','incomplete'].includes(r.status));
  const returned=rows.filter(r=>r.returned&&r.text);
  const content=returned.length?returned.map((r,i)=>`<p class="reasoning-source">模型请求 ${i+1} · ${esc(r.model)}${r.streaming?' · 接收中':''}</p><pre>${esc(r.text)}${r.omitted_chars?`\n[展示省略 ${r.omitted_chars} 字符]`:''}</pre>`).join(''):streaming?'<p>正在接收模型响应，尚无可展示的推理字段…</p>':'<p>本轮供应商未返回可展示的推理字段。可在模型设置中查看或调整支持的思考档位。</p>';
  return `<details class="reasoning-detail" data-detail="reasoning-${esc(t.id)}" ${streaming?'open':''}><summary>模型推理<small>${streaming?'接收中':interrupted?'接收已中断':returned.length?'已返回 · 点击查看':'未返回'}</small></summary><div class="reasoning-content"><p class="reasoning-source">由模型供应商返回，供查看；不代表执行成功或事实核验。</p>${content}</div></details>`;
}
function progressHTML(t){
  const p=activityDisplay?.progress(t)||t.live_progress||(active(t)?{
    phase:t.stage==='preparing'?'preparing':'working',text:t.stage==='preparing'?'正在接收要求并准备执行。':'任务正在执行，正在连接进度记录。',events:[],totalEvents:0}:null);
  if(!p)return '';
  const updates=p.updates?.rows||[],events=p.events||[];
  if(!active(t)&&!events.length&&!updates.length)return '';
  // 助手面向用户的进展与宿主状态、工具回执分别标识，不从日志伪造叙述。
  const entry=u=>`<li class="task-update" data-update="${esc(u.id)}"><span class="update-marker"></span>${typeof u.created==='number'&&Number.isFinite(u.created)?`<time class="update-time">${esc(new Date(u.created*1000).toLocaleTimeString('zh-CN',{hour12:false}))}</time>`:''}<div>${esc(u.text)}${u.omitted_chars?`<small>另有 ${u.omitted_chars} 字符，完整原文保留在执行详情。</small>`:''}</div></li>`;
  const older=updates.slice(0,-8),recent=updates.slice(-8);
  const earlier=older.length?`<details class="earlier-updates" data-detail="updates-${esc(t.id)}"><summary>更早的 ${older.length} 条进展</summary><ol>${older.map(entry).join('')}</ol></details>`:'';
  const comments=updates.length?`${earlier}<ol class="task-update-list">${recent.map(entry).join('')}</ol>`:'';
  const receiptRows=events.map(e=>`<li class="progress-event progress-${esc(e.state)}">${esc(e.text)}</li>`).join('');
  const receipts=events.length?`<details class="progress-receipts" data-detail="receipts-${esc(t.id)}"><summary>工具执行记录 · ${p.totalEvents||events.length}</summary><ol class="progress-events">${receiptRows}</ol></details>`:'';
  const message=online?p.text:'连接中断，保留最后一次进展，正在等待重新连接。';
  const status=active(t)?`<div class="progress-current" role="status"><span class="progress-dot"></span>${esc(message)}</div>`:'';
  return `<section class="task-updates ${active(t)?'is-running':'is-finished'}" aria-label="任务进展" data-detail="commentary-${esc(t.id)}"><div class="progress-heading"><strong>过程</strong><small>${active(t)?'执行中':'已结束'}${updates.length?' · '+updates.length+' 条进展':''}</small></div>${p.updates?.omitted?'<p class="update-omitted">较早记录已折叠，原文保留。</p>':''}${comments}${status}${receipts}</section>`;
}
function taskOverviewHTML(t){
  const o=(activityDisplay?.progress(t)||t.live_progress)?.overview;
  if(!o)return '';
  const verdicts={reply_delivered:'答复已交付 · 执行结果见回执',verified:'执行验收通过',analysis_delivered:'分析已交付',partial:'部分结果获接受',not_verified:'尚未通过整项验收',pending:'等待当前尝试的验收'};
  const checks={satisfied:'已核验',missing:'尚未满足',pending:'待核验'};
  // 工作卡仅解释宿主回执；保留原控制按钮，不从展示状态自动恢复或执行任务。
  const rows=(o.checks||[]).map(r=>`<li><span class="task-check-${esc(r.state)}">${esc(checks[r.state]||'待核验')}</span><span>${esc(r.description||r.id)}</span></li>`).join('');
  // 逐工具重验结果独立显示，不把模型答复或过期文件回执标为整项成功。
  const receiptRows=(o.receiptChecks||[]).map(r=>`<li><span>${esc(r.tool)} · ${esc(r.execution_status)}</span><small>${r.process?.receipt_authenticated?'进程状态 '+esc(r.process.status)+' · 退出码 '+esc(r.process.returncode??'未知')+'；':''}${r.executed===false?'未执行':r.receipt_valid===true?'回执重验有效':r.receipt_valid===false?'回执核验未通过':'回执未独立核验'}</small></li>`).join('');
  const files=(o.files||[]).map(f=>`<li><span>${esc(f.name)}</span><small>${f.state==='receipt_valid'?'当时回执有效':'文件结果待核实'}</small></li>`).join('');
  const uncertain=o.status==='unknown'||o.status==='conflict';
  const next=o.cancelRequested&&o.workerActive?'停止请求已提交，仍在等待操作退出；已发生的改动保留。':
    uncertain?'当前运行状态无法确认，请先查看错误与过程记录。':
    ['failed','blocked','waiting_user','partial','canceled'].includes(o.status)?'可补充要求或使用下方「继续任务」；恢复仍会检查权限和执行状态。':
    o.report==='not_recorded'&&o.verification==='verified'?'执行已有验收结果，但还没有归档答复正文。':'查看下方正文、工具详情和可用产物。';
  return `<details class="task-overview" data-detail="overview-${esc(t.id)}"><summary>任务工作卡<small>${esc(verdicts[o.verification]||'状态待核对')} · ${o.receiptCount} 条回执</small></summary><div class="task-overview-body"><p>${online?'当前':'上次读取的'}执行状态：${esc(o.verification==='reply_delivered'&&o.status==='completed'?'答复已交付':states[o.status]||o.status)} · 第 ${o.attempt} 次尝试</p><p>答复正文：${o.report==='available'?'已归档（不等于外部送达）':'尚无当前尝试的归档正文'}</p>${o.verification==='analysis_delivered'?`<p>${o.answerEvidence==='verified'?'分析证据已核验。':'分析中的观点或引用仍须结合正文限制判断。'}</p>`:''}${receiptRows?`<h4>实际工具回执</h4><ul class="task-files">${receiptRows}</ul><p>回执只说明所列调用；整项目标与正文观点未做独立验收。</p>`:''}${rows?`<h4>目标与核验</h4><ul class="task-checks">${rows}</ul>`:''}${o.omittedChecks?`<p>另有 ${o.omittedChecks} 项，请查看执行详情。</p>`:''}${files?`<h4>本次执行关联的文件记录</h4><ul class="task-files">${files}</ul><p class="task-note">来自本次工具回执；不是当前磁盘差异，也不包含对外部编辑的归因。</p>`:''}${o.omittedFiles?`<p>另有 ${o.omittedFiles} 条文件记录，请查看工具详情。</p>`:''}${o.reason&&o.reason!=='reply_delivered'?`<p class="decision-record">${esc(messageError(o.reason))}</p>`:''}<p class="task-next">${esc(next)}</p></div></details>`;
}
function selectedReasoning(){
  const selected=new Set(),selection=document.getSelection();
  $('messages').querySelectorAll('.reasoning-detail,.task-updates').forEach(detail=>{
    if(detail===reasoningDrag)selected.add(detail);
    if(!selection||selection.isCollapsed)return;
    const content=detail.querySelector('.reasoning-content')||detail;
    for(let i=0;i<selection.rangeCount;i++){
      if(content&&selection.getRangeAt(i).intersectsNode(content))selected.add(detail);
    }
  });
  return selected;
}
function messageNodeKey(node){
  if(node.nodeType!==1)return '#'+node.nodeType;
  return node.tagName+':'+(node.dataset.message||node.dataset.detail||node.className);
}
function patchMessages(parent,next,paused){
  // 对话节点按轮次/类型保留，禁止重建选区的祖先。先删过期兄弟节点，避免
  // 新进度插入时搬动正在选择的思考节点；其他状态仍可继续更新。
  const desired=[...next.childNodes];
  for(const old of [...parent.childNodes]){
    if(!desired.some(node=>messageNodeKey(node)===messageNodeKey(old)))old.remove();
  }
  let cursor=parent.firstChild;
  for(const node of desired){
    let found=cursor;
    while(found&&messageNodeKey(found)!==messageNodeKey(node))found=found.nextSibling;
    if(!found){found=node.cloneNode(true);parent.insertBefore(found,cursor)}
    else{
      if(found!==cursor)parent.insertBefore(found,cursor);
      if(paused.has(found)){messagesDeferred=true}
      else if(found.nodeType===1){
        // details 展开状态由用户持有，轮询只更新内容；已有子滚动容器保持原节点。
        for(const attr of [...found.attributes]){
          if(attr.name!=='open'&&!node.hasAttribute(attr.name))found.removeAttribute(attr.name);
        }
        for(const attr of [...node.attributes]){
          if(attr.name!=='open'&&found.getAttribute(attr.name)!==attr.value)found.setAttribute(attr.name,attr.value);
        }
        patchMessages(found,node,paused);
      }else if(found.nodeValue!==node.nodeValue)found.nodeValue=node.nodeValue;
    }
    cursor=found.nextSibling;
  }
  while(cursor){const nextSibling=cursor.nextSibling;cursor.remove();cursor=nextSibling}
}
// 回复正文里的路径链接：只把“当前工作区内存在的文本文件路径”渲染成按钮，其余保持纯文本。
// 判定从保守出发——绝对路径须以当前工作区根开头，正文若含 ~/、..、file:// 等写法一律不识别；
// 点击后交给 workspace.js 的 openPath 做最终换算与越界校验，本函数不新增任何读取接口。
function replyHTML(turn){
 const text=turn.reply||turn.result?.verified_answer?.text||turn.result?.execution_report?.text||'';
 if(!text)return '';
 const workspacePath=workspaces.find(w=>w.id===current?.workspaceId)?.path||'';
 // 允许的前置分隔符：行首、空白、常见中英文括号/引号；分隔符本身仍按纯文本输出。
 const lead='(^|[\\s(\\[\\u3000\u300a\u300c\u201c`\'\uff08])';
 // 路径候选：以 / 开头的绝对路径，或带扩展名的文件名/相对路径；字符集不含引号、尖括号、反引号，
 // 因此不可能把引号/尖括号带进属性值，从源头上避免了属性逃逸。
 const pattern=new RegExp(lead+'((?:\\/(?:[A-Za-z0-9_.\\-]+))+|(?:[A-Za-z0-9_.\\-]+\\/)*[A-Za-z0-9_\\-][A-Za-z0-9_.\\-]*)`?','g');
 // 纯数字/纯版本号的候选（如 v1.2）和域名式写法不是文件路径，不识别。
 const notFile=/^[vV]?\d+(\.\d+)+$/;
 const unsafe=/(^|\/)(\.\.|~)(\/|$)/;
 let out='',cursor=0,match;
 while((match=pattern.exec(text))){
  const raw=match[2],start=match.index+match[1].length;
  // 尾部的句点不是路径的一部分（“见 src/a.py.” 里的句号）。
  const candidatePath=raw.replace(/[.]+$/,'');
  const looksLikeFile=/\.[A-Za-z0-9]{1,8}$/.test(candidatePath);
  const insideWorkspace=workspacePath&&(candidatePath.startsWith(workspacePath+'/')||!candidatePath.startsWith('/'));
  if(!looksLikeFile||notFile.test(candidatePath)||unsafe.test(candidatePath)||!insideWorkspace){pattern.lastIndex=start+1;continue}
  out+=esc(text.slice(cursor,start));
  // 按钮只携带路径字符串，不携带动作或权限；点击行为由 workspace.js 的既有校验决定。
  out+=`<button class="path-link" type="button" data-open-path="${esc(candidatePath)}" title="在右侧文件栏打开 ${esc(candidatePath)}">${esc(candidatePath)}</button>`;
  cursor=start+raw.length;
 }
 if(!cursor)return `<p class="message-text">${esc(text)}</p>`;
 return `<p class="message-text">${out}${esc(text.slice(cursor))}</p>`;
}
function renderMessages(){
  if(!current)return;
  const html=current.turns.map(t=>{
    const state=turnState(t), result=t.result||{};
    const supplements=(t.supplements||[]).map((s,i)=>`<article class="message user" data-message="supplement-${esc(t.id)}-${i}"><div class="message-head"><span class="avatar">X</span><strong>主管 · 补充</strong><time>${esc(time(s.created))}</time></div><div class="message-body"><p class="message-text">${esc(s.text)}</p></div></article>`).join('');
    const report=t.reply||result.verified_answer?.text||result.execution_report?.text||'';
    const layers=acceptanceLayers(t),reason=t.read_error||t.error||(t.question&&t.question_error_id===t.question.id?t.question_error:'')||(t.question?'':(layers.analysis_delivery==='delivered'||['conversation_reply','agent_reply_delivery'].includes(layers.policy)?'':t.view?.reason));
    const progress=t.question?'':progressHTML(t)+taskOverviewHTML(t);
    const reply=t.question?'':report||(progress&&active(t)?'':t.stage==='preparing'?'正在理解这条要求…':states[state]||state);
    const retry=['failed','interrupted'].includes(t.stage)&&!t.task_id?`<div class="task-actions"><button data-retry="${esc(t.id)}">重新编辑并发送</button></div>`:'';
    const controls=t.question?questionHTML(t):t.task_id&&!active(t)&&['failed','blocked','waiting_user','partial','canceled','unknown'].includes(state)?`<div class="task-actions"><button data-control="resume" data-turn="${esc(t.id)}">继续任务</button><button data-control="supplement" data-turn="${esc(t.id)}">将输入补充到此任务</button></div>`:'';
    return `<article class="message user" data-message="user-${esc(t.id)}"><div class="message-head"><span class="avatar">X</span><strong>主管</strong><time>${esc(time(t.created))}</time></div><div class="message-body"><p class="message-text">${esc(t.text)}</p>${t.attachments.length?`<div class="message-attachments">${t.attachments.map(id=>`<span class="label">${esc(current.attachments.find(a=>a.id===id)?.name||'附件')}</span>`).join('')}</div>`:''}</div></article>${supplements}<article class="message agent" data-message="agent-${esc(t.id)}"><div class="message-head"><span class="avatar">A</span><strong class="blue">Angelos</strong><small>${esc(turnLabel(t))}</small></div><div class="message-body">${progress}${reasoningHTML(t)}${reply?'<div class="result-heading">结果</div>'+replyHTML(t):''}${remainingHTML(t)}${layers.analysis_delivery==='delivered'?`<p class="decision-record">${esc(layers.answer_evidence==='verified'?'证据核验已通过。':layers.warning||'部分观点或引用待核实。')}</p>`:''}${reason?`<p class="decision-record">${esc(messageError(reason))}</p>`:''}${toolHTML(t)}${artifactsHTML(t)}${questionHistoryHTML(t)}${controls}${retry}${usageDisplay.turnHTML(t.id)}</div></article>`;
  }).join('');
  const messages=$('messages'),sameConversation=messages.dataset.conversation===current.id;
  if(sameConversation&&messages.dataset.rendered===html&&!messagesDeferred)return;
  // 回答输入框被轮询替换会丢选区；同会话编辑期间延迟消息补丁，侧栏仍刷新。
  if(sameConversation&&messages.contains(document.activeElement)&&document.activeElement.matches('[data-question-answer]')){messagesDeferred=true;return;}
  const paused=sameConversation?selectedReasoning():new Set();
  const previous=new Map([...messages.querySelectorAll('.reasoning-content')].map(node=>[node,node.innerHTML]));
  const scroll=$('conversation-scroll'),top=scroll.scrollTop,nearEnd=scroll.scrollHeight-top-scroll.clientHeight<100;
  messagesDeferred=false;
  if(!sameConversation){messages.innerHTML=html;reasoningDrag=null}
  else{const next=document.createElement('template');next.innerHTML=html;patchMessages(messages,next.content,paused)}
  messages.dataset.rendered=html;messages.dataset.conversation=current.id;
  messages.querySelectorAll('.reasoning-detail[open]').forEach(detail=>{
    const content=detail.querySelector('.reasoning-content')||detail;
    if(!paused.has(detail)&&previous.get(content)!==content.innerHTML)content.scrollTop=content.scrollHeight;
  });
  scroll.scrollTop=nearEnd&&!paused.size?scroll.scrollHeight:top;
}
function scheduleSelectionRender(){
  if(!selectionFrame)selectionFrame=requestAnimationFrame(()=>{selectionFrame=0;renderMessages()});
}
// 选择期间仅冻结相应思考区域；取消选择后用已经收到的快照补齐，不增网络请求。
document.addEventListener('selectionchange',scheduleSelectionRender);
$('messages').addEventListener('pointerdown',e=>{
  reasoningDrag=e.target.closest('.reasoning-content')?.closest('.reasoning-detail')||null;
});
$('messages').addEventListener('click',e=>{const b=e.target.closest('[data-show-tools]');if(b)activityDisplay?.show(b.dataset.showTools)});
// 正文路径链接：只转发路径字符串，打开、越界校验和草稿保护全部由 workspace.js 负责。
$('messages').addEventListener('click',e=>{const b=e.target.closest('[data-open-path]');if(!b)return;e.preventDefault();window.angelosWorkspace?.openPath?.(b.dataset.openPath)});
function endReasoningDrag(){reasoningDrag=null;scheduleSelectionRender()}
document.addEventListener('pointerup',endReasoningDrag);
document.addEventListener('pointercancel',endReasoningDrag);
window.addEventListener('blur',endReasoningDrag);
$('messages').addEventListener('toggle',e=>{
  const detail=e.target;
  if(detail.matches('.reasoning-detail[open]')&&!selectedReasoning().has(detail)){
    const content=detail.querySelector('.reasoning-content')||detail;content.scrollTop=content.scrollHeight;
  }
},true);
function renderPending(){if(!current)return;$('pending-files').innerHTML=current.pending.map(a=>`<div class="pending-file"><span>${esc(a.name)}</span><button type="button" data-remove="${esc(a.id)}" aria-label="移除附件 ${esc(a.name)}">×</button></div>`).join('');updateComposer()}
function updateComposer(){const busy=current?.turns.some(active);$('send-message').disabled=!online||!current||sending||modelSaving||!!modelDraftForCurrent()||!$('message-input').value.trim()||busy;$('send-message').textContent=sending?'提交中…':'发送';$('message-input').readOnly=sending;$('send-message').hidden=!!busy;$('stop-response').hidden=!busy;$('stop-response').disabled=actionBusy;$('composer-hint').textContent=preferences.send==='enter'?'Enter 发送 / Shift + Enter 换行':'Ctrl / ⌘ + Enter 发送'}
function renderConversation(reset=false){if(!current)return;activityDisplay?.select(current.id,current.turns);
  // 切换会话时关闭并清空日志面板：日志必须属于原任务，不能把上一会话的行留在新会话里。
  if(logDisplay&&logDisplay.current()&&!(current.turns||[]).some(t=>t.id===logDisplay.current().turnId))logDisplay.hide();
  const empty=!current.turns.length;$('stage').classList.toggle('is-empty',empty);$('empty-intro').hidden=!empty;$('empty-workspace').hidden=!empty;$('chat-document').hidden=empty;usageDisplay.select(current.id,agent.usageStats);renderUsage();renderSidebar();renderHeader();renderMessages();if(reset){$('message-input').value=current.draft;resizeComposer()}renderPending();if($('context-dialog').open)renderContext();saveLocal('selection',{workspace:activeWorkspaceId,conversation:current.id});saveLocal('last:'+activeWorkspaceId,current.id);window.angelosWorkspace?.refresh()}
async function bootstrap(){const revision=modelRevision;const data=await api('/bootstrap');token=data.token;if(modelSaving||revision!==modelRevision)return;workspaces=data.workspaces;conversations=data.conversations.filter(c=>!deletedConversations.has(c.id));pruneDeletedDrafts();agent=data.agent;if(window.angelosWorkspace)await window.angelosWorkspace.permissions();const cid=current?.id,version=selection;const catalog=(await api('/models'+(cid?'?conversationId='+encodeURIComponent(cid):''))).models;if(version===selection&&!modelSaving&&revision===modelRevision){modelCatalog=catalog;if(current&&catalog.conversationId===current.id)current.modelSettings=catalog.selected;renderModelSelectors()}taskNotices.observe(conversations);renderTaskNotices();online=true;}
async function selectConversation(id){const version=++selection,revision=modelRevision;modelDraft=null;modelStatus=null;$('settings-dialog').close();const data=await api('/conversations/'+id);if(version!==selection||deletedConversations.has(id))return;const catalog=(await api('/models?conversationId='+encodeURIComponent(id))).models;if(version!==selection||deletedConversations.has(id))return;if(modelSaving||revision!==modelRevision){const saved=conversations.find(c=>c.id===id)?.modelSettings;if(saved)catalog.selected=saved}modelCatalog=catalog;adopt({...data.conversation,modelSettings:catalog.selected},data.attachments);renderModelSelectors(false,true);$('messages').dataset.rendered='';renderConversation(true);closeSidebar();await loadWorkspaceInstructions(current.workspaceId);}
// 新建只操作会话，不中断后台任务；空白页连续点击复用当前会话，草稿/附件不清空。
async function newConversation(workspaceId){
  if(sending||creatingConversation)return;
  const wid=typeof workspaceId==='string'?workspaceId:activeWorkspaceId||workspaces[0]?.id;
  if(!workspaces.some(w=>w.id===wid))return;
  if(current?.workspaceId===wid&&!current.turns.length&&!current.draft.trim()&&!current.pending.length&&!current.request){closeSidebar();$('message-input').focus();return}
  const intent=++selection;creatingConversation=true;
  $('new-conversation').disabled=true;$('mobile-new').disabled=true;renderSidebar();
  try{
    if(current)saveDraft(current);
    const c=(await api('/conversations',{workspaceId:wid})).conversation;upsert(c);
    if(intent!==selection)return; // 迟到创建回执不抢走用户随后选中的会话。
    await selectConversation(c.id);$('message-input').focus();
  }finally{creatingConversation=false;$('new-conversation').disabled=false;$('mobile-new').disabled=false;renderSidebar()}
}
async function selectWorkspace(id){
  if(creatingConversation||!workspaces.some(w=>w.id===id))return;
  $('conversation-search').value='';
  const previous=readLocal('last:'+id,null);
  const c=conversations.find(c=>c.id===previous&&c.workspaceId===id)||conversations.filter(c=>c.workspaceId===id).sort((a,b)=>b.updated-a.updated)[0];
  if(c)await selectConversation(c.id);else await newConversation(id);
  $('workspace-dialog').close();
}
let refreshing=false;
async function refresh(explicit=false){if(refreshing||creatingConversation||conversationDeleting||modelSaving)return;refreshing=true;const wasOffline=!online;try{await bootstrap();if(current&&!conversations.some(c=>c.id===current.id)){localStorage.removeItem('angelos:draft:'+current.id);current=null;selection++}if(current){const id=current.id,version=selection,revision=modelRevision;const data=await api('/conversations/'+id);if(version===selection&&current?.id===id&&!deletedConversations.has(id)&&!modelSaving&&revision===modelRevision){adopt(data.conversation,data.attachments);renderConversation()}}else{const chosen=readLocal('selection',{});activeWorkspaceId=workspaces.some(w=>w.id===chosen.workspace)?chosen.workspace:workspaces[0].id;const c=conversations.find(c=>c.id===chosen.conversation&&c.workspaceId===activeWorkspaceId);if(c)await selectConversation(c.id);else await selectWorkspace(activeWorkspaceId)}if(explicit||wasOffline)clearError();if(explicit)notify('已重新连接')}catch(e){online=false;showError(e)}finally{refreshing=false;updateComposer();renderMessages()}}
function renderWorkspaces(){$('workspace-options').innerHTML=workspaces.filter(w=>(w.name+' '+w.path).toLocaleLowerCase().includes($('workspace-search').value.trim().toLocaleLowerCase())).map(w=>`<div class="workspace-row"><button class="workspace-option" data-select-workspace="${esc(w.id)}" aria-pressed="${w.id===activeWorkspaceId}"><span class="workspace-symbol">↳</span><span class="option-copy"><strong>${esc(w.name)}</strong><small class="mono">${esc(w.path||'未关联目录')}</small><small>${conversations.filter(c=>c.workspaceId===w.id).length} 个会话</small></span><span class="option-check">${w.id===activeWorkspaceId?'当前工作区':'进入 ↗'}</span></button><button class="workspace-new" data-new-workspace-conversation="${esc(w.id)}" aria-label="在 ${esc(w.name)} 中新建对话">新对话</button>${w.id==='default'?'<small class="workspace-protected">默认保留</small>':`<button class="workspace-delete" data-delete-workspace="${esc(w.id)}" aria-label="删除工作区 ${esc(w.name)}">删除</button>`}</div>`).join('')}
let workspaceDeleteTarget='',workspaceDeleting=false;
function openWorkspaceDelete(id){
  const target=workspaces.find(w=>w.id===id);if(!target||id==='default')return;
  workspaceDeleteTarget=id;$('workspace-delete-name').textContent=target.name;
  $('workspace-delete-note').textContent=`${conversations.filter(c=>c.workspaceId===id).length} 个会话及其草稿将移到「${workspaces.find(w=>w.id==='default')?.name||'默认工作区'}」。`;
  $('workspace-delete-error').hidden=true;$('workspace-delete-dialog').showModal();$('cancel-workspace-delete').focus();
}
async function deleteWorkspace(){
  if(workspaceDeleting||!workspaceDeleteTarget)return;
  const id=workspaceDeleteTarget;workspaceDeleting=true;$('confirm-workspace-delete').disabled=true;$('cancel-workspace-delete').disabled=true;
  try{
    await api('/workspaces/'+encodeURIComponent(id)+'/delete',{confirm:true});await bootstrap();
    // 只迁移本机归属，沿用会话ID及本地草稿；其他工作区的运行任务和编辑器草稿不动。
    if(activeWorkspaceId===id){activeWorkspaceId='default';if(current)await selectConversation(current.id);else await selectWorkspace('default')}
    else renderConversation();
    localStorage.removeItem('angelos:last:'+id);renderWorkspaces();$('workspace-form').hidden=true;
    $('workspace-delete-dialog').close();notify('工作区已删除，历史会话已保留，磁盘文件未删除。');
  }catch(error){$('workspace-delete-error').textContent=error.message;$('workspace-delete-error').hidden=false}
  finally{workspaceDeleting=false;$('confirm-workspace-delete').disabled=false;$('cancel-workspace-delete').disabled=false}
}
function openWorkspace(){if(!workspaces.length)return;closeSidebar();$('workspace-search').value='';renderWorkspaces();$('workspace-form').hidden=true;$('workspace-error').hidden=true;$('workspace-dialog').showModal()}
function showWorkspaceForm(mode){workspaceFormRevision++;$('workspace-picker-status').textContent='';workspaceFormMode=mode;$('workspace-form').reset();$('workspace-form').hidden=false;$('workspace-error').hidden=true;$('workspace-form-title').textContent=mode==='edit'?'编辑当前工作区':'新工作区';$('save-workspace').textContent=mode==='edit'?'保存工作区':'创建并进入';if(mode==='edit'){$('workspace-name').value=workspace().name;$('workspace-path').value=workspace().path;$('workspace-description').value=workspace().description}$('workspace-name').focus()}
function renderContext(){if(!current)return;$('context-workspace-name').textContent=workspace().name;$('context-path').textContent=workspace().path||'未关联目录';$('context-description').textContent=workspace().description||'暂无说明';$('context-conversation-name').textContent=current.title;$('context-resources').innerHTML=current.attachments.map(a=>`<div class="resource-row"><strong>${esc(a.name)}</strong><small>本机文本附件</small></div>`).join('')||'<p class="empty-copy">尚未添加资料。支持 UTF-8 文本附件。</p>';$('context-artifacts').innerHTML=current.turns.map(artifactsHTML).join('')||'<p class="empty-copy">当前还没有文件产物。</p>';$('context-records').innerHTML=current.turns.map(t=>`<div class="record-row"><time>${esc(time(t.created))}</time><span>${esc(turnLabel(t))}<br>${esc(t.model||'')}</span></div>`).join('')||'<p class="empty-copy">发送消息后，这里会显示执行记录。</p>';$('task-control').hidden=!current.turns.some(active)}
function openRename(){if(!current)return;$('context-dialog').close();$('rename-input').value=current.title;$('rename-dialog').showModal()}
function applyPreferences(){document.body.classList.toggle('large-text',preferences.size==='large');document.body.classList.toggle('compact',preferences.compact);resizeComposer()}
function openSettings(){renderModelSelectors(true);$('reading-size').value=preferences.size;$('send-mode').value=preferences.send;$('compact-setting').checked=preferences.compact;$('details-setting').checked=preferences.details;$('settings-dialog').showModal()}
function closeSidebar(){$('sidebar').classList.remove('open');$('drawer-scrim').classList.remove('open');$('sidebar-toggle').setAttribute('aria-expanded','false');syncSidebarAccess()}
function syncSidebarAccess(){const mobile=matchMedia('(max-width:760px)').matches,open=mobile&&$('sidebar').classList.contains('open');$('sidebar').inert=mobile&&!open;$('stage').inert=open;document.querySelector('.topbar').inert=open;const footer=document.querySelector('.workbench-statusbar');if(footer)footer.inert=open}
$('conversation-search').addEventListener('input',renderSidebar);
$('conversation-list').addEventListener('click',safely(async e=>{const remove=e.target.closest('[data-delete-conversation]');if(remove){openConversationDelete(remove.dataset.deleteConversation);return}const b=e.target.closest('[data-conversation]');if(b)await selectConversation(b.dataset.conversation)}));
$('new-conversation').addEventListener('click',safely(newConversation));$('mobile-new').addEventListener('click',safely(newConversation));
$('sidebar-toggle').addEventListener('click',()=>{if($('sidebar').classList.contains('open'))return closeSidebar();$('sidebar').classList.add('open');$('drawer-scrim').classList.add('open');$('sidebar-toggle').setAttribute('aria-expanded','true');syncSidebarAccess()});$('drawer-scrim').addEventListener('click',closeSidebar);
addEventListener('resize',()=>{syncSidebarAccess();resizeComposer()});addEventListener('keydown',e=>{if(e.key==='Escape')closeSidebar()});
document.querySelectorAll('[data-workspace]').forEach(b=>b.addEventListener('click',openWorkspace));$('context-workspace').addEventListener('click',()=>{$('context-dialog').close();openWorkspace()});
$('workspace-options').addEventListener('click',safely(async e=>{const start=e.target.closest('[data-new-workspace-conversation]');if(start){await newConversation(start.dataset.newWorkspaceConversation);$('workspace-dialog').close();return}const remove=e.target.closest('[data-delete-workspace]');if(remove)return openWorkspaceDelete(remove.dataset.deleteWorkspace);const b=e.target.closest('[data-select-workspace]');if(b)await selectWorkspace(b.dataset.selectWorkspace)}));
$('cancel-workspace-delete').addEventListener('click',()=>$('workspace-delete-dialog').close());
$('workspace-delete-dialog').addEventListener('cancel',e=>{if(workspaceDeleting)e.preventDefault()});
$('confirm-workspace-delete').addEventListener('click',deleteWorkspace);
$('show-create-workspace').addEventListener('click',()=>showWorkspaceForm('create'));$('edit-workspace').addEventListener('click',()=>showWorkspaceForm('edit'));['hide-workspace-form','cancel-workspace-form'].forEach(id=>$(id).addEventListener('click',()=>{$('workspace-form').hidden=true}));
$('workspace-form').addEventListener('submit',safely(async e=>{e.preventDefault();$('save-workspace').disabled=true;try{const data={name:$('workspace-name').value.trim(),path:$('workspace-path').value.trim(),description:$('workspace-description').value.trim()};const result=await api('/workspaces'+(workspaceFormMode==='edit'?'/'+activeWorkspaceId:''),data);await bootstrap();await selectWorkspace(result.workspace.id);notify('工作区已保存')}catch(error){$('workspace-error').textContent=error.message;$('workspace-error').hidden=false}finally{$('save-workspace').disabled=false}}));
$('context-button').addEventListener('click',()=>{renderContext();$('context-dialog').showModal()});$('settings-button').addEventListener('click',openSettings);$('rename-conversation').addEventListener('click',openRename);$('context-rename').addEventListener('click',openRename);
$('rename-form').addEventListener('submit',safely(async e=>{e.preventDefault();const id=current.id,title=$('rename-input').value.trim();await api('/conversations/'+id,{title});if(current.id===id){current.title=title;upsert(current);renderHeader();renderSidebar()}$('rename-dialog').close()}));
$('settings-form').addEventListener('submit',e=>{e.preventDefault();preferences={size:$('reading-size').value,send:$('send-mode').value,compact:$('compact-setting').checked,details:$('details-setting').checked};saveLocal('preferences',preferences);applyPreferences();$('messages').querySelectorAll('.tool-detail').forEach(d=>{d.open=preferences.details});$('messages').dataset.rendered='';renderMessages();updateComposer();$('settings-dialog').close();notify('阅读设置已保存')});
$('message-input').addEventListener('input',()=>{if(!current)return;current.draft=$('message-input').value;saveDraft(current);resizeComposer();updateComposer();renderSidebar()});
$('message-input').addEventListener('blur',()=>{if(current)queueDraft(current)});
$('message-input').addEventListener('keydown',e=>{if(e.isComposing||e.key!=='Enter'||e.shiftKey)return;if(preferences.send==='enter'||e.ctrlKey||e.metaKey){e.preventDefault();if(!$('send-message').disabled)$('composer').requestSubmit()}});
$('composer').addEventListener('submit',safely(async e=>{
  e.preventDefault();if(!current||sending||modelSaving||modelDraftForCurrent()||current.turns.some(active))return;
  const c=current,text=c.draft.trim();if(!text)return;
  const payload={text,attachments:c.pending.map(a=>a.id)};const signature=JSON.stringify(payload);
  if(c.request?.signature!==signature)c.request={signature,key:crypto.randomUUID()};
  saveDraft(c);sending=true;clearError();updateComposer();
  try{
    // 先等本会话已排队的草稿写入，避免迟到写入在发送成功后恢复旧草稿。
    await (writes.get(c.id)||Promise.resolve()).catch(()=>{});
    const result=await api('/conversations/'+c.id+'/messages',{...payload,request_key:c.request.key});
    c.draft='';c.pending=[];c.request=null;saveDraft(c);upsert(result.conversation);
    if(current.id===c.id){adopt(result.conversation,c.attachments);current.draft='';current.pending=[];current.request=null;saveDraft(current);renderConversation(true);$('conversation-scroll').scrollTop=$('conversation-scroll').scrollHeight;$('message-input').focus()}
  }finally{sending=false;updateComposer()}
}));
async function controlTurn(turn, action){
  if(actionBusy||!turn)return;
  const cid=current.id,body={attempt:turn.view?.attempt,version:turn.view?.version,request_key:crypto.randomUUID()};
  if(action==='supplement'){body.text=current.draft.trim();if(!body.text){$('context-dialog').close();$('message-input').focus();return notify('先在输入框写下补充要求，再点击这项操作。')}if(current.pending.length)throw new Error('任务补充暂不接受新附件，请将资料文字写在输入框中。')}
  if(['answer','defer','undefer'].includes(action)){
    body.question_id=turn.question?.id;
    if(action==='answer'){body.text=(turn.question?.answer_received?turn.question.answer_text:questionDraft(turn.question?.id)).trim();if(!body.text)return notify('请先填写回答，再点击回答并继续。');}
  }
  actionBusy=true;updateComposer();
  try{const result=await api('/conversations/'+cid+'/turns/'+encodeURIComponent(turn.id)+'/'+action,body);if(action==='answer')saveLocal('question:'+body.question_id,'');upsert(result.conversation);if(current.id===cid){if(action==='supplement'){current.draft='';saveDraft(current)}adopt(result.conversation,current.attachments);renderConversation(action==='supplement')}notify(action==='answer'?'回答已交给任务，正在继续':action==='defer'?'问题已暂存，任务不会自动执行':action==='undefer'?'已展开待回答问题':action==='cancel'?'停止请求已提交，正在核实执行状态':action==='supplement'?'补充已交给 Agent；继续执行仍需点击「继续任务」':'继续请求已提交')}
  finally{actionBusy=false;updateComposer()}
}
$('stop-response').addEventListener('click',safely(()=>controlTurn(current.turns.findLast(active),'cancel')));
$('task-control').addEventListener('click',()=>{$('context-dialog').close();$('cancel-dialog').showModal()});
$('cancel-form').addEventListener('submit',safely(async e=>{e.preventDefault();await controlTurn(current.turns.findLast(active),'cancel');$('cancel-dialog').close()}));
$('messages').addEventListener('click',safely(async e=>{const retry=e.target.closest('[data-retry]');if(retry){const turn=current.turns.find(t=>t.id===retry.dataset.retry);current.draft=turn.text;current.pending=current.attachments.filter(a=>turn.attachments.includes(a.id));current.request=null;saveDraft(current);renderConversation(true);$('message-input').focus();return}const b=e.target.closest('[data-control]');if(b)await controlTurn(current.turns.find(t=>t.id===b.dataset.turn),b.dataset.control);await fileClick(e)}));
function chooseFiles(){if(!current)return;attachmentTarget=current.id;$('file-picker').click()}
$('attach-file').addEventListener('click',chooseFiles);$('context-attach').addEventListener('click',()=>{$('context-dialog').close();chooseFiles()});
$('file-picker').addEventListener('change',safely(async e=>{
  const cid=attachmentTarget,files=[...e.target.files];e.target.value='';if(!cid)return;
  if(files.length>8)throw new Error('每轮最多选择 8 个文本附件。');
  for(const file of files){if(file.size>48000)throw new Error(messageError('attachment_text_required'));const bytes=new Uint8Array(await file.arrayBuffer());const data=btoa(Array.from(bytes,b=>String.fromCharCode(b)).join(''));const result=await api('/conversations/'+cid+'/attachments',{name:file.name,data});
    if(current?.id===cid){current.pending.push(result.attachment);current.attachments.push(result.attachment);saveDraft(current);renderPending()}else{const draft=drafts({id:cid});draft.pending.push(result.attachment);saveLocal('draft:'+cid,draft)}
  }notify('文本资料已保存，可随下一条消息发送。')
}));
$('pending-files').addEventListener('click',e=>{const b=e.target.closest('[data-remove]');if(!b)return;current.pending=current.pending.filter(a=>a.id!==b.dataset.remove);saveDraft(current);renderPending()});
async function fileClick(e){const b=e.target.closest('[data-preview]');if(!b)return;const t=current.turns.find(t=>t.id===b.dataset.preview);const index=Number(b.dataset.index);const url=artifactURL(t,index);const response=await fetch(url,{cache:'no-store'});if(!response.ok){const error=await response.json();throw new Error(messageError(error.reason))}const blob=await response.blob();if(blob.size>300000)throw new Error('这个文件较大，请下载后查看。');let text;try{text=new TextDecoder('utf-8',{fatal:true}).decode(await blob.arrayBuffer());if(text.includes('\0'))throw new Error()}catch{throw new Error('这是二进制文件，请下载后用对应软件打开。')}const f=t.result.artifacts[index];currentPreview={url,name:(f.path||'产物').split(/[\\/]/).pop()};$('preview-title').textContent=currentPreview.name;$('preview-note').textContent='已校验文件内容与任务回执一致';$('preview-content').textContent=text;$('preview-dialog').showModal()}
$('context-artifacts').addEventListener('click',safely(fileClick));$('download-preview').addEventListener('click',()=>{if(!currentPreview)return;const a=document.createElement('a');a.href=currentPreview.url;a.download=currentPreview.name;document.body.append(a);a.click();a.remove()});
document.querySelectorAll('[data-close]').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
// 字体和可用宽度变化会改变自动换行；观察宽度，避免高度变化触发自身循环。
let composerWidth=0;
new ResizeObserver(([entry])=>{if(entry.contentRect.width!==composerWidth){composerWidth=entry.contentRect.width;resizeComposer()}}).observe($('composer'));
document.fonts.ready.then(resizeComposer);
applyPreferences();syncSidebarAccess();refresh();setInterval(()=>refresh(),2500);

// 模型目录属于后端；浏览器只持有展示副本。显示能力未知时不虚构可用档位。
const effortLabels={auto:'供应商默认',off:'关闭思考',minimal:'极低',low:'低',medium:'中',high:'高',xhigh:'更高',max:'最高'};
function modelProvider(id){return modelCatalog.providers.find(p=>p.id===id)}
function modelEfforts(providerId,modelId){const row=modelProvider(providerId)?.models.find(m=>m.id===modelId);return ['auto',...(row?.thinkingControl?['off',...(row.effortLevels||[])]:[])]}
function updateModelSelect(id,html,value,force=false){const el=$(id);if(!force&&document.activeElement===el)return;if(el.dataset.options!==html){el.innerHTML=html;el.dataset.options=html}el.value=value}
function fillEfforts(id,providerId,modelId,selected,force=false){const levels=modelEfforts(providerId,modelId);updateModelSelect(id,levels.map(v=>`<option value="${esc(v)}">${esc(effortLabels[v]||v)}</option>`).join(''),levels.includes(selected)?selected:'auto',force)}
// 未保存的供应商选择仅属于当前会话；轮询不能把用户正在选择的内容覆盖。
function modelDraftForCurrent(){return modelDraft?.cid===current?.id?modelDraft:null}
function modelSelectionForCurrent(){return modelDraftForCurrent()||current?.modelSettings||modelCatalog.selected||{}}
function renderModelStatus(){const status=modelStatus?.cid===current?.id?modelStatus:null,el=$('session-model-status');el.hidden=!status;el.textContent=status?.text||'';el.classList.toggle('error',!!status?.error)}
function changeModelProvider(){
 if(!current||modelSaving||sending||current.turns.some(active))return;
 const providerId=$('provider-select').value;
 if(providerId===(current.modelSettings||modelCatalog.selected).providerId){modelDraft=null;modelStatus=null}
 else{modelDraft={cid:current.id,providerId,modelId:'',effort:'auto'};modelStatus={cid:current.id,text:'请选择该供应商的模型，选定后保存；当前尚未切换。'}}
 renderModelSelectors(false,true);updateComposer();
}
function renderModelSelectors(settings=false,force=false){
  const selected=modelSelectionForCurrent(),p=modelProvider(selected.providerId),busy=!current||current?.turns.some(active)||modelSaving||sending;
  const providerOptions=modelCatalog.providers.map(v=>`<option value="${esc(v.id)}" ${!v.configured?'disabled':''}>${esc(v.label)}</option>`).join('');updateModelSelect('provider-select',providerOptions,selected.providerId||'',force);$('provider-select').disabled=!modelCatalog.available||!!busy;
  const modelOptions=(modelDraftForCurrent()&&!selected.modelId?'<option value="" disabled>选择模型…</option>':'')+(p?.models?.length?p.models:[{id:agent.model||'未连接 Agent'}]).map(m=>`<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('');updateModelSelect('model-select',modelOptions,selected.modelId??agent.model,force);
  $('model-select').disabled=!modelCatalog.available||!!busy;fillEfforts('effort-select',selected.providerId,selected.modelId,selected.effort,force);$('effort-select').disabled=!modelCatalog.available||!!busy||!selected.modelId;
  if(settings){
    $('settings-provider').innerHTML=modelCatalog.providers.map(v=>`<option value="${esc(v.id)}" ${!v.configured?'disabled':''}>${esc(v.label)}${!v.configured?' · 未配置密钥':''}</option>`).join('');$('settings-provider').value=selected.providerId||'';renderSettingsModels(selected.modelId,selected.effort);
  }
  renderModelStatus();$('apply-model').disabled=!modelCatalog.available||!!busy;$('refresh-models').disabled=!modelCatalog.available||modelSaving;
}
function renderSettingsModels(modelId,effort){const pid=$('settings-provider').value,p=modelProvider(pid);$('settings-model').innerHTML=(p?.models||[]).map(m=>`<option value="${esc(m.id)}">${esc(m.name||m.id)}</option>`).join('');if(modelId&&p?.models.some(m=>m.id===modelId))$('settings-model').value=modelId;fillEfforts('settings-effort',pid,$('settings-model').value,effort);$('model-capability-note').textContent=(modelEfforts(pid,$('settings-model').value).length>1?'档位来自供应商实际模型元数据。':'尚无可调档位信息，可刷新模型列表。')+' 未返回推理内容时不会补造。'}
async function applyModelSelection(data){
 const cid=current?.id;if(!cid||modelSaving||sending||current.turns.some(active)||!data.modelId)return;
 // 保存开始和结束均推进代际，拒绝任何跨越保存边界的旧轮询响应。
 modelSaving=true;modelRevision++;modelDraft={...data,cid};modelStatus={cid,text:'正在保存 '+data.modelId+'…'};
 renderModelSelectors(false,true);updateComposer();
 try{const result=await api('/conversations/'+encodeURIComponent(cid)+'/model',data);
  const c=conversations.find(c=>c.id===cid);if(c)c.modelSettings=result.models.selected;
  if(current?.id===cid){current.modelSettings=result.models.selected;modelCatalog=result.models;modelStatus={cid,text:'已应用：'+result.models.selected.modelId};$('model-setting-status').textContent=modelStatus.text}
  notify('会话模型已保存，其他会话不受影响');return true;
 }catch(error){
  // 模型选择失败在原入口显示，不能冒充整个工作台连接中断。
  if(current?.id===cid){modelStatus={cid,text:data.modelId+'：'+error.message,error:true};$('model-setting-status').textContent=modelStatus.text}
  return false;
 }finally{modelSaving=false;modelRevision++;if(modelDraft?.cid===cid)modelDraft=null;renderModelSelectors(false,true);updateComposer()}
}
$('settings-provider').addEventListener('change',()=>renderSettingsModels());$('settings-model').addEventListener('change',()=>renderSettingsModels($('settings-model').value));
// 刷新只更新目录，保留当前未提交的模型/思考选择；失败在设置框内可见。
$('refresh-models').addEventListener('click',safely(async()=>{const pid=$('settings-provider').value,mid=$('settings-model').value,effort=$('settings-effort').value;$('refresh-models').disabled=true;$('model-setting-status').textContent='正在读取供应商模型列表…';try{modelCatalog=(await api('/models/refresh',{providerId:pid})).models;renderSettingsModels(mid,effort);$('model-setting-status').textContent='模型列表已更新'}catch(error){$('model-setting-status').textContent=error.message;throw error}finally{$('refresh-models').disabled=false}}));
$('apply-model').addEventListener('click',safely(()=>applyModelSelection({providerId:$('settings-provider').value,modelId:$('settings-model').value,effort:$('settings-effort').value})));
$('model-select').addEventListener('change',safely(()=>applyModelSelection({providerId:modelSelectionForCurrent().providerId,modelId:$('model-select').value,effort:'auto'})));
$('effort-select').addEventListener('change',safely(()=>applyModelSelection({providerId:modelSelectionForCurrent().providerId,modelId:modelSelectionForCurrent().modelId,effort:$('effort-select').value})));

// 常用选择直接呈现；创建/编辑/删除保留在明确的管理入口。
for(const id of ['workspace-select','header-workspace-select'])$(id).addEventListener('change',safely(async e=>{await selectWorkspace(e.target.value);renderSidebar()}));
$('manage-workspaces').addEventListener('click',openWorkspace);
$('quick-create-workspace').addEventListener('click',()=>{openWorkspace();showWorkspaceForm('create')});
$('workspace-search').addEventListener('input',renderWorkspaces);

// 删除与自动刷新互斥；迟到响应不抢走后来选择的会话，也不恢复已删草稿。

function openConversationDelete(id){if(conversationDeleting)return;const c=conversations.find(c=>c.id===id);if(!c)return;conversationDeleteTarget=id;$('conversation-delete-name').textContent=c.title;$('conversation-delete-error').hidden=true;$('conversation-delete-dialog').showModal()}
$('cancel-conversation-delete').onclick=()=>{if(!conversationDeleting)$('conversation-delete-dialog').close()};
$('confirm-conversation-delete').onclick=safely(async()=>{
 if(conversationDeleting||sending||creatingConversation)return;
 const id=conversationDeleteTarget;if(!id)return;conversationDeleting=true;$('confirm-conversation-delete').disabled=true;
 try{
  // 等待已经排队的草稿保存，后端拒绝任何迟到写入已删除记录。
  await Promise.allSettled([...writes.values()]);
  await api('/conversations/'+id+'/delete',{confirm:true});deletedConversations.add(id);
  conversations=conversations.filter(c=>c.id!==id);localStorage.removeItem('angelos:draft:'+id);
  for(const w of workspaces)if(readLocal('last:'+w.id,null)===id)localStorage.removeItem('angelos:last:'+w.id);
  $('conversation-delete-dialog').close();
  if(current?.id===id){current=null;selection++;localStorage.removeItem('angelos:selection');await selectWorkspace(activeWorkspaceId)}
  renderSidebar();notify('会话已删除；工作区、文件和累计用量保留');
 }catch(e){$('conversation-delete-error').textContent=e.message;$('conversation-delete-error').hidden=false}
 finally{conversationDeleting=false;$('confirm-conversation-delete').disabled=false}
});
async function loadWorkspaceInstructions(id){
 const version=selection,node=$('workspace-instructions');node.textContent='正在读取 AGENTS.md…';
 try{const {instructions:r}=await api('/workspaces/'+id+'/instructions');if(version!==selection||current?.workspaceId!==id)return;
 node.textContent=({unbound:'未关联目录',absent:'未找到 AGENTS.md',loaded:'已读取 '+r.files.length+' 份 AGENTS.md',partial:'AGENTS.md 部分读取失败',unavailable:'工作区目录不可用'})[r.status];
 node.title=[...r.files.map(f=>f.path),...r.issues.map(x=>typeof x==='string'?x:x.path+'：'+x.reason)].join('\n');
 }catch(e){if(version===selection){node.textContent='AGENTS.md 读取失败';node.title=e.message}}
}

function pruneDeletedDrafts(){
 // 刷新完整列表后清掉已从其他页面或维护入口删除的会话草稿。
 const ids=new Set(conversations.map(c=>c.id));
 for(const key of Object.keys(localStorage))if(key.startsWith('angelos:draft:')&&!ids.has(key.slice('angelos:draft:'.length)))localStorage.removeItem(key);
}

// 任务状态取后端阶段及回执；不依据回复里的“失败/完成”等词猜测。
function conversationActivity(turns){const t=[...turns].reverse().find(active);return t?{stage:t.stage,status:t.view?.status||t.stage}:null}
function conversationBadge(c){
 const a=c.activity;if(!a)return '';
 const label=a.stage==='preparing'?'准备中':a.stage==='submitting'?'提交中':({queued:'排队中',planning:'规划中',yielded:'执行中',waiting_background:'等待工具',waiting_user:'等待主管',blocked:'受阻',completed:'同步结果',failed:'同步结果',canceled:'同步结果'})[a.status]||'执行中';
 return `<span class="conversation-running" aria-label="任务${esc(label)}"><span aria-hidden="true">●</span> ${esc(label)}</span>`;
}

// 会话级入口靠近输入框；供应商切换仍使用已发现的模型与原权限验证。
$('session-model-settings').addEventListener('click',openSettings);
$('provider-select').addEventListener('change',changeModelProvider);
const taskNotices=new AngelosTaskNotices(readLocal('task-notices',null),value=>saveLocal('task-notices',value));
function renderRunningSessions(){
 const rows=conversations.filter(c=>c.activity).sort((a,b)=>b.updated-a.updated);
 $('running-count').textContent=rows.length;
 const html=rows.map(c=>`<button class="conversation-item" data-conversation="${esc(c.id)}" aria-current="${c.id===current?.id}"><strong>${esc(c.title)}</strong>${conversationBadge(c)}<small>${esc(workspaces.find(w=>w.id===c.workspaceId)?.name||'工作区')}</small></button>`).join('')||'<p class="list-empty">暂无运行中的任务</p>';
 if($('running-list').dataset.rendered!==html){$('running-list').innerHTML=html;$('running-list').dataset.rendered=html}
}
$('running-list').addEventListener('click',safely(async e=>{const b=e.target.closest('[data-conversation]');if(b)await selectConversation(b.dataset.conversation)}));
// 旧提醒的 key 已包含宿主状态；只读取它做配色，不按标签文字推断成功或失败。
function noticeStatus(n){try{return JSON.parse(n.key)[1]||''}catch{return ''}}
function renderTaskNotices(){
 const rows=taskNotices.unread();$('task-notices-count').textContent=rows.length;
 $('task-notices-button').classList.toggle('has-unread',!!rows.length);
 const html=rows.map(n=>`<button class="task-notice" data-state="${esc(noticeStatus(n))}" data-notice="${esc(n.key)}" data-conversation="${esc(n.cid)}"><strong>${esc(n.label)} · ${esc(n.title)}</strong><small>${esc(time(n.at))} · 点击查看</small></button>`).join('')||'<p class="list-empty">没有未读任务提醒</p>';
 $('task-notices-list').innerHTML=html;
 const fresh=taskNotices.takeFresh();if(fresh.length){const last=fresh.at(-1);notify(last.label+'：'+last.title);$('task-live-notice').hidden=false;$('task-live-notice').dataset.state=noticeStatus(last);$('task-live-notice').innerHTML=`<button data-conversation="${esc(last.cid)}">${esc(last.label)} · ${esc(last.title)}${fresh.length>1?' 等 '+fresh.length+' 项':''} ↗</button><button id="dismiss-task-live" aria-label="收起任务提示">×</button>`;}
}
$('task-notices-button').addEventListener('click',()=>{renderTaskNotices();$('task-notices-dialog').showModal()});
$('task-notices-list').addEventListener('click',safely(async e=>{const b=e.target.closest('[data-notice]');if(!b)return;await selectConversation(b.dataset.conversation);taskNotices.read(b.dataset.notice);renderTaskNotices();$('task-notices-dialog').close();$('task-live-notice').hidden=true}));
$('task-live-notice').addEventListener('click',safely(async e=>{const b=e.target.closest('[data-conversation]');if(b)await selectConversation(b.dataset.conversation);$('task-live-notice').hidden=true}));
$('clear-task-notices').addEventListener('click',()=>{taskNotices.readAll();renderTaskNotices();$('task-live-notice').hidden=true});

// 问题内容来自宿主waiting_user结果，草稿只留当前浏览器；回答按钮绑定问题ID。
function questionDraft(id){return readLocal('question:'+id,'');}
function questionHTML(t){
 const q=t.question;
 if(q.deferred)return `<section class="question-card"><strong>等待主管 · 已暂存</strong><p class="message-text">${esc(q.text)}</p><button data-control="undefer" data-turn="${esc(t.id)}">现在回答</button><button data-control="cancel" data-turn="${esc(t.id)}">取消任务</button></section>`;
 return `<section class="question-card" aria-label="等待主管回答"><strong>等待主管</strong><p class="message-text">${esc(q.text)}</p><p class="question-note">${q.answer_received?'回答已保存，但恢复尚未获准。核对提示原因后可重试继续。':'已暂停执行，等待不消耗模型请求。回答后会核对当前状态再继续。'}</p><textarea rows="2" maxlength="12000" ${q.answer_received?'readonly':''} data-question-answer="${esc(q.id)}" aria-label="回答 Angelos 的问题" placeholder="输入你的决定，或说明已完成的操作…">${esc(q.answer_received?q.answer_text:questionDraft(q.id))}</textarea><div class="task-actions"><button class="primary" data-control="answer" data-turn="${esc(t.id)}">${q.answer_received?'重试继续':'回答并继续'}</button><button data-control="defer" data-turn="${esc(t.id)}">稍后回答</button><button data-control="cancel" data-turn="${esc(t.id)}">取消任务</button></div></section>`;
}
$('messages').addEventListener('input',e=>{const input=e.target.closest('[data-question-answer]');if(input)saveLocal('question:'+input.dataset.questionAnswer,input.value);});
function renderWaitingSessions(){
 const rows=conversations.filter(c=>c.waiting?.length).sort((a,b)=>b.updated-a.updated);
 $('waiting-count').textContent=rows.length;$('waiting-sessions').hidden=!rows.length;
 const html=rows.map(c=>`<button class="conversation-item" data-conversation="${esc(c.id)}" aria-current="${c.id===current?.id}"><strong>${esc(c.title)}</strong><span class="waiting-badge">${c.waiting.every(q=>q.deferred)?'已暂存':'待回答'}</span><small>${esc(workspaces.find(w=>w.id===c.workspaceId)?.name||'工作区')}</small></button>`).join('');
 if($('waiting-list').dataset.rendered!==html){$('waiting-list').innerHTML=html;$('waiting-list').dataset.rendered=html;}
}
$('waiting-list').addEventListener('click',safely(async e=>{const b=e.target.closest('[data-conversation]');if(b)await selectConversation(b.dataset.conversation);}));

function questionHistoryHTML(t){
 const rows=(t.questions||[]).filter(q=>q.id!==t.question?.id);
 return rows.length?`<details class="question-history" data-detail="questions-${esc(t.id)}"><summary>此前提问 · ${rows.length}</summary>${rows.map(q=>`<p class="message-text">${esc(q.text)}</p>`).join('')}</details>`:'';
}

// 快捷操作只转交既有入口；不持有任务状态，不隐式发送、取消或批准任务。
// 列表来自当前已加载会话，搜索不会增加网络轮询；打开其它对话仍由原作用域逻辑处理。
(()=>{
 const dialog=$('command-dialog'),input=$('command-search'),list=$('command-results');
 if(!dialog||!input||!list)return; // 兼容热更新期间仍在加载旧 HTML 的页面。
 let visible=[],cursor=0;
 function entries(){return [
  {label:'新建对话',hint:'当前工作区',run:()=>newConversation()},
  {label:'打开文件',hint:'文件路径或系统选择器',run:()=>{document.getElementById('view-code').click();$('open-code-path').focus()}},
  {label:'查看文件改动',hint:'当前工作区',run:()=>document.getElementById('view-code').click()},
  {label:'查看工具调用',hint:'当前会话',run:()=>{if($('tools-panel').hidden)$('tools-button').click()}},
  {label:'切换工作区',hint:'工作区管理',run:()=>$('manage-workspaces').click()},
  {label:'设置与阅读密度',hint:'字号、发送方式、模型',run:()=>$('settings-button').click()},
  ...conversations.map(c=>({label:c.title||'新会话',hint:workspaces.find(w=>w.id===c.workspaceId)?.name||'会话',run:()=>selectConversation(c.id)}))
 ];}
 function render(){const q=input.value.trim().toLocaleLowerCase();visible=entries().filter(c=>(c.label+' '+c.hint).toLocaleLowerCase().includes(q));cursor=0;
  list.innerHTML=visible.map((c,i)=>`<button type="button" data-command-index="${i}"><span>${esc(c.label)}</span><small>${esc(c.hint)}</small></button>`).join('')||'<p class="list-empty">没有匹配的操作或会话</p>';highlight();}
 function highlight(){list.querySelectorAll('button').forEach((b,i)=>b.classList.toggle('command-current',i===cursor));}
 async function run(index){const entry=visible[index];if(!entry)return;dialog.close();await entry.run();}
 function open(){if(document.querySelector('dialog[open]'))return;input.value='';render();dialog.showModal();input.focus();}
 // 顶栏换行后使用实际高度，避免窄屏面板和输入区互相遮挡。
 const topbar=document.querySelector('.topbar');
 if(topbar&&typeof ResizeObserver==='function')new ResizeObserver(()=>{
  document.body.style.setProperty('--top',Math.ceil(topbar.getBoundingClientRect().height)+'px');
 }).observe(topbar);
 $('command-button')?.addEventListener('click',open);input.addEventListener('input',render);
 list.addEventListener('click',safely(e=>{const b=e.target.closest('[data-command-index]');if(b)return run(Number(b.dataset.commandIndex))}));
 input.addEventListener('keydown',safely(e=>{if(e.isComposing)return;if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();cursor=(cursor+(e.key==='ArrowDown'?1:-1)+visible.length)%Math.max(1,visible.length);highlight();list.querySelector('.command-current')?.scrollIntoView({block:'nearest'});}else if(e.key==='Enter'){e.preventDefault();return run(cursor)}}));
 document.addEventListener('keydown',e=>{if(!e.isComposing&&(e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();open()}},true);
 // 面板切换只隐藏视图，编辑器、选区与未保存草稿仍由 workspace.js 持有。
 document.querySelectorAll('[data-detail-tab]').forEach(b=>b.addEventListener('click',()=>{
  if(b.dataset.detailTab==='files')$('view-code').click();else if($('tools-panel').hidden)$('tools-button').click();
 }));
 $('usage-instance-hit')?.addEventListener('click',()=>{usageScope='instance';$('usage-popover').showPopover();renderUsage()});
})();
