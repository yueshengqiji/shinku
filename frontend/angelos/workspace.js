'use strict';
/* 权限/审批来自宿主；编辑草稿绑定工作区和路径，轮询只更新事实，不覆盖草稿。 */
(()=>{
 const ui=id=>document.getElementById(id);
 let permissions={available:false}, permissionBusy=false, refreshing=false, panel=false;
 let opened=null, editor=null, dirty=false, saving=false, request=0, selectedWorkspace='', discardAction=null;
 let codeSelection=null, attaching=false;
 const labels={added:'新增',modified:'修改',deleted:'删除'};
 Object.assign(errors,{permission_controls_unavailable:'当前宿主尚未加载权限控制。',permission_version_conflict:'权限已被其他页面修改，已刷新，请重新选择。',agent_permission_changed:'权限已切换，旧任务授权已失效。',approval_expired:'这项批准已失效或已经处理。',approval_denied:'该工具执行被拒绝。',approval_timed_out:'等待批准超时，工具没有执行。',file_edit_not_permitted:'当前权限只允许查看；切换到 WAW 或 ALEPH 后可编辑。',file_version_conflict:'磁盘文件已被其他操作修改。草稿已保留，请重新读取并确认差异。',file_binary_or_oversized:'文件为二进制、非 UTF-8 或超过 512 KiB，请用本地编辑器打开。',workspace_scan_limit:'工作区超过 2,000 个候选文件，未返回不完整的改动统计。',workspace_snapshot_limit:'普通目录快照超过 16 MiB，请选择更小的工作区或使用 Git。',file_path_outside_workspace:'只能打开当前工作区内的普通文件。',file_path_unavailable:'路径不可访问，或包含符号链接。',file_hardlink_not_editable:'该文件有硬链接，不能从网页替换。',file_not_found:'找不到该文件。'});
 const status=value=>{ui('code-status').textContent=value};
 function renderPermissions(){
  const selector=ui('permission-select');
  if(!permissions.available){selector.innerHTML='<option>宿主未连接</option>';selector.disabled=true;return}
  const markup=permissions.levels.map(l=>`<option value="${l.id}">${l.label} · ${l.description}</option>`).join('');
  if(selector.dataset.markup!==markup){selector.innerHTML=markup;selector.dataset.markup=markup}
  selector.value=permissions.level;selector.className='permission-'+permissions.level;selector.disabled=permissionBusy;
  const rows=permissions.pending||[];
  const html=rows.map(r=>`<section class="tool-approval"><strong>等待批准 · ${esc(r.tool)}</strong><small>${r.originHost?'来源：'+esc(r.originHost):(r.conversationId===current?.id?'当前会话':'其他会话')} · 仅本次调用</small><pre>${esc(JSON.stringify(r.arguments,null,2))}</pre><div><button data-approval="${esc(r.id)}" data-verdict="deny">拒绝执行</button><button class="primary" data-approval="${esc(r.id)}" data-verdict="approve">批准这次执行</button></div></section>`).join('');
  if(ui('tool-approvals').dataset.html!==html){ui('tool-approvals').innerHTML=html;ui('tool-approvals').dataset.html=html}
  if(opened){const writable=opened.exists&&['waw','aleph'].includes(permissions.level);ui('code-mode').disabled=!writable;if(!writable){ui('code-mode').value='read';editor?.setEditable(false)}updateSave()}
 }
 async function loadPermissions(){if(permissionBusy)return;if(agent.workbenchControls!==true){permissions={available:false};renderPermissions();return}permissions=(await api('/permissions')).permissions;renderPermissions()}
 ui('permission-select').addEventListener('change',safely(async()=>{
  const level=ui('permission-select').value;permissionBusy=true;renderPermissions();
  try{permissions=(await api('/permissions',{level,revision:permissions.revision})).permissions;notify('权限已切换为 '+level.toUpperCase()+'；旧任务会在下一执行检查点停止，已发生的改动保留。')}
  finally{permissionBusy=false;await loadPermissions()}
 }));
 ui('tool-approvals').addEventListener('click',safely(async e=>{
  const b=e.target.closest('[data-approval]');if(!b)return;const row=permissions.pending.find(r=>r.id===b.dataset.approval);if(!row)return;
  b.disabled=true;try{permissions=(await api('/approvals/'+row.id,{revision:row.revision,decision:b.dataset.verdict})).permissions;renderPermissions()}finally{b.disabled=false}
 }));
 function updateSave(){ui('code-save').disabled=saving||!dirty||!opened?.exists||!['waw','aleph'].includes(permissions.level)||ui('code-mode').value!=='edit';ui('code-save').textContent=saving?'保存中…':dirty?'保存 *':'保存'}
 function makeEditor(content){
  codeSelection=null;renderSelection();
  editor?.destroy();ui('code-editor').replaceChildren();
  editor=AngelosEditor.create(ui('code-editor'),{path:opened.path,original:opened.original,content,diff:ui('code-diff').checked,onChange(value){dirty=value!==opened.content;updateSave();status(dirty?'有未保存修改；自动刷新不会覆盖草稿。':'内容与读取时一致。')},onSelection(value){codeSelection=value;renderSelection()},onSave:()=>save().catch(e=>status(e.message))});
  editor.setEditable(ui('code-mode').value==='edit');
 }
 function afterDiscard(action){if(!dirty){action();return}discardAction=action;ui('discard-code-dialog').showModal()}
 const pathWithin=(candidate,root)=>{const r=String(root||'').replace(/\/+$/,'');return !!r&&(candidate===r||candidate.startsWith(r+'/'))};
 // 路径不属于当前工作区时，只在前端比较已注册工作区的目录前缀，找到最深的归属工作区；
 // 这里不扩大任何读取权限：后端仍按目标工作区的 no-follow 规则逐层核验。
 // 只读探测：用现有文件接口问「这个相对路径在该工作区里存不存在」。存在为 200，不存在为 409。
 // 不新建接口、不放宽后端核验；网络异常一律当作“未命中”，不影响原有报错。
 async function pathExistsInWorkspace(wid,path){
  try{const value=await api('/workspaces/'+encodeURIComponent(wid)+'/file?path='+encodeURIComponent(path));return !!value&&!!value.file}
  catch(error){return false}
 }
 // 相对路径（如从正文裁剪出的 `shinku-standalone/AGENTS.md`）不以任何工作区根开头，前缀归属接不住。
 // 做法：当输入首段等于某个工作区根目录的末段（说明它是从根里截出来、把根名重复了一次），
 // 就去掉这个首段，把剩下的当作该根下的相对路径，交给只读探测确认。
 // 不碰 `..`、绝对路径、反斜杠与 URL；有歧义或探测不到一律返回 null，保留原有报错。
 async function completeRelativePath(value,wid){
  const raw=String(value||'').trim().replace(/^(\.\/)+/,'');
  if(!raw||raw.startsWith('/')||raw.includes('..')||raw.includes('\\')||raw.includes('://'))return null;
  const segs=raw.split('/').filter(p=>p&&p!=='.');if(segs.length<2)return null;
  const cand=segs.slice(1).join('/');if(!cand)return null;
  const hits=[];
  for(const w of workspaces){
   const base=String(w.path||'').split('/').filter(Boolean);if(!base.length)continue;
   if(base[base.length-1]!==segs[0])continue;
   if(await pathExistsInWorkspace(w.id,cand))hits.push({id:w.id,name:w.name,root:w.path,rest:cand});
  }
  if(!hits.length)return null;
  // 多个工作区可能指向同一个目录（重复注册）；同一根只算一个候选，不当作歧义。
  const byRoot=[];for(const h of hits){if(!byRoot.some(x=>x.root===h.root))byRoot.push(h)}
  if(byRoot.length===1)return byRoot[0];
  const others=byRoot.filter(h=>h.id!==wid);
  return others.length===1?others[0]:null;
 }
 function resolveWorkspaceForPath(value,wid){
  const raw=String(value||'').trim();if(!raw||raw.includes('..'))return null;
  if(!raw.startsWith('/'))return null;
  const candidates=workspaces.filter(w=>pathWithin(raw,w.path)).sort((a,b)=>b.path.length-a.path.length);
  if(!candidates.length)return null;
  const target=candidates[0];if(target.id===wid)return null;
  return {id:target.id,name:target.name,root:target.path,rest:stripWorkspaceRoot(raw,target.path)};
 }
 // 两种归认合一：绝对路径前缀归属优先，其次是相对路径探测补全。
 async function locatePath(value,wid){
  const byPrefix=resolveWorkspaceForPath(value,wid);
  if(byPrefix)return byPrefix;
  return completeRelativePath(value,wid);
 }
 // 把目标工作区根目录前缀去掉，得到该工作区内的相对路径。
 function stripWorkspaceRoot(value,root){const raw=String(value||'').trim(),r=String(root||'').replace(/\/+$/,'');if(!r||!pathWithin(raw,r))return null;const rest=raw.slice(r.length).replace(/^\/+$/,'').replace(/^\/+/,'');return rest||null}
 // 打开失败时面板主体原本是空白，只有 #code-status 一行小字；这里把原因写进面板主体，让用户看得见。
 function showEditorNotice(message){
  const root=ui('code-editor');
  let box=root.querySelector('.path-open-notice');
  if(!box){box=document.createElement('div');box.className='empty-copy path-open-notice';box.setAttribute('role','status');root.replaceChildren(box)}
  box.textContent=message;
 }
 // 成功打开前先清掉错误提示占位，避免编辑器与提示并存。
 function clearEditorNotice(){ui('code-editor').querySelector('.path-open-notice')?.remove()}
 ui('discard-code-cancel').onclick=()=>{discardAction=null;ui('discard-code-dialog').close()};
 ui('discard-code-confirm').onclick=()=>{dirty=false;const action=discardAction;discardAction=null;ui('discard-code-dialog').close();action?.()};
 // 旧回复可能使用子项目根下的相对路径。仅在原工作区明确报不存在时，探测
 // 已登记工作区中的同一路径；不扫描磁盘、不吞掉权限/网络错误，也不猜选同名文件。
 async function missingPathCandidates(wid,path){
  const candidates=[],seen=new Set();
  for(const w of workspaces){
   const root=String(w.path||'').replace(/\/+$/,'');if(!root)continue;
   const paths=[path],name=root.split('/').pop();
   if(path.startsWith(name+'/'))paths.push(path.slice(name.length+1));
   for(const relative of paths){
    const key=root+'/'+relative;if(seen.has(key)||(w.id===wid&&relative===path))continue;seen.add(key);
    try{
     const value=await api('/workspaces/'+encodeURIComponent(w.id)+'/file?path='+encodeURIComponent(relative));
     if(value.file?.exists)candidates.push({id:w.id,name:w.name,root,path:relative});
    }catch(error){if(error.reason!=='file_not_found')throw error}
   }
  }
  return candidates;
 }
 function chooseFile(candidates,seq){
  clearEditorNotice();
  const box=document.createElement('div');box.className='empty-copy path-open-notice';box.setAttribute('role','status');
  const label=document.createElement('p');label.textContent='多个工作区存在这个文件，请选择实际项目：';box.append(label);
  for(const target of candidates){
   const button=document.createElement('button');button.type='button';button.textContent=target.name+' · '+target.root+'/'+target.path;
   button.onclick=()=>{if(seq!==request)return;afterDiscard(()=>openFile(target.id,target.path))};box.append(button);
  }
  // 保留已有编辑器，选择归属不会清除用户草稿；真正打开仍经过丢弃确认。
  ui('code-editor').prepend(box);status('请选择文件所属项目；当前会话工作区保持不变。');
 }
 async function openFile(wid,path,{resolveMissing=false}={}){
  const seq=++request,originWorkspace=activeWorkspaceId,workspacePath=workspaces.find(w=>w.id===wid)?.path;status('正在读取…');
  try{const data=(await api('/workspaces/'+wid+'/file?path='+encodeURIComponent(path))).file;if(seq!==request||activeWorkspaceId!==originWorkspace)return;clearEditorNotice();
   opened={...data,workspaceId:wid,workspacePath};dirty=false;ui('code-path').textContent=data.path;ui('code-path').title=workspacePath+'/'+data.path;
   if(ui('code-location')){ui('code-location').textContent=workspacePath;ui('code-location').title=workspacePath;ui('copy-code-path').disabled=false;}ui('code-mode').value='read';ui('code-reload').disabled=false;
   makeEditor(data.content);renderPermissions();status(data.comparisonAvailable===false?'当前文件已打开；尚无目录对照快照，未计算历史差异。':data.exists?'左侧为对照版本，右侧为当前文件；可切换只读和编辑。':'文件已删除，展示原文件；此视图不会自动恢复文件。');
  }catch(e){
   if(seq!==request||activeWorkspaceId!==originWorkspace)return;
   if(resolveMissing&&e.reason==='file_not_found'){
    try{
     const candidates=await missingPathCandidates(wid,path);
     if(seq!==request||activeWorkspaceId!==originWorkspace)return;
     if(candidates.length===1){const target=candidates[0];afterDiscard(()=>openFile(target.id,target.path));notify('已定位到项目「'+target.name+'」：'+target.root);return}
     if(candidates.length>1){chooseFile(candidates,seq);return}
    }catch(error){if(seq!==request||activeWorkspaceId!==originWorkspace)return;e=error}
   }
   status(e.message);if(!dirty)showEditorNotice(e.message);
  }
 }
 async function save(){
  if(ui('code-save').disabled||!opened)return;
  const target=opened, content=editor.getValue();saving=true;updateSave();
  try{const result=(await api('/workspaces/'+target.workspaceId+'/file',{path:target.path,content,version:target.version})).file;
   if(opened===target){opened={...result,workspaceId:target.workspaceId,workspacePath:target.workspacePath};dirty=editor.getValue()!==result.content;renderSelection();status('已保存到工作区；原版本保留在本机编辑记录。')}
   await refreshChanges();
  }catch(e){status(e.message)}finally{saving=false;updateSave()}
 }
 ui('code-save').onclick=()=>save();
 ui('code-mode').onchange=()=>{editor?.setEditable(ui('code-mode').value==='edit');updateSave();if(ui('code-mode').value==='edit')editor?.focus()};
 ui('code-diff').onchange=()=>{if(editor){makeEditor(editor.getValue());status(ui('code-diff').checked?'左侧为对照版本，右侧为当前文件。':'单栏显示当前源码；行号、语法高亮和草稿保持。')}};
 ui('code-reload').onclick=()=>afterDiscard(()=>openFile(opened.workspaceId,opened.path));
 // 正文里的文件引用常带文本行号（`a.py:144`、`a.py:82-95`、`a.py:144:5`）。
 // 这里只剥掉路径末尾的行号/行列/行区间标记，不碰路径中间的冒号，也不做任何链接解析。
 // 返回 null 表示剥掉后不成形，调用方按原样交给后续校验报错。
 const TEXT_LINE_SUFFIX=/^(.*\.[A-Za-z0-9]{1,16}):\d+(?::\d+)?(?:-\d+(?::\d+)?)?$/;
 function stripTextLineSuffix(path){
  const m=TEXT_LINE_SUFFIX.exec(path);return m?m[1]:null;
 }
 function relativeInput(value,root){
  // 只做输入便利转换，不 resolve 链接或扩大工作区；后端仍逐层 no-follow 核验。
  let path=value.trim();if((path.startsWith('"')&&path.endsWith('"'))||(path.startsWith("'")&&path.endsWith("'")))path=path.slice(1,-1);
  // `./a/b.md` 里的 `.` 只是当前目录的同义写法，先正规化掉再进边界比对；不解析符号链接。
  if(path.startsWith('./'))path=path.replace(/^(\.\/)+/,'');
  // 行号只在文件名部分剥离；剥离后仍走下面全部越界与合法性检查，不放宽任何边界。
  path=stripTextLineSuffix(path)||path;
  if(!root)throw new Error('请先点击「工作区」关联本地目录，再打开文件。');
  if(!path||path.includes('\\')||path.includes('\0')||path.includes('://')||path.startsWith('~'))throw new Error('请输入相对路径或完整的本机绝对路径。');
  if(path.split('/').includes('..'))throw new Error('路径不能包含 ..；请粘贴文件的完整路径，或切换到对应工作区。');
  const parts=path.split('/').filter(p=>p&&p!=='.'),base=root.split('/').filter(Boolean);
  if(path.startsWith('/')){
   if(!base.every((p,i)=>parts[i]===p))throw new Error('文件不在当前工作区内。点击「工作区」选择或创建文件所在目录，再打开。');
   parts.splice(0,base.length);
  }
  if(!parts.length)throw new Error('这里需要文件路径；如要打开目录，请点击「工作区」。');
  return parts.join('/');
 }
 ui('open-code-form').addEventListener('submit',async e=>{
  e.preventDefault();const wid=activeWorkspaceId;
  const value=ui('open-code-path').value;
  try{const path=relativeInput(value,workspace()?.path);afterDiscard(()=>openFile(wid,path,{resolveMissing:!/^['"]?\//.test(String(value).trim())}))}
  catch(e){
   const target=await locatePath(value,wid),rest=target&&target.rest;
   if(!rest){status(e.message);showEditorNotice(e.message);return}
   try{
    const path=relativeInput(rest,target.root);ui('open-code-path').value=rest;showPanel(true);
    afterDiscard(()=>openFile(target.id,path));
    notify('路径属于工作区「'+target.name+'」，已按该工作区打开。');
   }catch(error){status(error.message);showEditorNotice(error.message)}
  }
 });
 ui('browse-code').addEventListener('click',async()=>{
  const wid=activeWorkspaceId,root=workspace()?.path,old=ui('open-code-path').value,openedRequest=request;
  ui('browse-code').disabled=true;status('请在系统窗口选择文件…');
  try{
   const result=await chooseNativePath('file');
   if(wid!==activeWorkspaceId||root!==workspace()?.path||old!==ui('open-code-path').value||openedRequest!==request)return;
   if(result.cancelled){status('已取消选择，当前文件保持。');return}
   ui('open-code-path').value=result.path;
   const path=relativeInput(result.path,root);afterDiscard(()=>openFile(wid,path));
  }catch(error){status(error.message)}finally{ui('browse-code').disabled=false}
 });
 // 列表携带读取时的工作区，不能将旧条目的相对路径交给新工作区。
 ui('changed-files').addEventListener('click',e=>{
  const b=e.target.closest('[data-code-path]'),list=ui('changed-files');if(!b)return;
  const wid=list.dataset.workspaceId,root=list.dataset.workspacePath,path=b.dataset.codePath;
  if(wid!==activeWorkspaceId||root!==workspace()?.path){status('工作区已变化，正在重新读取文件列表。');refreshChanges();return}
  afterDiscard(()=>{if(wid===activeWorkspaceId&&root===workspace()?.path)openFile(wid,path)});
 });
 // 视图切换只控制布局，不重建编辑器、不改权限，不调用任务控制或模型接口。
 function showPanel(value){if(value&&!ui('tools-panel').hidden)ui('tools-close').click();panel=value;ui('changes-panel').hidden=!value;document.body.classList.toggle('files-open',value);document.body.classList.toggle('coding-view',value);ui('changes-button').setAttribute('aria-expanded',String(value));ui('view-chat').setAttribute('aria-pressed',String(!value));ui('view-code').setAttribute('aria-pressed',String(value));saveLocal('work-view',value?'code':'chat');if(value)refreshChanges();requestAnimationFrame(resizeComposer)}
 document.addEventListener('angelos:detail-view',e=>{if(e.detail==='tools'&&panel)showPanel(false)});
 ui('copy-code-path')?.addEventListener('click',()=>{if(opened)navigator.clipboard.writeText(opened.workspacePath+'/'+opened.path).then(()=>notify('已复制文件完整路径')).catch(()=>status('无法复制，请从路径提示中选取完整路径。'))});
 ui('view-chat').onclick=()=>showPanel(false);ui('view-code').onclick=()=>showPanel(true);
 ui('changes-button').onclick=()=>showPanel(!panel);ui('changes-close').onclick=()=>showPanel(false);ui('changes-refresh').onclick=()=>refreshChanges();
 function renderSelection(){
  const sameWorkspace=opened&&current?.workspaceId===opened.workspaceId&&workspace()?.path===opened.workspacePath;
  const disabled=!codeSelection||!sameWorkspace||!online||attaching||sending;
  ui('ask-code').disabled=disabled;ui('ask-code-new').disabled=disabled;
  ui('code-selection-note').textContent=!sameWorkspace&&opened?'文件属于其他工作区或目录已改变，请切回原工作区或重新打开文件。':codeSelection?`${codeSelection.side==='original'?'对照版本':dirty?'未保存草稿':'读取快照'} · ${codeSelection.startLine}:${codeSelection.startColumn}–${codeSelection.endLine}:${codeSelection.endColumn}${current?.turns.some(active)?' · 当前任务继续运行，可在新会话提问':''}`:'选中代码后，可以带着文件位置提问。';
 }
 async function askSelection(inNewConversation){
  renderSelection();if(ui('ask-code').disabled)return;
  if(!inNewConversation&&current.pending.length>=8)throw new Error('每轮最多 8 份资料，请先移除不需要的附件。');
  // 在任何 await 之前固定选区、工作区和目标会话；迟到响应不能混入新会话。
  const target={...opened}, snippet={...codeSelection}, version=selection;
  const body=JSON.stringify({source:'user_editor_selection',workspace:target.workspacePath,path:target.path,
   diskVersionAtRead:target.version,contentState:snippet.side==='original'?'comparison_snapshot':dirty?'unsaved_draft':'read_snapshot',
   range:{startLine:snippet.startLine,startColumn:snippet.startColumn,endLine:snippet.endLine,endColumn:snippet.endColumn},
   note:'用户选中的参考代码，不是当前磁盘内容或执行结果的证明；修改前须重新读取文件。行列从1开始，结束位置不包含在选区内。',code:snippet.text},null,2);
  if(body.length>12000)throw new Error('选区过长，请缩小到 12,000 字以内（含文件位置说明）。');
  let cid=current.id;attaching=true;renderSelection();
  try{
   if(inNewConversation){const c=(await api('/conversations',{workspaceId:target.workspaceId})).conversation;cid=c.id;upsert(c);if(version===selection)await selectConversation(cid)}
   const data=btoa(Array.from(new TextEncoder().encode(body),b=>String.fromCharCode(b)).join(''));
   const name=(target.path.split('/').pop().slice(0,55))+` L${snippet.startLine}-${snippet.endLine}.json`;
   const result=await api('/conversations/'+cid+'/attachments',{name,data});
   if(current?.id===cid){current.pending.push(result.attachment);current.attachments.push(result.attachment);saveDraft(current);renderPending();ui('message-input').focus()}
   else{const draft=drafts({id:cid});draft.pending.push(result.attachment);saveLocal('draft:'+cid,draft)}
   notify('代码选区已加入'+(current?.id===cid?'当前':'目标')+'会话草稿。写下问题后发送；原任务不受影响。');
  }finally{attaching=false;renderSelection()}
 }
 ui('ask-code').onclick=safely(()=>askSelection(false));ui('ask-code-new').onclick=safely(()=>askSelection(true));
 async function refreshChanges(){
  renderSelection();
  const wid=activeWorkspaceId,w=workspace(),root=w?.path,key=JSON.stringify([wid,root]);
  // 即便上一个目录请求尚未返回或新目录扫描失败，也先撤下旧列表。
  if(selectedWorkspace!==key){
   selectedWorkspace=key;ui('changed-files').replaceChildren();ui('changed-files').dataset.html='';
   ui('changed-files').dataset.workspaceId=wid||'';ui('changed-files').dataset.workspacePath=root||'';
   ui('changes-count').textContent='0';ui('changes-totals').textContent='';ui('changes-note').textContent='正在读取当前工作区的改动…';
   if(opened&&(opened.workspaceId!==wid||opened.workspacePath!==root))status('已切换工作区；编辑器保留原工作区的文件和草稿。');
  }
  if(refreshing||!wid||!online)return;
  if(agent.workbenchControls!==true){ui('changes-note').textContent='后端尚未加载文件与权限更新，原有对话仍可使用。';return}
  if(!w?.path){ui('changes-note').textContent='当前工作区未关联目录。请先在工作区中设置本地目录。';ui('changed-files').replaceChildren();ui('changes-count').textContent='0';ui('changes-totals').textContent='';return}
  refreshing=true;
  try{const data=(await api('/workspaces/'+wid+'/changes')).changes;if(wid!==activeWorkspaceId||root!==workspace()?.path)return;
   ui('changes-count').textContent=data.files.length;ui('changes-totals').innerHTML=`<span class="diff-added">+${data.added}</span> <span class="diff-removed">−${data.removed}</span>`;
   ui('changes-note').textContent='相对 '+data.baseline+' · 每 2.5 秒更新 · 包含工作区内所有来源的改动'+(data.skipped.length?' · '+data.skipped.length+' 个文件无法预览':'');
   const html=data.files.map(f=>`<button data-code-path="${esc(f.path)}"><span class="file-change-kind">${labels[f.status]}</span><span class="file-change-path">${esc(f.path)}</span><span class="diff-added">+${f.added}</span><span class="diff-removed">−${f.removed}</span></button>`).join('')||'<p class="empty-copy">没有检测到文件改动</p>';
   if(ui('changed-files').dataset.html!==html){ui('changed-files').innerHTML=html;ui('changed-files').dataset.html=html}
   if(opened?.workspaceId===wid){const file=data.files.find(f=>f.path===opened.path);if(file&&file.version!==opened.version)status('磁盘版本已变化；当前视图保留，点击「重新读取」查看。')}
  }catch(e){if(wid===activeWorkspaceId&&root===workspace()?.path){ui('changed-files').replaceChildren();ui('changed-files').dataset.html='';ui('changes-count').textContent='0';ui('changes-totals').textContent='';ui('changes-note').textContent=e.message}}finally{refreshing=false}
 }
 window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});
 // 供对话正文里的路径链接调用：把用户可见的路径（绝对或相对当前工作区）按既有
 // relativeInput 规则换算，再复用 openFile；越界或非法路径只提示不打开，不新增读取接口。
 // 已打开文件有未保存草稿时沿用 afterDiscard 的确认对话框，不静默覆盖草稿。
 function openPathValue(value,wid=activeWorkspaceId){
   return openPathValueAsync(value,wid);
 }
 async function openPathValueAsync(value,wid){
  const workspacePath=workspaces.find(w=>w.id===wid)?.path;
  try{const path=relativeInput(String(value||''),workspacePath);showPanel(true);afterDiscard(()=>openFile(wid,path,{resolveMissing:!/^['"]?\//.test(String(value).trim())}));return}
  catch(error){
   // 不属于当前工作区时，先在其他已注册工作区中找归属；没有归属才按原样提示。
   const target=await locatePath(String(value||''),wid),rest=target&&target.rest;
   if(rest){
    try{const path=relativeInput(rest,target.root);showPanel(true);afterDiscard(()=>openFile(target.id,path));notify('文件属于工作区「'+target.name+'」，已按该工作区打开。');return}
    catch(inner){status(inner.message);showEditorNotice(inner.message);return}
   }
   status(error.message);showEditorNotice(error.message);
  }
 }
 window.angelosWorkspace={permissions:loadPermissions,refresh:refreshChanges,openPath:openPathValue};
 showPanel(readLocal('work-view','chat')==='code');
 loadPermissions().catch(showError);
})();
