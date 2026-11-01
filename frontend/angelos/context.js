'use strict';
/* 容量来自后端，按供应商/模型持久化。复用现有目录与任务快照；间隔更新
   只刷新文字，不额外轮询或调用模型。正在编辑的容量不被后台刷新覆盖。 */
(() => {
  const sources = {host_config:'宿主配置', provider_metadata:'供应商元数据',
    angelos_override:'手动确认', deepseek_official_20261019:'已登记官方资料', unknown:'未知'};
  Object.assign(errors, {
    invalid_context_window:'请输入 1 至 2,000,000 之间的整数，或留空恢复自动识别。',
    model_capacity_update_failed:'容量未保存，请重试；原设置保留。',
    'planner_error:execution_context_oversized':'规划提示超过本机装配限额，原始资料与读取记录保留。',
    'planner_error:conversation_authorization_changed':'任务授权已变化，本轮执行已停止。',
    'planner_error:work_plan_task_changed_during_read':'读取工作计划时任务版本发生变化，本轮未继续执行。'
  });
  let selectionKey='', saving=false, dirty=false;
  const input=$('model-context-window'), note=$('model-context-source'), status=$('model-context-status');
  const summary=document.createElement('small');summary.id='context-capacity-status';
  summary.className='dialog-note';summary.setAttribute('role','status');
  $('conversation-caption').parentNode.appendChild(summary);
  function render() {
    if(document.hidden)return;
    const ctx=lastTurn()?.context_usage;
    const text=ctx ? (ctx.window_tokens ?
      `本轮窗口 ${ctx.window_tokens.toLocaleString('zh-CN')} token · 来源：${sources[ctx.capacity_source]||'历史记录'} · 占用为估算` :
      '本轮上下文窗口未知 · 自动压缩未启用，请在设置中确认容量') : '';
    if(summary.textContent!==text)summary.textContent=text;
    if(!$('settings-dialog').open)return;
    const providerId=$('settings-provider').value,modelId=$('settings-model').value;
    const key=JSON.stringify([providerId,modelId]);
    const row=modelProvider(providerId)?.models.find(m=>m.id===modelId)||{};
    if(selectionKey!==key){selectionKey=key;dirty=false;status.textContent='';}
    if(!dirty && document.activeElement!==input && !saving)input.value=row.contextWindowOverride||'';
    const current=modelCatalog.selected?.providerId===providerId && modelCatalog.selected?.modelId===modelId;
    const limits=current?modelCatalog.contextLimits:null;
    const size=limits?.window_tokens||row.contextWindowOverride||row.contextWindow||0;
    const source=limits?.source||(row.contextWindowOverride?'angelos_override':row.contextWindow?'provider_metadata':'unknown');
    note.textContent=(size?`当前容量 ${size.toLocaleString('zh-CN')} token · ${sources[source]||source}`:
      '容量未知，90% 自动压缩未启用。')+' 填写此连接实际支持的窗口；留空恢复自动识别。';
    $('save-context-window').disabled=saving||modelSaving||!providerId||!modelId;
  }
  input.addEventListener('input',()=>{dirty=true;});
  $('save-context-window').addEventListener('click',async()=>{
    if(saving)return;
    const raw=input.value.trim();
    if(input.validity.badInput || raw && !/^[0-9]+$/.test(raw) || raw && (+raw<1||+raw>2000000)){
      status.textContent=errors.invalid_context_window;return;
    }
    const data={providerId:$('settings-provider').value,modelId:$('settings-model').value,contextWindow:raw?Number(raw):0};
    saving=true;render();
    try{
      modelCatalog=(await api('/models/context-window',data)).models;
      if(JSON.stringify([$('settings-provider').value,$('settings-model').value])===JSON.stringify([data.providerId,data.modelId])){
        dirty=false;status.textContent='容量已保存，原模型与思考档位保持不变。';
      }
    }catch(error){status.textContent=error.message;}
    finally{saving=false;render();}
  });
  for(const id of ['settings-provider','settings-model'])$(id).addEventListener('change',render);
  $('settings-button').addEventListener('click',()=>{selectionKey='';render();});
  // 同一目录对象更新时重画来源；不因刷新页面自动提交容量或发出摘要请求。
  setInterval(render,5000);
  render();
})();
