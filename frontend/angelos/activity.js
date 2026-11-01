'use strict';
/* SSE只更新本地展示；连接故障/切换视图不提交、取消或重跑任何任务。 */
window.AngelosActivity={
  // SSE与轮询可能乱序；仅同一任务/尝试的新版本可覆盖，等待计时也不能倒退。
  progressFor(turn,snapshot){
    const polled=turn.live_progress,live=snapshot?.progress;
    if(!live||live.available===false)return polled||live;
    if((live.taskId&&live.taskId!==turn.task_id)||(turn.task_id&&!live.taskId))return polled;
    if(polled?.available!==false&&polled?.taskId){
      if((live.attempt||0)<(polled.attempt||0))return polled;
      if(live.attempt===polled.attempt&&((live.version||0)<(polled.version||0)||
        (live.version===polled.version&&(live.waitSeconds||0)<(polled.waitSeconds||0))))return polled;
    }
    return live;
  },
  create({api,changed,layout=()=>{},connect=url=>new EventSource(url),visible=()=>!document.hidden,log=null}){
    let cid='',generation=0,source=null,target='',turns=[],snapshots=new Map(),pending='',deferred=false,followLatest=true;
    const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const labels={ok:'已取得回执',error:'执行报错',blocked:'被阻止',not_executed:'未执行',pending_receipt:'等待回执',unknown:'结果待确认'};
    function disconnect(){if(source){source.close();source=null}}
    function accept(value,version){
      if(version!==generation||value?.conversationId!==cid||!value.turnId)return;
      snapshots.set(value.turnId,value);while(snapshots.size>32)snapshots.delete(snapshots.keys().next().value);
      if(value.turnId===target)render();changed();
    }
    function watch(){
      if(source||!cid||!turns.some(t=>['preparing','submitting','executing'].includes(t.stage))||!visible())return;
      const version=generation;
      let stream;
      try{stream=source=connect('/angelos/api/conversations/'+encodeURIComponent(cid)+'/activity')}
      catch(_){$('tools-note').textContent='实时连接不可用，仍可刷新查看记录。';return}
      stream.onmessage=e=>{
        if(version!==generation||source!==stream)return;
        try{const value=JSON.parse(e.data);accept(value,version);if(value.active===false)disconnect()}
        catch(_){$('tools-note').textContent='实时快照无法读取，保留上次记录。'}
      };
      stream.onerror=()=>{if(version===generation&&source===stream)$('tools-note').textContent='实时连接暂时中断，保留上次记录，正在重新连接。'};
      stream.addEventListener('unavailable',()=>{if(version===generation&&source===stream){disconnect();$('tools-note').textContent='实时快照暂不可读，等待下一次连接。'}});
      stream.addEventListener('rotate',()=>{if(version===generation&&source===stream){disconnect();watch()}});
    }
    async function load(){
      if(!target||!cid||pending===target)return;
      const id=cid,turn=target,version=generation;pending=turn;
      try{const data=await api('/conversations/'+encodeURIComponent(id)+'/tools/'+encodeURIComponent(turn));accept(data.activity,version)}
      catch(_){if(version===generation)$('tools-note').textContent='工具记录读取失败，保留上次记录。'}
      finally{if(version===generation&&pending===turn)pending=''}
    }
    function selected(){
      const selection=document.getSelection();
      if(!selection||selection.isCollapsed)return false;
      for(let i=0;i<selection.rangeCount;i++)if(selection.getRangeAt(i).intersectsNode($('tool-calls')))return true;
      return false;
    }
    // 轮次内快照替换时只更新已有节点；新调用才追加。返回 true 表示已按增量处理。
    function incremental(value){
      const list=$('tool-calls');if(!list.dataset.rendered||!value?.available)return false;
      const seen=new Set([...list.querySelectorAll('[data-call]')].map(n=>n.dataset.call));
      const rows=value.rows||[];let added=0;
      for(const r of rows){
        const id=String(r.id),node=list.querySelector(`[data-call="${CSS.escape(id)}"]`);
        if(node){
          // 只改状态与结果，不重建节点，保住用户已展开的详情与选区。
          const state=node.querySelector('[data-state]');if(state){state.dataset.state=r.status;state.textContent=labels[r.status]||'结果待确认'}
          const pre=node.querySelector('[data-field="result"] pre');if(pre)pre.textContent=JSON.stringify({result:r.result,error:r.error},null,2);
          continue;
        }
        const article=document.createElement('article');article.className='tool-call';article.dataset.call=id;
        const i=rows.indexOf(r);
        article.innerHTML=`<div class="tool-call-head"><strong>${esc(r.name||'未知工具')}</strong><span data-state="${esc(r.status)}">${esc(labels[r.status]||'结果待确认')}</span></div><small>调用 ${i+1} · ${esc(id)}</small><div class="tool-call-actions"><button type="button" data-log="${esc(id)}">查看完整日志</button></div><details data-field="arguments"><summary>参数</summary><pre>${esc(JSON.stringify(r.arguments??{},null,2))}</pre></details><details data-field="result"><summary>结果与错误</summary><pre>${esc(JSON.stringify({result:r.result,error:r.error},null,2))}</pre></details>`;
        list.appendChild(article);added++;
      }
      if(added)list.dataset.rendered=null;
      return true;
    }
    function render(){
      if(selected()){deferred=true;return}deferred=false;
      const value=snapshots.get(target)?.tools,list=$('tool-calls');
      $('tools-count').textContent=!target?'0':!value||!value.available?'—':(value.total||0)+(value.rows||[]).filter(r=>r.status==='pending_receipt').length;
      if(!value){$('tools-note').textContent=target?'正在读取本轮工具记录…':'选择一轮对话查看工具调用。';list.textContent='选择一轮对话查看工具调用。';list.dataset.rendered='';return}
      $('tools-note').textContent=!value.available?'本轮工具记录暂不可读。':
        `第 ${value.attempt??'—'} 次尝试 · ${value.total||0} 条执行回执`+
        (value.omitted?` · 另有 ${value.omitted} 条未展开`:'')+
        (value.information?.status==='truncated'?' · 长参数或结果已节选，原账本保留。':'');
      // 增量路径：同一轮次收到新快照时，不重建整张列表，只追加新调用并更新状态。
      if(incremental(value))return;
      const json=v=>esc(JSON.stringify(v??{},null,2));
      const html=(value.rows||[]).map((r,i)=>`<article class="tool-call" data-call="${esc(r.id)}"><div class="tool-call-head"><strong>${esc(r.name||'未知工具')}</strong><span data-state="${esc(r.status)}">${esc(labels[r.status]||'结果待确认')}</span></div><small>调用 ${i+1} · ${esc(r.id)}</small><div class="tool-call-actions"><button type="button" data-log="${esc(r.id)}">查看完整日志</button></div><details data-field="arguments"><summary>参数</summary><pre>${json(r.arguments)}</pre></details><details data-field="result"><summary>结果与错误</summary><pre>${json({result:r.result,error:r.error})}</pre></details></article>`).join('')||'<p class="tools-empty">本轮尚无工具调用。</p>';
      if(list.dataset.rendered===html)return;
      const opened=new Set([...list.querySelectorAll('details[open]')].map(n=>n.closest('[data-call]').dataset.call+':'+n.dataset.field));
      const top=list.scrollTop,atEnd=list.scrollHeight-top-list.clientHeight<60;
      list.innerHTML=html;list.dataset.rendered=html;
      list.querySelectorAll('details').forEach(n=>n.open=opened.has(n.closest('[data-call]').dataset.call+':'+n.dataset.field));
      list.scrollTop=atEnd?list.scrollHeight:top;
    }
    function open(value){
      if(value&&typeof CustomEvent==='function'&&document.dispatchEvent)document.dispatchEvent(new CustomEvent('angelos:detail-view',{detail:'tools'}));
      $('tools-panel').hidden=!value;document.body.classList.toggle('tools-open',value);
      $('tools-button').setAttribute('aria-expanded',String(value));if(value){fit();void load();render()}
      layout();
    }
    // 窄屏顶栏会换行，独立栏从实际顶栏下沿开始，保留顶部入口和关闭按钮。
    function fit(){if(!$('tools-panel').hidden)$('tools-panel').style.top=Math.ceil(document.querySelector('.topbar').getBoundingClientRect().bottom)+'px'}
    const resize=typeof ResizeObserver==='function'?new ResizeObserver(fit):null;
    resize?.observe(document.querySelector('.topbar'));
    window.addEventListener('resize',fit);
    $('tools-button').addEventListener('click',()=>open($('tools-panel').hidden));
    $('tools-close').addEventListener('click',()=>open(false));
    $('tools-turn').addEventListener('change',()=>{target=$('tools-turn').value;followLatest=target===turns.at(-1)?.id;render();void load()});
    // 点击“查看完整日志”打开日志面板；只切换展示，不重新请求任务或触发执行。
    $('tool-calls').addEventListener('click',event=>{
      const button=event.target.closest('[data-log]');if(!button||!log)return;
      const row=(snapshots.get(target)?.tools?.rows||[]).find(r=>String(r.id)===button.dataset.log);
      if(row)log.show(cid,target,button.dataset.log);
      layout();
    });
    $('view-code').addEventListener('click',()=>open(false));
    document.addEventListener('selectionchange',()=>{if(deferred&&!selected())render()});
    document.addEventListener('visibilitychange',()=>{if(!visible())disconnect();else watch()});
    const timestamp=rows=>Math.max(0,...(rows||[]).map(r=>r.updated||r.created||0));
    return {
      select(id,items){
        if(cid!==id){
          disconnect();generation++;cid=id;snapshots.clear();target='';pending='';followLatest=true;deferred=false;
          // 选区只冻结当前会话；换会话立即移除旧节点，避免把上一会话的回执留在新会话中。
          $('tool-calls').textContent='选择一轮对话查看工具调用。';$('tool-calls').dataset.rendered='';
          $('tools-note').textContent='选择一轮对话查看工具调用。';$('tools-count').textContent='0';
        }
        turns=items||[];
        if(followLatest||!turns.some(t=>t.id===target))target=turns.at(-1)?.id||'';
        const options=turns.map((t,i)=>`<option value="${esc(t.id)}">第 ${i+1} 轮 · ${esc((t.text||'').slice(0,32))}</option>`).join('');
        if($('tools-turn').dataset.rendered!==options){$('tools-turn').innerHTML=options;$('tools-turn').dataset.rendered=options}
        $('tools-turn').value=target;
        if(!turns.some(t=>['preparing','submitting','executing'].includes(t.stage)))disconnect();else watch();
        // 历史轮次也只读一次，让顶部计数有来源；未取得快照时显示未知，不伪装成零调用。
        if(!snapshots.has(target))void load();render();
      },
      reasoning(turn){const value=snapshots.get(turn.id)?.reasoning;return value&&timestamp(value)>=timestamp(turn.reasoning)?value:turn.reasoning},
      progress(turn){return window.AngelosActivity.progressFor(turn,snapshots.get(turn.id))},
      show(turn){target=turn;followLatest=target===turns.at(-1)?.id;$('tools-turn').value=turn;open(true)},
      // 供日志面板回写展开偏好等展示状态；不触碰任务事实。
      refresh(){render()},
      stop(){disconnect();generation++},
    };
  }
};
