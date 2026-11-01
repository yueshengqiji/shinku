'use strict';
/* 用量独立低频拉取，和任务进度轮询分开。展示对象只存在浏览器，不写会话或记忆。
   只使用供应商回执；缓存属于输入分项，不能再加进合计。 */
window.AngelosUsage = {
  create({api, changed, now=()=>performance.now(), visible=()=>!document.hidden}) {
    const interval=15000, number=n=>Number(n).toLocaleString('zh-CN');
    let cid='', generation=0, snapshot=null, error=false, enabled=false;
    let lastAttempt=-Infinity, pending=false, timer;
    function schedule(delay=interval){clearTimeout(timer);timer=setTimeout(refresh,delay)}
    const ratio=s=>typeof s?.cacheHitRatio==='number'?(s.cacheHitRatio*100).toFixed(1)+'%':'未提供';
    const count=(s,key)=>{
      if(s.totals[key]!==null)return number(s.totals[key]);
      return s.known[key]===null?'未提供':number(s.known[key])+'（部分）';
    };
    function metrics(s) {
      if(!s?.requests)return '<span>尚无供应商用量记录</span>';
      const total=s.totalTokens!==null?number(s.totalTokens):
        s.knownTotalTokens!==null?number(s.knownTotalTokens)+'（部分）':'未提供';
      return `<span>输入 <b>${count(s,'input_tokens')}</b></span><span>输出 <b>${count(s,'output_tokens')}</b></span><span>合计 <b>${total}</b> token</span><span>缓存命中 <b>${count(s,'cache_hit_tokens')}</b> token · 命中率 <b>${ratio(s)}</b></span>`;
    }
    function cacheDetails(s) {
      if(!s?.cacheGroups)return '';
      const labels={ordinary:'普通调用',summary:'摘要调用',post_compaction:'压缩后首轮',unclassified:'旧记录／未分类'};
      const groups=Object.entries(labels).map(([key,label])=>{
        const group=s.cacheGroups[key];
        return group?.requests?`<tr><td>${label}</td><td>${number(group.requests)}</td><td>${count(group,'input_tokens')}</td><td>${count(group,'output_tokens')}</td><td>${ratio(group)}</td></tr>`:'';
      }).join('');
      // 只渲染后端白名单中的计数/字段名，不展示请求正文、推理或工具参数。
      const fields=new Set(['model','tools','system_extra_blocks','thinking','reasoning_effort','extra_body',
        'response_format','tool_choice','temperature','max_tokens','max_completion_tokens','prompt_cache_key','prompt_cache_retention']);
      const requests=(s.recentRequests||[]).map((r,i)=>{
        const p=r.prefix;
        const text=!p?.available?'未采集':!p.baseline?'本绑定首条记录':
          `相同前 ${number(p.matchedMessages)} 条消息；${p.firstChangedMessage===null?'消息未变':'第 '+number(p.firstChangedMessage+1)+' 条起变化'}`+
          (p.changedEnvelope?.length?'；封装变化：'+p.changedEnvelope.filter(k=>fields.has(k)).join(', '):'；封装未变');
        return `<tr><td>${number(i+1+(s.omittedRequestDetails||0))}</td><td>${labels[r.category]||'未分类'}</td><td>${ratio(r)}</td><td>${text}</td></tr>`;
      }).join('');
      return `<section class="turn-usage"><strong>缓存分组</strong><div style="overflow-x:auto"><table class="phase-table"><thead><tr><th>调用类型</th><th>请求</th><th>输入 token</th><th>输出 token</th><th>命中率</th></tr></thead><tbody>${groups}</tbody></table></div><p class="usage-note">按实际输入 token 加权。结构前缀是完整消息的比较，不是供应商 token 前缀或命中证明；配置变化可能影响缓存。历史记录不补造分类。逐请求仅显示最近 20 条。</p><div style="overflow-x:auto"><table class="phase-table"><thead><tr><th>记录</th><th>类型</th><th>本次命中率</th><th>与上一次有诊断的请求比较</th></tr></thead><tbody>${requests}</tbody></table></div></section>`;
    }
    // 外层已有用量折叠区；此处保持可见，避免每15秒替换快照时再次折叠。
    function phases(s) {
      const p=s?.performance;
      if(!p?.available)return '<p class="usage-note">阶段耗时尚未采集。</p>';
      const esc=value=>String(value??'未知').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      const ms=value=>typeof value==='number'?(value/1000).toFixed(3)+' 秒':'未知';
      const rows=p.phases.map(r=>`<tr><td>${esc(r.label)}</td><td>${ms(r.totalMs)}</td><td>${esc(r.measured)}/${esc(r.samples)}</td><td>${ms(r.p50Ms)} / ${ms(r.p95Ms)}</td><td>${esc(r.requests)} 次 · ${ms(r.requestMs)}${r.incompleteRequestTiming?"（部分）":""}</td><td>${esc(r.knownInputTokens)} / ${esc(r.knownOutputTokens)}${r.incompleteUsage?'（部分）':''}</td><td>异常 ${esc(r.failed)} · 请求失败 ${esc(r.requestFailed)} · 未结束 ${esc(r.unsettled+r.requestUnsettled)}</td></tr>`).join('');
      return `<section class="turn-usage"><strong>阶段耗时与模型请求</strong><p class="usage-note">阶段包含子阶段，不可相加；p50/p95 按已结束调用采样。模型请求耗时之和不等于用户等待时间。正常返回不代表业务完成。</p>${p.truncated?'<p class="usage-note">采集不完整或容量已满，仅显示保留样本。</p>':''}<div style="overflow-x:auto"><table class="phase-table"><thead><tr><th>阶段</th><th>累计耗时</th><th>已计时/样本</th><th>单次 p50/p95</th><th>已记录模型请求</th><th>输入/输出 token</th><th>诊断</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
    }
    async function refresh() {
      if(!enabled||!cid||pending)return;
      if(!visible()){schedule();return}
      if(now()-lastAttempt<interval){schedule(interval-(now()-lastAttempt));return}
      const id=cid, version=generation;
      pending=true;lastAttempt=now();
      try {
        const value=(await api('/conversations/'+encodeURIComponent(id)+'/usage')).usage;
        if(version!==generation||id!==cid)return;
        if(!value?.available||value.conversationId!==id)throw new Error('usage_unavailable');
        snapshot=value;error=false;
      } catch(_) {
        if(version===generation)error=true; // 保留最后一次快照，绝不把断连显示为零消耗。
      } finally {
        if(version===generation){pending=false;changed();schedule()}
      }
    }
    // 页面隐藏时跳过。恢复可见后在下一次定时检查读取，不补发错过的轮询。
    return {
      select(id,supported) {
        if(cid===id&&enabled===!!supported)return;
        clearTimeout(timer);
        generation++;cid=id;enabled=!!supported;snapshot=null;error=false;
        pending=false;lastAttempt=-Infinity;void refresh();
      },
      // 状态栏只保留会话加权命中；缺失/空数据不能冒充0%，断连保留旧值并标注。
      statusHTML() {
        const s=snapshot?.summary;
        if(!enabled||!s?.requests)return '';
        return `会话命中 ${ratio(s)}${error?' · 待更新':''}`;
      },
      // 和会话统计共用15秒快照；缺失供应商数据明确标注，不能用会话值代替累计值。
      instanceStatusHTML() {
        if(!enabled)return '累计用量不可用';
        if(!snapshot)return '累计命中待采样';
        const instance=snapshot.instance,s=instance?.summary;
        if(!instance?.available||!s?.requests)return '累计命中未提供';
        return `累计命中 ${s.invalidCache?'未提供':ratio(s)}${error?' · 待更新':''}`;
      },
      panel(scope='session',id) {
        const s=scope==='instance'?snapshot?.instance?.summary:scope==='turn'?snapshot?.turns?.[id]:snapshot?.summary;
        const label=scope==='instance'?'整个 Angelos':scope==='turn'?'本轮':'当前会话';
        const sampled=snapshot?new Date(snapshot.sampledAt*1000).toLocaleTimeString('zh-CN',{hour12:false}):'尚未采样';
        let note=!enabled?'当前服务未提供用量统计。':error?'更新失败，保留上次统计。':!snapshot?'正在读取用量。':
          scope==='instance'&&!snapshot.instance?.available?'全局累计暂不可读。':scope==='turn'&&!s?'本轮等待下一次采样。':
          s?.available===false?'本轮用量暂不可读。':'按已记录请求统计。';
        if(s?.incompleteRequests)note+=` ${number(s.incompleteRequests)} 次请求用量不完整。`;
        if(s?.unsettledRequests)note+=` ${number(s.unsettledRequests)} 次尚无最终回执。`;
        if(s?.invalidCache)note+=' 缓存回执不一致，命中率不可用。';
        if(scope==='session'&&snapshot?.unrecordedTurns)note+=` ${number(snapshot.unrecordedTurns)} 轮未采集或暂不可读。`;
        if(snapshot?.healthy===false||scope==='instance'&&snapshot?.instance?.healthy===false)note+=' 采集不完整。';
        if(scope==='instance'&&s?.excludedCacheRequests)note+=` 命中率覆盖 ${number(s.cacheCoverageRequests)} / ${number(s.requests)} 次请求，缺失或异常回执未计入。`;
        return {summary:`<div class="usage-heading"><strong>${label}</strong><small>每 15 秒更新 · ${sampled}</small></div><div class="usage-metrics">${metrics(s)}</div>${s?.requests?`<p class="usage-note">${number(s.requests)} 次模型请求</p>`:''}<p class="usage-note">${note} 命中率按输入 token 加权，缓存命中已包含在输入中。</p>`,
          advanced:s?.requests?(cacheDetails(s)+phases(s)):'<p class="usage-note">暂无可展开的请求记录。</p>'};
      },
      compactHTML(id) {
        if(!enabled||!id)return '';
        const s=snapshot?.turns?.[id];
        // 新轮次尚未被15秒快照采集时，不能借上一轮或会话累计冒充本轮。
        let status=error?'更新失败，保留上次统计':!s?'等待下一次用量采样':
          !s.available?'本轮用量暂不可读':!s.requests?'本轮尚无用量记录':
          s.invalidCache?'缓存回执不一致':s.incompleteRequests?'含未返回用量，显示已知小计':'';
        if(snapshot?.healthy===false)status+=' · 采集不完整';
        const available=s?.available&&s.requests;
        return `<div class="usage-current-line"><strong>本轮</strong><span>缓存命中率 <b>${available?ratio(s):'—'}</b></span><span>输入 <b>${available?count(s,'input_tokens'):'—'}</b></span><span>输出 <b>${available?count(s,'output_tokens'):'—'}</b> token</span><small>每 15 秒更新</small></div>${status?`<p class="usage-current-note">${status}</p>`:''}`;
      },
      sessionHTML() {
        if(!enabled)return '';
        const s=snapshot?.summary;
        return s?.requests?`<div class="usage-session-line">会话累计 · 命中率 ${ratio(s)} · 输入 ${count(s,'input_tokens')} / 输出 ${count(s,'output_tokens')} token</div>`:'';
      },
      instanceHTML() {
        if(!enabled)return '';
        const value=snapshot?.instance,s=value?.summary;
        const note=error?'更新失败，保留上次统计':!snapshot?'正在读取累计用量':!value?.available?'累计统计尚不可用':
          s.excludedCacheRequests?`命中率覆盖 ${number(s.cacheCoverageRequests)} / ${number(s.requests)} 次请求；缺失或异常回执未计入比率`:'命中率按输入 token 加权';
        const warning=value?.healthy===false?' · 采集不完整':'';
        return `<div class="usage-instance-line"><strong>Angelos 累计</strong>${value?.available?metrics(s):'<span>等待统计</span>'}<small>每 15 秒更新</small></div><p class="usage-current-note">${note}${warning}</p>`;
      },
      summaryHTML() {
        if(!enabled)return '';
        const s=snapshot?.summary;
        const sampled=snapshot?new Date(snapshot.sampledAt*1000).toLocaleTimeString('zh-CN',{hour12:false}):'—';
        let note=error?'更新失败，保留上次统计。':snapshot?'按已记录请求统计。':'正在读取供应商用量。';
        if(snapshot?.unrecordedTurns)note+=` ${snapshot.unrecordedTurns} 轮未采集或暂不可读。`;
        if(s?.incompleteRequests)note+=` ${s.incompleteRequests} 次请求用量不完整。`;
        if(s?.unsettledRequests)note+=` ${s.unsettledRequests} 次尚无最终回执。`;
        if(snapshot?.healthy===false)note+=' 采集发生过错误，统计可能不完整。';
        if(s?.invalidCache)note+=' 缓存回执不一致，命中率不可用。';
        return `<div class="usage-heading"><strong>会话用量</strong><small>每 15 秒刷新 · 更新于 ${sampled}</small></div><div class="usage-metrics">${metrics(s)}</div><p class="usage-note">${note} 缓存命中率按输入 token 计算。</p>${cacheDetails(s)}${phases(s)}`;
      },
      turnHTML(id) {
        const s=snapshot?.turns?.[id];
        if(!enabled||!s)return '';
        if(!s.available)return '<div class="turn-usage">本轮用量暂不可读</div>';
        return `<details class="turn-usage" data-detail="usage-${id.replace(/[^a-zA-Z0-9_-]/g,'')}"><summary>本轮用量 · 命中率 ${ratio(s)} · ${number(s.requests)} 次已记录请求</summary><div class="usage-metrics">${metrics(s)}</div>${s.incompleteRequests?'<p class="usage-note">含未返回用量的请求，当前显示已知小计。</p>':''}${cacheDetails(s)}${phases(s)}</details>`;
      },
      stop(){clearTimeout(timer);generation++;enabled=false},
    };
  }
};
