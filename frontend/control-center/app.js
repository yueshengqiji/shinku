// 视图状态由当前页面持有；业务状态来自同源 API，不复制后端权限与任务生命周期。
const renderedHTML = new WeakMap();
const listLimits = new Map();
let traceChannel = 'all';
let permissionDraft = null;
let lastSnapshot = null;
let refreshInFlight = null;
let autoRefreshEnabled = true;
let mutationsInFlight = 0;
let characterName = '角色';
// 只改界面标签，消息正文和历史作者不做姓名替换；数据始终来自当前端口。
function renderCharacter(character) {
  characterName = character?.name || '角色';
  document.title = `${characterName} / 控制中枢`;
  document.querySelector('.brand-name').textContent = characterName;
  document.querySelector('.brand').setAttribute('aria-label', `${characterName}控制中枢首页`);
  document.getElementById('character-code').textContent = (character?.id || 'instance').toUpperCase();
  document.querySelector('.page-footer span').textContent = `${characterName} / CONTROL CENTER`;
  document.getElementById('console-chat-title').textContent = `和${characterName}对话`;
  document.querySelector('label[for="console-chat-input"]').textContent = `发给${characterName}`;
  document.getElementById('console-chat-input').placeholder = `和${characterName}说点什么……`;
}

// 这里只持有页面正文与会话定位符；角色记忆和执行状态仍由当前实例后端持有。
// 与 Angelos 的工作台、会话库、模型选择和记忆接口均无关联。
let consoleConversationId = 'control-center:' + crypto.randomUUID();
let consoleChatBusy = false;
function appendConsoleMessage(speaker, text) {
  const target = document.getElementById('console-chat-messages');
  const item = document.createElement('article');
  item.className = 'console-chat-message';
  const label = document.createElement('strong');
  label.textContent = speaker;
  const body = document.createElement('p');
  body.textContent = text;
  item.append(label, body);
  target.append(item);
  target.scrollTop = target.scrollHeight;
}
document.getElementById('console-chat-new').addEventListener('click', () => {
  if (consoleChatBusy) return;
  consoleConversationId = 'control-center:' + crypto.randomUUID();
  document.getElementById('console-chat-messages').replaceChildren();
  document.getElementById('console-chat-status').textContent = '已开始新对话，原后端记录保留。';
});
document.getElementById('console-chat-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
    event.preventDefault();
    if (!consoleChatBusy) document.getElementById('console-chat-form').requestSubmit();
  }
});
document.getElementById('console-chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = document.getElementById('console-chat-input');
  const text = input.value.trim();
  if (!text || consoleChatBusy) return;
  consoleChatBusy = true;
  input.readOnly = true;
  document.getElementById('console-chat-send').disabled = true;
  document.getElementById('console-chat-new').disabled = true;
  const status = document.getElementById('console-chat-status');
  status.textContent = `等待${characterName}回复，请勿重复发送…`;
  appendConsoleMessage('你', text);
  try {
    // 复用桌面渠道原入口，不创建第二个循环，也不请求 QQ 投递。
    // 网络失败不自动重发：原请求可能已执行，避免产生重复调用。
    const response = await fetch('/api/pet/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, conversation_id: consoleConversationId})
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.reason || 'request_failed');
    appendConsoleMessage(characterName, result.reply || '（后端未返回正文）');
    input.value = '';
    status.textContent = `本机回复 · ${result.agent_status || '状态未知'} · ${result.trace_id || ''}`;
  } catch (error) {
    status.textContent = '本轮未取得有效回复：' + error.message + '。输入已保留；未自动重发。';
  } finally {
    consoleChatBusy = false;
    input.readOnly = false;
    document.getElementById('console-chat-send').disabled = false;
    document.getElementById('console-chat-new').disabled = false;
    input.focus();
  }
});

function detailKey(detail, index) {
  const record = detail.closest('[data-record]');
  return record ? `${record.dataset.record}:${Array.from(record.querySelectorAll('details')).indexOf(detail)}` : String(index);
}

function setHTML(target, html) {
  // 未变化的片段保持原节点；展开状态绑定记录 ID，避免新记录插入后详情串位。
  if (renderedHTML.get(target) === html || target.closest('[data-busy]') || target.querySelector('[data-busy]')) return;
  const active = document.activeElement;
  if (target.contains(active) && active.matches('input,textarea,select')) return;
  const opened = new Set(Array.from(target.querySelectorAll('details')).flatMap((item, index) => item.open ? [detailKey(item, index)] : []));
  const scrolls = Array.from(target.querySelectorAll('pre,.history-scroll')).map((item) => [item.scrollLeft, item.scrollTop]);
  target.innerHTML = html;
  renderedHTML.set(target, html);
  target.querySelectorAll('details').forEach((item, index) => { item.open = opened.has(detailKey(item, index)); });
  target.querySelectorAll('pre,.history-scroll').forEach((item, index) => {
    if (scrolls[index]) { item.scrollLeft = scrolls[index][0]; item.scrollTop = scrolls[index][1]; }
  });
}

function matchesQuery(item, inputId) {
  const query = document.getElementById(inputId).value.trim().toLowerCase();
  return !query || JSON.stringify(item).toLowerCase().includes(query);
}

function paginateList(id) {
  const target = document.getElementById(id);
  const records = Array.from(target.querySelectorAll('[data-record]'));
  const limit = listLimits.get(id) || 8;
  records.forEach((item, index) => { item.hidden = index >= limit; });
  let footer = document.getElementById(`${id}-pagination`);
  if (!footer) {
    footer = document.createElement('div');
    footer.id = `${id}-pagination`;
    footer.className = 'list-footer';
    target.after(footer);
  }
  footer.hidden = !records.length;
  setHTML(footer, `<span>显示 ${Math.min(limit, records.length)} / ${records.length} 条</span>${records.length > 8 ? `<button type="button" data-list="${id}" data-expand="${limit < records.length}">${limit < records.length ? '展开更多' : '收起列表'}</button>` : ''}`);
}

function renderOverview(snapshot) {
  renderCharacter(snapshot.character);
  const agent = snapshot.agent || {};
  const controls = agent.controls || {};
  const config = agent.diagnostics?.config || {};
  const model = agent.live_models?.chat || config.chat_model_name || '未装配';
  document.getElementById('console-chat-model').textContent = `${characterName}运行模型：${model} · 使用当前实例的会话与记忆链路`;
  const connection = snapshot.qq_connection;
  const fresh = connection && Date.now() - Date.parse(connection.observed_at) < 30000;
  const labels = {online: 'QQ 已登录，账号匹配', offline: 'QQ 未在线', account_mismatch: 'QQ 账号不匹配', unavailable: 'QQ 连接未确认', unconfigured: 'QQ 尚未配置'};
  document.getElementById('instance-connection').textContent = `${fresh ? (labels[connection.status] || 'QQ 状态未知') : 'QQ 连接状态待核对'} · 消息接入：${snapshot.napcat?.dispatcher_attached ? '已装配' : '未装配'} · QQ 发送：${config.send_enabled === undefined ? '未知' : config.send_enabled ? '开启' : '关闭'} · 已接收事件：${connection?.received_events ?? '未知'}`;
  const metrics = agent.runtime?.metrics || {};
  const stats = agent.memory?.stats;
  const tasks = snapshot.agent_tasks?.tasks;
  const running = Array.isArray(tasks) ? tasks.filter((item) => !['completed','failed','canceled','cancelled'].includes(item.status)).length : null;
  setHTML(document.getElementById('runtime-summary'), `<strong>${snapshot.health?.status === 'ok' ? '■ 服务运行中' : '■ 服务状态异常'}</strong><span>模型 <code>${esc(model)}</code></span><span>回复 ${controls.responses_enabled === undefined ? '未知' : controls.responses_enabled ? '开启' : '暂停'}</span><span>发送 ${config.send_enabled === undefined ? '未知' : config.send_enabled ? '开启' : '关闭'}</span>`);
  setHTML(document.getElementById('overview-metrics'), [
    metricCard('聊天调用 / 当前运行', agent.runtime ? formatNumber(metricNumber(metrics, 'chat_json_calls') + metricNumber(metrics, 'chat_stream_calls')) : '—'),
    metricCard('Token / 当前运行', agent.runtime ? formatNumber(metricNumber(metrics, 'total_tokens_chat')) : '—'),
    metricCard('已存记忆回合', stats ? formatNumber(stats.turns) : '—'),
    metricCard('活跃任务 / 最近 20 项', running === null ? '—' : formatNumber(running))
  ].join(''));
  document.getElementById('agent-control-summary').textContent = `回复：${controls.responses_enabled === undefined ? '未知' : controls.responses_enabled ? '开启' : '暂停'} · 任务工具：${controls.agent_enabled === undefined ? '未知' : controls.agent_enabled ? '开启' : '关闭'} · 权限：${controls.permission_mode_label || controls.permission_mode || '未知'}`;
  document.getElementById('pet-control-summary').textContent = snapshot.pet?.available ? `桌宠：${snapshot.pet.status === 'online' ? '运行中' : '未启动 / 离线'}` : '桌宠模块未就绪。';
}

function notifyOperation(message, failed = false) {
  const dialog = document.querySelector('dialog[open] .dialog-result');
  const target = dialog || document.getElementById('operation-result');
  target.textContent = message;
  target.classList.toggle('error', failed);
  target.hidden = false;
}

function lockControls(button) {
  const root = button.closest('.provider, #agent-actions, #pet-actions') || button.parentElement;
  const controls = Array.from(root.querySelectorAll('button,input,select')).map((item) => [item, item.disabled]);
  root.dataset.busy = 'true';
  controls.forEach(([item]) => { item.disabled = true; });
  mutationsInFlight += 1;
  return () => {
    delete root.dataset.busy;
    controls.forEach(([item, disabled]) => { item.disabled = disabled; });
    mutationsInFlight -= 1;
  };
}

const esc = (value) => String(value ?? '—').replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[char]));
const row = (key, value) => `<div>${esc(key)}</div><div>${value}</div>`;
const text = (value) => `<code>${esc(value)}</code>`;
const yesNo = (value) => value
  ? '<span class="value-ok">是</span>'
  : '<span class="value-warn">否</span>';
const metricNumber = (metrics, key) => Number(metrics?.[key] || 0);
const formatNumber = (value) => Number(value || 0).toLocaleString('zh-CN');
const metricCard = (label, value, className = '') => `<div class="metric"><span class="metric-label">${esc(label)}</span><strong class="metric-value ${className}">${value}</strong></div>`;

function renderMetrics(snapshot) {
  const metrics = snapshot.agent?.runtime?.metrics || {};
  const cacheRead = metricNumber(metrics, 'cache_read_tokens_chat');
  const cacheCreated = metricNumber(metrics, 'cache_creation_tokens_chat');
  const cacheObserved = cacheRead + cacheCreated;
  const cacheRate = cacheObserved > 0 ? `${((cacheRead / cacheObserved) * 100).toFixed(1)}%` : '暂无供应商字段';
  const lastError = snapshot.agent?.runtime?.last_error || {};
  const errorValue = lastError.type ? `${lastError.type}: ${lastError.phase || 'runtime'}` : '无';
  setHTML(document.getElementById('metrics'), [
    metricCard('聊天调用', formatNumber(metricNumber(metrics, 'chat_json_calls') + metricNumber(metrics, 'chat_stream_calls'))),
    metricCard('输入 Token', formatNumber(metricNumber(metrics, 'input_tokens_chat'))),
    metricCard('输出 Token', formatNumber(metricNumber(metrics, 'output_tokens_chat'))),
    metricCard('总 Token', formatNumber(metricNumber(metrics, 'total_tokens_chat'))),
    metricCard('缓存读入', formatNumber(cacheRead)),
    metricCard('可观测命中率', cacheRate, cacheObserved > 0 ? 'value-ok' : 'value-warn'),
    metricCard('工具决策', formatNumber(metricNumber(metrics, 'native_tool_decision_sent'))),
    metricCard('最近错误', esc(errorValue), lastError.type ? 'value-bad' : 'value-ok')
  ].join(''));
}

function renderRuntimeConfig(snapshot) {
  const target = document.getElementById('runtime-config');
  const diagnostics = snapshot.agent?.diagnostics || {};
  const config = diagnostics.config || {};
  const flow = diagnostics.chat_flow || {};
  const extension = diagnostics.runtime_extension || {};
  if (!Object.keys(config).length) {
    setHTML(target, '<div class="empty">当前没有可读取的运行配置。</div>');
    return;
  }
  setHTML(target, [
    row('协议', text(config.chat_api_protocol)),
    row('模型', text(config.chat_model_name)),
    row('Base URL', text(config.chat_base_url)),
    row('密钥', text(config.chat_api_key)),
    row('记忆', yesNo(config.memory_enabled)),
    row('模型判定记忆', yesNo(config.memory_judge_enabled)),
    row('历史轮数', text(config.history_turns)),
    row('上下文条数', text(config.recent_context_messages)),
    row('Agent 最大轮数', text(config.max_rounds)),
    row('回复宿主', yesNo(config.reply_host_enabled)),
    row('长回复阈值', text(config.reply_long_trigger_chars)),
    row('最大分段字数', text(config.reply_max_segment_chars)),
    row('聊天层', flow.enabled ? `<span class="value-ok">启用</span>（${esc(flow.active_states ?? 0)} 个状态）` : '<span class="value-warn">关闭</span>'),
    row('运行扩展', extension.configured ? text(extension.module || '已配置') : '<span class="muted">无</span>')
  ].join(''));
}

function renderRecentMetrics(days) {
  const target = document.getElementById('metrics-recent');
  if (!Array.isArray(days) || !days.length) {
    setHTML(target, '<div class="empty">当前没有持久化的日指标；运行时内存指标仍显示在上方。</div>');
    return;
  }
  const keys = ['chat_json_calls', 'chat_stream_calls', 'total_tokens_chat', 'cache_read_tokens_chat', 'native_tool_call_extracted'];
  const labels = ['聊天 JSON', '聊天流式', '聊天 Token', '缓存读入', '工具调用'];
  const header = ['日期', ...labels].map((label) => `<th>${esc(label)}</th>`).join('');
  const rows = days.slice().reverse().map((day) => {
    const values = keys.map((key) => `<td>${formatNumber(metricNumber(day, key))}</td>`).join('');
    return `<tr><td>${esc(day.day || day.date || '—')}</td>${values}</tr>`;
  }).join('');
  setHTML(target, `<div class="history-scroll"><table class="history-table"><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>`);
}

function traceStatusClass(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'completed' || value === 'command' || value === 'finished') return 'trace-status-ok';
  if (value === 'failed' || value === 'sender_error' || value === 'partial') return 'trace-status-bad';
  return 'trace-status-warn';
}

function traceDuration(value) {
  if (value === null || value === undefined) return '—';
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} ms` : '—';
}

function traceDetails(value) {
  try {
    return JSON.stringify(value || {}, null, 0);
  } catch (_) {
    return '{}';
  }
}

function phasePerformanceHTML(performance) {
  if (!performance?.available) return '<p class="muted">阶段耗时尚未采集</p>';
  const rows = (performance.phases || []).map(row => `<tr><td>${esc(row.label)}</td><td>${traceDuration(row.totalMs)}</td><td>${esc(row.measured)}/${esc(row.samples)}</td><td>${traceDuration(row.p50Ms)} / ${traceDuration(row.p95Ms)}</td><td>${esc(row.requests)} 次 · ${traceDuration(row.requestMs)}${row.incompleteRequestTiming?"（部分）":""}</td><td>${esc(row.knownInputTokens ?? '未知')} / ${esc(row.knownOutputTokens ?? '未知')}${row.incompleteUsage ? '（部分）' : ''}</td><td>异常 ${esc(row.failed)} · 请求失败 ${esc(row.requestFailed)} · 未结束 ${esc(row.unsettled + row.requestUnsettled)}</td></tr>`).join('');
  const tasks = performance.delegatedTaskIds?.length ? `<p>关联委托任务：${performance.delegatedTaskIds.map(esc).join('、')}（远端细分在 Angelos 查看）</p>` : '';
  return `<details class="trace-stages"><summary>阶段耗时与模型请求</summary><p class="muted">阶段包含子阶段，不可相加。p50/p95 按已结束调用采样；请求耗时之和不等于用户等待时间。正常返回不代表业务完成。</p>${tasks}${performance.truncated ? '<p>采集不完整或记录已截断，仅显示保留样本。</p>' : ''}<div style="overflow-x:auto"><table class="phase-table"><thead><tr><th>阶段</th><th>累计耗时</th><th>已计时/样本</th><th>单次 p50/p95</th><th>已记录模型请求</th><th>输入/输出 token</th><th>诊断</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
}

function renderTraces(traces) {
  const target = document.getElementById('traces');
  traces = (Array.isArray(traces) ? traces : []).filter((item) => (traceChannel === 'all' || item.conversation_type === traceChannel) && matchesQuery(item, 'trace-search'));
  if (!Array.isArray(traces) || !traces.length) {
    setHTML(target, '<div class="empty">暂无符合条件的处理链；新回合产生后自动更新。</div>');
    return;
  }
  setHTML(target, `<div class="trace-list">${traces.slice(0, 30).map((trace) => {
    const status = String(trace.status || 'finished');
    const stages = Array.isArray(trace.stages) ? trace.stages : [];
    const isDesktop = trace.conversation_type === 'desktop';
    const isGroup = trace.conversation_type === 'group';
    const channelBadge = isDesktop
      ? '<span class="badge-desktop">桌面桌宠</span>'
      : (isGroup ? '<span class="badge-group">QQ 群聊</span>' : '<span class="badge-private">QQ 私聊</span>');
    const conversation = [trace.conversation_type, trace.conversation_id].filter(Boolean).join(' / ') || '—';
    const usageStage = stages.find((s) => s && s.name === 'llm_usage');
    const memoryStage = stages.find((s) => s && s.name === 'memory');
    const tokenInfo = usageStage
      ? `Token: ${usageStage.total_tokens || ((usageStage.input_tokens || 0) + (usageStage.output_tokens || 0))} (入 ${usageStage.input_tokens || 0} / 出 ${usageStage.output_tokens || 0})`
      : '';
    const memoryInfo = memoryStage
      ? `记忆: ${memoryStage.history_mode || '无'} (${memoryStage.retrieved_count || 0} 条 / ${memoryStage.injected_chars || 0} 字)`
      : '';
    const stageHtml = stages.length
      ? `<details class="trace-stages"><summary>阶段 ${stages.length} 个</summary>${stages.map((stage) => {
        const details = { ...stage };
        delete details.name;
        delete details.elapsed_ms;
        return `<div class="trace-stage"><span class="trace-stage-name">${esc(stage.name || 'stage')}</span><span class="trace-stage-time">${traceDuration(stage.elapsed_ms)}</span><span class="trace-stage-details">${esc(traceDetails(details))}</span></div>`;
      }).join('')}</details>`
      : '<div class="trace-stages muted">没有阶段记录</div>';
    const reason = trace.reason ? `<div class="trace-reason">原因：${esc(trace.reason)}</div>` : '';
    return `<article class="trace" data-record="${esc(trace.trace_id)}">
      <div class="trace-head"><strong title="${esc(trace.trace_id)}">${esc(String(trace.trace_id || '').slice(0, 12) || '—')}</strong>${channelBadge}<span class="trace-status ${traceStatusClass(status)}">${esc(status)}</span></div>
      <div class="trace-meta">
        <span>${esc(conversation)}</span>
        <span>耗时 ${traceDuration(trace.elapsed_ms)}</span>
        <span>输入 ${esc(trace.input_chars ?? 0)} 字</span>
        <span>输出 ${esc(trace.reply_chars ?? 0)} 字</span>
        ${tokenInfo ? `<span>${esc(tokenInfo)}</span>` : ''}
        ${memoryInfo ? `<span>${esc(memoryInfo)}</span>` : ''}
        <span>Agent ${esc(trace.agent_status || '—')}</span>
        <span>投递 ${esc(trace.delivery_status || '—')}</span>
      </div>${reason}${phasePerformanceHTML(trace.performance)}${stageHtml}
    </article>`;
  }).join('')}</div>`);
}

function renderAttention(payload) {
  const target = document.getElementById('attention-traces');
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const summary = payload?.summary || {};
  if (!payload?.available || !items.length) {
    setHTML(target, '<div class="empty">暂无注意力记录；事件只保留有限元数据，不保存消息原文。</div>');
    return;
  }
  const headline = `最近 ${esc(summary.window_items ?? items.length)} 条：入站接受 ${esc(summary.admitted ?? 0)} / 拦截 ${esc(summary.filtered ?? 0)}；注意力调度 ${esc(summary.scheduled ?? 0)} / 未调度 ${esc(summary.not_scheduled ?? 0)}`;
  setHTML(target, `<div class="memory-summary">${headline}</div><div class="trace-list">${items.slice(0, 80).map((item) => {
    const admission = item.admission || {};
    const route = item.route || {};
    const attention = item.attention || {};
    const topic = item.topic_state || {};
    const turn = item.turn || {};
    const usage = turn.llm_usage || {};
    const conversation = [item.conversation_type, item.conversation_id].filter(Boolean).join(' / ') || '—';
    const time = Number(item.observed_at) > 0 ? new Date(Number(item.observed_at) * 1000).toLocaleTimeString() : '—';
    const tokenText = usage.available === false
      ? '用量未关联/供应方未返回'
      : `调用 ${usage.calls ?? 0} 次 · 输入 ${usage.input_tokens ?? 0} · 输出 ${usage.output_tokens ?? 0} · 总计 ${usage.total_tokens ?? 0} Token`;
    const state = attention.should_schedule ? '已调度' : '未调度';
    const stateClass = attention.should_schedule ? 'trace-status-ok' : 'trace-status-warn';
    const details = {
      admission: admission.reason || '—',
      route: route.reason || '—',
      attention: attention.reason || '—',
      immediate: Boolean(attention.immediate),
      text_chars: item.text_chars ?? 0,
      has_media: Boolean(item.has_media),
      topic_reply_count: topic.reply_message_count ?? 0,
      topic_context_count: topic.group_context_count ?? 0,
      topic_context_reason: topic.group_context_reason || '',
      outcome: turn.status || 'not_started',
      sent: Boolean(turn.sent),
      delivery: turn.delivery_status || '—',
      turn_reason: turn.reason || '',
      usage
    };
    return `<article class="trace" data-record="${esc(item.trace_id || `${item.conversation_id}:${item.message_id}:${item.observed_at}`)}">
      <div class="trace-head"><strong>${esc(time)} · ${esc(item.message_id || item.trace_id || '—')}</strong><span class="trace-status ${stateClass}">${state}</span></div>
      <div class="trace-meta"><span>${esc(conversation)}</span><span>发送者 ${esc(item.sender_id || '—')}</span><span>投递 ${esc(turn.sent ? '已发送' : turn.status === 'not_scheduled' ? '未调用' : turn.status || '待确认')}</span><span>${esc(tokenText)}</span></div>
      <details class="trace-stages"><summary>判定详情</summary><div class="trace-stage-details">${esc(traceDetails(details))}</div></details>
    </article>`;
  }).join('')}</div>`);
}

function renderMemoryRouting(payload) {
  const target = document.getElementById('memory-routing');
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const summary = payload?.summary || {};
  if (!payload?.available) {
    setHTML(target, '<div class="empty">记忆服务未装配或审计暂不可用。</div>');
    return;
  }
  if (!items.length) {
    setHTML(target, '<div class="empty">暂无记忆路由记录。此视图只展示决策元数据，不显示检索内容。</div>');
    return;
  }
  const counts = summary.mode_counts || {};
  const headline = `最近 ${esc(summary.window_items ?? items.length)} 轮：none ${esc(counts.none ?? 0)} · recent ${esc(counts.recent ?? 0)} · retrieve ${esc(counts.retrieve ?? 0)} · summary ${esc(counts.summary ?? 0)}；模型判定 ${esc(summary.model_judged ?? 0)} 轮 · 找到记录 ${esc(summary.with_results ?? 0)} 轮`;
  const modeLabels = { none: '不检索', recent: '近期承接', retrieve: '主题检索', summary: '日期摘要' };
  setHTML(target, `<div class="memory-summary">${headline}</div><div class="trace-list">${items.slice(0, 80).map((item) => {
    const mode = String(item.history_mode || 'none');
    const stamp = Number(item.created_at) > 0 ? new Date(Number(item.created_at) * 1000).toLocaleString() : '—';
    const details = {
      scope: item.scope || '—',
      reason: item.reason || '—',
      local_signals: item.local_signals || [],
      used_model: Boolean(item.used_model),
      confidence: item.confidence ?? 0,
      need_retrieval: Boolean(item.need_retrieval),
      retrieved_count: item.retrieved_count ?? 0,
      injected_chars: item.injected_chars ?? 0
    };
    const stateClass = mode === 'none' ? 'trace-status-warn' : 'trace-status-ok';
    return `<article class="trace" data-record="${esc(item.audit_id || `${item.scope}:${item.created_at}`)}">
      <div class="trace-head"><strong>${esc(stamp)} · ${esc(item.conversation_id || '—')}</strong><span class="trace-status ${stateClass}">${esc(modeLabels[mode] || mode)}</span></div>
      <div class="trace-meta"><span>${esc(item.used_model ? '记忆模型判定' : '本地判定')}</span><span>检索 ${esc(item.retrieved_count ?? 0)} 条</span><span>注入 ${esc(item.injected_chars ?? 0)} 字</span><span>置信度 ${esc(item.confidence ?? 0)}</span></div>
      <details class="trace-stages"><summary>判定详情（不含 query 和聊天原文）</summary><div class="trace-stage-details">${esc(traceDetails(details))}</div></details>
    </article>`;
  }).join('')}</div>`);
}

function renderQQDiagnostics(payload) {
  const target = document.getElementById('qq-diagnostics');
  const checks = Array.isArray(payload?.checks) ? payload.checks : [];
  if (!checks.length) {
    setHTML(target, '<div class="empty">QQ 诊断接口未装配或暂时不可用。</div>');
    return;
  }
  const summary = payload.ok
    ? '<span class="value-ok">通过</span>'
    : '<span class="value-warn">有项目需要注意</span>';
  setHTML(target, `<div class="memory-summary">总体状态：${summary}</div><div class="check-list">${checks.map((check) => `
    <div class="check-row"><strong>${check.ok ? '<span class="value-ok">✓</span>' : '<span class="value-bad">✕</span>'} ${esc(check.name || 'check')}</strong><span class="check-detail">${esc(check.detail || '')}</span></div>
  `).join('')}</div>`);
}

function renderCapabilities(payload) {
  const target = document.getElementById('capabilities');
  const rows = Array.isArray(payload?.capabilities) ? payload.capabilities : [];
  if (!rows.length) {
    setHTML(target, '<div class="empty">当前没有可展示的工具，或能力目录尚未装配。</div>');
    return;
  }
  setHTML(target, `<div class="capability-list">${rows.map((item) => {
    const risk = String(item.risk || 'unknown').toLowerCase();
    const riskClass = ['low', 'medium', 'high', 'critical'].includes(risk) ? risk : '';
    const permissionClass = ({ read: '只读', change: '会改变状态', sensitive: '敏感数据' })[String(item.permission_class || '')] || '未标记（只读模式下禁用）';
    const schema = item.input_schema && Object.keys(item.input_schema).length
      ? `<details class="trace-stages"><summary>输入结构</summary><pre>${esc(traceDetails(item.input_schema))}</pre></details>`
      : '';
    return `<article class="capability" data-record="${esc(item.name)}">
      <div class="capability-head"><strong>${esc(item.name || 'unnamed')}</strong><span class="risk-${riskClass}">${esc(risk)}</span></div>
      <div class="capability-meta"><span>当前档位授权：${item.requires_approval === true ? '需要' : item.requires_approval === false ? '不需要' : '未知'}</span><span>权限属性：${esc(permissionClass)}</span><span>允许：${item.allowlisted === false ? '否' : '是'}</span></div>
      <div class="capability-description">${esc(item.description || '没有描述')}</div>${schema}
    </article>`;
  }).join('')}</div>`);
}

function renderTasks(payload) {
  const target = document.getElementById('agent-tasks');
  const rows = (Array.isArray(payload?.tasks) ? payload.tasks : []).filter((task) => {
    const filter = document.getElementById('task-filter').value;
    if (filter === 'all') return true;
    if (filter === 'failed') return ['failed', 'canceled', 'cancelled'].includes(task.status);
    if (filter === 'completed') return task.status === 'completed';
    return !['completed', 'failed', 'canceled', 'cancelled'].includes(task.status);
  });
  if (!rows.length) {
    setHTML(target, '<div class="empty">当前没有符合条件的任务，或任务协调器尚未装配。</div>');
    return;
  }
  setHTML(target, `<div class="task-list">${rows.slice(0, 20).map((task) => {
    const status = String(task.status || 'unknown');
    const statusClass = status === 'completed' ? 'value-ok' : ['failed', 'canceled'].includes(status) ? 'value-bad' : 'value-warn';
    return `<article class="task" data-record="${esc(task.task_id)}"><div class="task-head"><strong>${esc(task.task_id || '—')}</strong><span class="${statusClass}">${esc(status)}</span></div><div class="task-meta">${esc(task.goal || '未填写任务目标')} · 第 ${esc(task.attempt ?? 1)} 次</div>${task.error ? `<div class="error-mark">${esc(task.error)}</div>` : ''}<div class="task-actions"><button type="button" data-task-id="${esc(task.task_id || '')}">查看详情</button></div></article>`;
  }).join('')}</div>`);
}

function renderTaskDetail(payload) {
  const target = document.getElementById('task-detail');
  if (!target) return;
  if (!payload || payload.ok === false || !payload.task) {
    setHTML(target, `<div class="task-detail error-mark">读取任务失败：${esc(payload?.reason || 'task_not_found')}</div>`);
    return;
  }
  const task = payload.task;
  const budget = task.metadata?.budget;
  const budgetDetails = budget ? `<div>预算档位 / 阶段</div><div>${esc(budget.profile || 'legacy')} / ${esc(budget.stage || '—')}</div>
    <div>累计模型请求</div><div>${esc(budget.requests ?? 0)} / ${esc(budget.max_requests ?? '—')}</div>
    <div>累计 Token 记账</div><div>${esc(budget.tokens ?? 0)} / ${esc(budget.max_tokens ?? '—')}</div>
    <div>累计工具调用</div><div>${esc(budget.tools ?? 0)} / ${esc(budget.max_tools ?? '—')}</div>
    <div>有效执行秒数</div><div>${esc(Math.round(budget.active_seconds || 0))} / ${esc(budget.max_active_seconds ?? '—')}</div>` : '';
  setHTML(target, `<div class="task-detail"><div class="task-detail-head"><h3>任务详情</h3><button type="button" data-close-task>收起详情</button></div><div class="kv">
    <div>任务号</div><div>${text(task.task_id)}</div>
    <div>状态</div><div>${esc(task.status || '—')}</div>
    <div>目标</div><div>${esc(task.goal || '—')}</div>
    <div>结果</div><div>${esc(task.result_summary || '—')}</div>
    <div>会话</div><div>${esc(task.session_id || '—')}</div>
    ${budgetDetails}
  </div>${task.metadata && Object.keys(task.metadata).length ? `<details class="trace-stages"><summary>元数据</summary><pre>${esc(traceDetails(task.metadata))}</pre></details>` : ''}</div>`);
}

function renderApproval(payload) {
  const target = document.getElementById('approval-result');
  if (!payload || payload.ok === false) {
    setHTML(target, `<div class="error-mark">读取失败：${esc(payload?.reason || 'approval_endpoint_unavailable')}</div>`);
    return;
  }
  if (!payload.pending || !payload.approval) {
    setHTML(target, '<div class="empty">当前会话没有待授权工具调用。</div>');
    return;
  }
  const approval = payload.approval;
  const tools = Array.isArray(approval.tools) && approval.tools.length
    ? approval.tools
    : [{ tool: approval.tool, risk: approval.risk, arguments: approval.arguments }];
  setHTML(target, `<div class="approval-box">
    <div class="memory-summary">请求：${esc(approval.request_id || '—')} · 风险：${esc(approval.risk || 'unknown')} · 任务：${esc(approval.task_id || '—')}</div>
    <div>${esc(approval.question || '等待用户确认')}</div>
    <div class="approval-tools">${tools.map((item) => `<div class="approval-tool"><strong>${esc(item.tool || 'unnamed')}</strong> · ${esc(item.risk || 'unknown')}<pre>${esc(traceDetails(item.arguments))}</pre></div>`).join('')}</div>
  </div>`);
}

function renderMemoryResult(payload) {
  const target = document.getElementById('memory-result');
  if (!payload || payload.ok === false) {
    setHTML(target, `<div class="error-mark">读取失败：${esc(payload?.reason || 'memory_endpoint_unavailable')}</div>`);
    return;
  }
  const rows = Array.isArray(payload.facts) ? payload.facts : Array.isArray(payload.records) ? payload.records : [];
  const mode = payload.mode || 'facts';
  const summary = `作用域：${payload.scope || '—'} · 模式：${mode} · ${rows.length} 条`;
  if (!rows.length) {
    setHTML(target, `<div class="memory-summary">${esc(summary)}</div><div class="empty">没有找到记录。</div>`);
    return;
  }
  setHTML(target, `<div class="memory-summary">${esc(summary)}</div><div class="memory-list">${rows.map((record) => {
    const created = Number(record.created_at);
    const when = Number.isFinite(created) && created > 0 ? new Date(created * 1000).toLocaleString() : '时间未知';
    return `<article class="memory-item"><strong>${esc(record.layer || 'record')} · ${esc(when)}</strong><div class="memory-meta">${esc(record.source_id || record.record_id || '')}</div><pre>${esc(record.text || '')}</pre></article>`;
  }).join('')}</div>`);
}

function projectionRows(features) {
  const projection = features?.ok === true ? features.context_projection : null;
  if (!projection) return row('上下文投影', '<span class="value-warn">状态接口不可用或后端版本较旧</span>');
  const enabled = projection.configured === true && projection.initialized === true;
  const index = projection.index_status || {};
  const diagnostic = projection.last_diagnostics || {};
  const semantic = diagnostic.semantic || {};
  const indexLabels = {ready: '就绪', indexing: '后台更新中', not_started: '尚未开始',
    fallback: '索引暂不可用，使用回退检索', busy: '维护中', cleanup_pending: '等待清理'};
  const semanticLabels = {not_needed: '无需调用', clarification_needed: '需澄清人物，未调用',
    completed: '已完成', disabled: '未启用', cache_hit: '命中缓存', coalesced: '复用并发结果',
    timeout: '超时，已回退', failed: '失败，已回退', budget_exhausted: '预算耗尽'};
  const hasDiagnostic = Object.keys(diagnostic).length > 0;
  return [
    row('上下文投影', enabled ? '<span class="value-ok">已启用并初始化</span>'
      : projection.configured ? '<span class="value-bad">已配置，但未初始化</span>' : '<span class="value-warn">未启用</span>'),
    row('按需语义处理', esc(enabled && projection.semantic_configured ? '已启用' : '未启用')),
    row('历史索引', esc(indexLabels[index.state] || index.state || (enabled ? '尚无状态' : '未启用'))
      + (index.error_type ? ` · ${esc(index.error_type)}` : '')),
    row('最近投影', hasDiagnostic
      ? esc(`${diagnostic.source_count ?? '—'} 条来源 / ${diagnostic.injected_chars ?? '—'} 字 / ${diagnostic.elapsed_ms ?? '—'} ms`)
      : '尚无诊断记录'),
    row('最近语义处理', esc(semanticLabels[semantic.status] || semantic.status || '尚无诊断记录')
      + (semantic.error_type ? ` · ${esc(semantic.error_type)}` : '')),
    row('证据缺口', hasDiagnostic ? text((diagnostic.gap_reasons || []).join('、') || '无已报告缺口') : '尚无诊断记录'),
    row('投影版本', text(projection.policy_version))
  ].join('');
}

function render(snapshot) {
  const health = snapshot.health || {};
  const agent = snapshot.agent || {};
  const diagnostics = (agent.diagnostics || {}).config || {};
  const memory = (agent.memory || {}).stats || {};
  const napcat = snapshot.napcat || {};
  const bridge = agent.bridge || {};
  const runtimeHealth = bridge.runtime_health || {};
  const liveModels = agent.live_models || {};
  const controls = agent.controls || {};

  setHTML(document.getElementById('service'), [
    row('状态', health.status === 'ok' ? '<span class="value-ok">运行中</span>' : '<span class="value-bad">异常</span>'),
    row('项目', text(health.project)),
    row('PID', text((snapshot.process || {}).pid)),
    row('端口', text(health.port)),
    row('Python', text(health.python))
  ].join(''));

  setHTML(document.getElementById('agent'), [
    row('状态', agent.enabled ? '<span class="value-ok">已装配</span>' : '<span class="value-warn">未装配</span>'),
    row('进程', text(`${agent.process?.service || 'backend'} / PID ${agent.process?.pid || snapshot.process?.pid || '—'}（${agent.process?.mode || '—'}）`)),
    row('运行状态', controls.responses_enabled === false ? '<span class="value-warn">已暂停</span>' : '<span class="value-ok">可接收</span>'),
    row('任务工具', controls.agent_enabled === false ? '<span class="value-warn">已关闭（日常聊天仍可回复）</span>' : '<span class="value-ok">已开启，按需交给 Agent</span>'),
    row('工具权限', text(controls.permission_mode_label || controls.permission_mode || '传统')),
    row('模型', text(liveModels.chat || diagnostics.chat_model_name)),
    row('发送', yesNo(diagnostics.send_enabled)),
    row('人设', text((agent.diagnostics || {}).persona?.path)),
    row('工具', text((bridge.tool_names || []).join(', ') || '无')),
    row('原生工具', runtimeHealth.native_tools_supported === true ? '<span class="value-ok">支持</span>' : runtimeHealth.native_tools_supported === false ? '<span class="value-bad">不支持</span>' : '<span class="value-warn">未知</span>'),
    row('轮数上限', text(bridge.max_rounds))
  ].join(''));

  setHTML(document.getElementById('memory'), [
    row('回合', text(memory.turns || 0)),
    row('活动事件', text(memory.active_events || 0)),
    row('长期事实', text(memory.active_facts || 0)),
    row('遗忘记录', text(memory.forgotten || 0)),
    row('记忆路由审计', text(memory.routing_audits || 0)),
    row('数据库', text((agent.memory || {}).path)),
    projectionRows(snapshot.memory_features)
  ].join(''));

  setHTML(document.getElementById('napcat'), [
    row('已配置', yesNo(napcat.configured)),
    row('回合调度', yesNo(napcat.dispatcher_attached)),
    row('传输', text(napcat.transport)),
    row('Webhook', text((napcat.diagnostics || {}).webhook_path))
  ].join(''));

  renderOverview(snapshot);
  renderAgentActions(controls);
  renderDesktopPet(snapshot.pet || {});
  renderMetrics(snapshot);
  renderRuntimeConfig(snapshot);
  renderRecentMetrics(snapshot.metrics_recent || []);
  renderAttention(snapshot.attention || {});
  renderMemoryRouting(snapshot.memory_routing || {});
  renderTraces(snapshot.traces || []);
  renderQQDiagnostics(snapshot.qq_diagnostics || {});
  renderCapabilities(snapshot.capabilities || {});
  renderTasks(snapshot.agent_tasks || {});
  renderModelServices(snapshot.model_services || {});

  const logs = document.getElementById('logs');
  const items = snapshot.logs || [];
  setHTML(logs, items.length
    ? items.map((item) => `<div class="log-title"><span>${text(item.name)}</span><span>${esc(item.updated_at)}</span></div><pre>${esc((item.lines || []).join('\n'))}</pre>`).join('<hr style="border:0;border-top:1px solid var(--line);margin:14px 0">')
    : '<div class="empty">当前日志目录没有可读日志</div>');
  document.getElementById('updated').textContent = `最近刷新：${new Date(snapshot.observed_at || Date.now()).toLocaleString()} · 每 3 秒读取，设置编辑期间暂停`;
  ['traces', 'attention-traces', 'agent-tasks', 'memory-routing', 'capabilities'].forEach(paginateList);
  updateNavigation();
}

function providerKey(id) {
  return `provider_${String(id || '').replace(/[^a-zA-Z0-9_-]/g, '_')}`;
}

function renderAgentActions(controls) {
  const target = document.getElementById('agent-actions');
  if (!controls || controls.agent_enabled === undefined) {
    setHTML(target, '');
    return;
  }
  const modes = [
    ['traditional', '传统（风险分级）'],
    ['readonly', '只读'],
    ['confirm', '工具调用均确认'],
    ['full', '完全访问（全电脑自动读写）'],
    ['ultra', 'Ultra（本机工具层全开）']
  ];
  const actualMode = modes.some(([mode]) => mode === controls.permission_mode) ? controls.permission_mode : 'traditional';
  const currentMode = permissionDraft ?? actualMode;
  setHTML(target, [
    '<button data-agent-action="start">恢复回复</button>',
    '<button data-agent-action="stop">暂停回复</button>',
    '<button data-agent-action="on">启用任务工具</button>',
    '<button data-agent-action="off">关闭任务工具</button>',
    `<div class="permission-control"><label for="agent-permission-mode">Agent 权限档位</label><select id="agent-permission-mode">${modes.map(([mode, label]) => `<option value="${mode}"${mode === currentMode ? ' selected' : ''}>${label}</option>`).join('')}</select><button data-agent-action="permission">应用档位</button><span class="permission-warning">完全访问/Ultra 会自动执行高影响操作；Ultra 仅开放已注册私用工具层能触及的能力，仍受操作系统当前账户权限限制。</span></div>`
  ].join(''));
}

function renderDesktopPet(pet) {
  const target = document.getElementById('desktop-pet');
  const actions = document.getElementById('pet-actions');
  if (!target) return;
  if (!pet || !pet.available) {
    setHTML(target, '<div class="empty">桌宠模块未就绪。</div>');
    if (actions) setHTML(actions, '');
    return;
  }
  const isOnline = pet.status === 'online';
  const statusHtml = isOnline
    ? `<span class="value-ok">运行中</span>${pet.pid ? ` (PID ${esc(pet.pid)})` : ''}`
    : '<span class="value-warn">未启动 / 离线</span>';

  const settings = pet.settings || {};
  const scale = Number(settings.scale || 1.0);
  const opacity = Number(settings.opacity || 1.0);
  const lastTurn = pet.last_turn;
  const lastTurnText = lastTurn
    ? `${esc(lastTurn.time || '最近')} · ${traceDuration(lastTurn.elapsed_ms)} (${esc(lastTurn.reply_chars ?? 0)} 字)`
    : '<span class="muted">暂无桌面交互</span>';

  setHTML(target, [
    row('状态', statusHtml),
    row('角色', text(pet.character_name || characterName)),
    row('设定', text(pet.worldview || '未配置')),
    row('Live2D 模型', text(pet.live2d?.loaded ? (pet.live2d.model || '已加载') : '待装载 (未检测到模型)')),
    row('基础尺寸', text(`${pet.window?.width || 340} × ${pet.window?.height || 560}`)),
    row('缩放比例', text(`${Math.round(scale * 100)}%`)),
    row('窗口透明度', text(`${Math.round(opacity * 100)}%`)),
    row('音效提示', yesNo(settings.soundEnabled ?? true)),
    row('最近交互', lastTurnText)
  ].join(''));

  if (actions) {
    setHTML(actions, isOnline
      ? '<button type="button" data-pet-action="stop">停止桌宠</button><button type="button" data-pet-action="restart">重启桌宠</button>'
      : '<button type="button" data-pet-action="launch">启动桌面桌宠</button>');
  }
}

const providerDrafts = new Map();

// 草稿仅在用户编辑时持有；接口刷新不把旧配置重新当成草稿，密钥不写本地存储。
function captureProviderDrafts(element) {
  const provider = element.closest('.provider[data-provider]');
  if (!provider) return;
  const inputs = provider.querySelectorAll('input');
  providerDrafts.set(provider.dataset.provider, {
    model: inputs[0].value, base: inputs[1].value, key: inputs[2].value
  });
}

function renderModelServices(services) {
  const target = document.getElementById('model-services');
  if (!services || services.available === false) {
    setHTML(target, '<div class="empty">模型服务控制尚未装配。</div>');
    return;
  }
  if (services.status === 'error') {
    setHTML(target, `<div class="error-mark">读取模型服务失败：${esc(services.reason)}</div>`);
    return;
  }
  const providers = services.providers || [];
  if (!providers.length) {
    setHTML(target, '<div class="empty">暂时没有供应方目录。</div>');
    return;
  }
  setHTML(target, `<div class="muted">当前激活：<code>${esc(services.activeProviderId || '—')}</code> / <code>${esc(services.activeModel || '—')}</code></div>` + providers.map((provider) => {
    const key = providerKey(provider.id);
    const providerId = String(provider.id || '');
    const draft = providerDrafts.get(providerId) || {};
    const model = Object.prototype.hasOwnProperty.call(draft, 'model') ? draft.model : (provider.chatModel || '');
    const baseUrl = Object.prototype.hasOwnProperty.call(draft, 'base') ? draft.base : (provider.baseUrl || '');
    const apiKey = Object.prototype.hasOwnProperty.call(draft, 'key') ? draft.key : '';
    const models = provider.models || [];
    const options = models.map((model) => `<option value="${esc(model)}"${model === provider.chatModel ? ' selected' : ''}>${esc(model)}</option>`).join('');
    const active = provider.active ? '<span class="active-mark">● 当前激活</span>' : '';
    const probeError = provider.modelProbeError ? `<div class="error-mark">上次刷新失败：${esc(provider.modelProbeError)}</div>` : '';
    return `<div class="provider" data-provider="${esc(providerId)}" data-protocol="${esc(provider.protocol)}">
      <div class="provider-head"><strong>${esc(provider.label || provider.id)}</strong><span>${active}</span></div>
      <div class="provider-meta">${esc(provider.id)} · ${esc(provider.protocol)} · ${esc(provider.baseUrl || '未配置')}</div>
      <div class="provider-form">
        <label>模型名<input id="${key}_model" value="${esc(model)}" placeholder="模型名"></label>
        <label>Base URL<input id="${key}_base" value="${esc(baseUrl)}" placeholder="Base URL"></label>
        <label>API Key<input id="${key}_key" type="password" autocomplete="new-password" value="${esc(apiKey)}" placeholder="${provider.hasApiKey ? '已保存密钥，留空保持' : 'API Key'}"></label>
      </div>
      <div class="provider-actions">
        <select aria-label="${esc(provider.label || provider.id)} 可用模型" id="${key}_models"${models.length ? '' : ' disabled'}><option value="">${models.length ? '选择已发现的模型' : '尚未刷新模型列表'}</option>${options}</select>
        <button data-model-action="save" data-provider="${esc(provider.id)}">保存并激活</button>
        <button data-model-action="select" data-provider="${esc(provider.id)}">激活当前模型</button>
        <button data-model-action="probe" data-provider="${esc(provider.id)}">刷新模型列表</button>
      </div>${probeError}
    </div>`;
  }).join(''));
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  let data = {};
  try { data = await response.json(); } catch (_) { /* empty response */ }
  if (!response.ok || data.ok === false) throw new Error(data.reason || `HTTP ${response.status}`);
  return data;
}

document.getElementById('agent-actions').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-agent-action]');
  if (!button) return;
  const unlock = lockControls(button);
  try {
    const action = button.dataset.agentAction;
    let body = { action };
    if (action === 'permission') {
      const mode = document.getElementById('agent-permission-mode').value;
      if (mode === 'full' || mode === 'ultra') {
        const warning = mode === 'full'
          ? `完全访问会允许${characterName}按已注册工具范围自动执行读写等操作，具体边界仍由实例工具策略决定。确认切换？`
          : 'Ultra 会解除 Shinku 工具层限制，允许其使用本机已注册能力（包括已注册的系统命令工具）；操作系统仍会限制无权限操作。确认切换？';
        if (!window.confirm(warning)) return;
      }
      body = { action: 'permission', mode, acknowledge_risk: mode === 'full' || mode === 'ultra' };
    }
    await postJson('/control-center/agent/control', body);
    if (action === 'permission') permissionDraft = null;
    notifyOperation('操作成功，正在核对最新状态。');
    await refreshAfterAction();
  } catch (error) {
    notifyOperation(`Agent 控制失败：${error.message}`, true);
  } finally {
    unlock();
    if (lastSnapshot) renderAgentActions(lastSnapshot.agent?.controls || {});
  }
});

document.getElementById('pet-actions').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-pet-action]');
  if (!button) return;
  const unlock = lockControls(button);
  try {
    await postJson('/control-center/pet/control', { action: button.dataset.petAction });
    notifyOperation('操作成功，正在核对最新状态。');
    await refreshAfterAction();
  } catch (error) {
    notifyOperation(`桌宠控制失败：${error.message}`, true);
  } finally {
    unlock();
    if (lastSnapshot) renderDesktopPet(lastSnapshot.pet || {});
  }
});

document.getElementById('model-services').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-model-action]');
  if (!button) return;
  const providerId = button.dataset.provider;
  const key = providerKey(providerId);
  const action = button.dataset.modelAction;
  const unlock = lockControls(button);
  try {
    if (action === 'probe') {
      await postJson(`/control-center/model-services/${encodeURIComponent(providerId)}/models`, {});
    } else if (action === 'select') {
      await postJson('/control-center/model-services/select', {
        providerId,
        modelId: document.getElementById(`${key}_model`).value.trim()
      });
    } else if (action === 'save') {
      await postJson(`/control-center/model-services/${encodeURIComponent(providerId)}/config`, {
        protocol: button.closest('.provider').dataset.protocol,
        baseUrl: document.getElementById(`${key}_base`).value.trim(),
        apiKey: document.getElementById(`${key}_key`).value,
        chatModel: document.getElementById(`${key}_model`).value.trim(),
        activate: true
      });
      providerDrafts.delete(providerId);
      document.getElementById(`${key}_key`).value = '';
    }
    notifyOperation('操作成功，正在核对最新状态。');
    await refreshAfterAction();
  } catch (error) {
    notifyOperation(`模型服务操作失败：${error.message}`, true);
  } finally {
    unlock();
    if (lastSnapshot) renderModelServices(lastSnapshot.model_services || {});
  }
});

document.getElementById('model-services').addEventListener('change', (event) => {
  const select = event.target.closest('select[id$="_models"]');
  if (!select || !select.value) return;
  const input = document.getElementById(`${select.id.slice(0, -7)}_model`);
  if (input) { input.value = select.value; captureProviderDrafts(input); }
});

document.getElementById('agent-tasks').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-task-id]');
  if (!button) return;
  const taskId = button.dataset.taskId || '';
  button.disabled = true;
  try {
    const response = await fetch(`/agent-work/tasks/${encodeURIComponent(taskId)}`, { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.reason || `HTTP ${response.status}`);
    renderTaskDetail(data);
    document.getElementById('task-detail').scrollIntoView({ block: 'nearest' });
  } catch (error) {
    renderTaskDetail({ ok: false, reason: error.message });
  } finally {
    button.disabled = false;
  }
});

document.getElementById('approval-inspect').addEventListener('click', async () => {
  const button = document.getElementById('approval-inspect');
  const conversationId = document.getElementById('approval-conversation').value.trim();
  if (!conversationId) {
    renderApproval({ ok: false, reason: '请先填写会话 ID' });
    return;
  }
  button.disabled = true;
  try {
    const response = await fetch(`/capabilities/approvals/${encodeURIComponent(conversationId)}`, { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.reason || `HTTP ${response.status}`);
    renderApproval(data);
  } catch (error) {
    renderApproval({ ok: false, reason: error.message });
  } finally {
    button.disabled = false;
  }
});

document.getElementById('memory-inspect').addEventListener('click', async () => {
  const button = document.getElementById('memory-inspect');
  const conversationId = document.getElementById('memory-conversation').value.trim();
  const conversationType = document.getElementById('memory-type').value;
  const mode = document.getElementById('memory-mode').value;
  const query = document.getElementById('memory-query').value.trim();
  if (!conversationId) {
    setHTML(document.getElementById('memory-result'), '<div class="error-mark">请先填写会话 ID。</div>');
    return;
  }
  button.disabled = true;
  try {
    const params = new URLSearchParams({ conversation_id: conversationId, conversation_type: conversationType, limit: '20' });
    if (mode === 'facts') {
      const response = await fetch(`/memory/facts?${params.toString()}`, { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.reason || `HTTP ${response.status}`);
      renderMemoryResult(data);
    } else {
      params.set('mode', mode);
      if (query) params.set('query', query);
      const response = await fetch(`/memory/timeline?${params.toString()}`, { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.reason || `HTTP ${response.status}`);
      renderMemoryResult(data);
    }
  } catch (error) {
    renderMemoryResult({ ok: false, reason: error.message });
  } finally {
    button.disabled = false;
  }
});


// 读取请求有截止时间；可选接口失败保留上次成功值并标出陈旧状态，不能把失败显示为零条。
async function fetchOptional(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(url, { cache: 'no-store', signal: controller.signal });
    // 正文解析也在截止时间内，避免收到响应头后连接悬挂而永久锁住轮询。
    const payload = await response.json();
    return { ok: response.ok, status: response.status, payload };
  } catch (_) {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

const optionalSources = [
  ['metrics_recent', '/metrics/recent?days=7', '近七日指标', 'metrics-recent', (data) => data.days || []],
  ['traces', '/debug/hub/traces?limit=30', '处理链', 'traces', (data) => data.traces || []],
  ['attention', '/debug/hub/attention?limit=80', '注意力', 'attention-traces'],
  ['memory_routing', '/debug/hub/memory-routing?limit=60', '记忆路由', 'memory-routing'],
  ['qq_connection', '/api/qq/connection', 'QQ 连接', null],
  ['qq_diagnostics', '/api/qq/self-check', 'QQ 自检', 'qq-diagnostics'],
  ['capabilities', '/capabilities', '工具目录', 'capabilities'],
  ['agent_tasks', '/agent-work/tasks?limit=20', '后台任务', 'agent-tasks'],
  ['memory_features', '/memory/features', '记忆投影', null]
];

async function readSnapshot() {
  const button = document.getElementById('refresh-now');
  button.disabled = true;
  try {
    const [stateResponse, ...responses] = await Promise.all([
      fetchOptional('/debug/hub/state'), ...optionalSources.map(([, url]) => fetchOptional(url))
    ]);
    if (!stateResponse?.ok) throw new Error(stateResponse ? `HTTP ${stateResponse.status}` : '连接失败或请求超时');
    const snapshot = stateResponse.payload;
    const failures = [];
    await Promise.all(optionalSources.map(async ([key, , label, target, extract], index) => {
      try {
        const response = responses[index];
        if (!response?.ok) throw new Error('unavailable');
        const data = response.payload;
        if (data.ok === false && key !== 'qq_diagnostics') throw new Error(data.reason || 'unavailable');
        snapshot[key] = extract ? extract(data) : data;
      } catch (_) {
        if (lastSnapshot?.[key] !== undefined) snapshot[key] = lastSnapshot[key];
        failures.push({ key, label, target });
      }
    }));
    render(snapshot);
    failures.filter(({ key, target }) => target && snapshot[key] === undefined).forEach(({ target, label }) => {
      setHTML(document.getElementById(target), `<div class="error-mark">${esc(label)}暂时读取失败，稍后自动重试。</div>`);
      const footer = document.getElementById(`${target}-pagination`);
      if (footer) footer.hidden = true;
    });
    const warning = document.getElementById('connection-warning');
    warning.hidden = !failures.length;
    warning.textContent = failures.length ? `部分数据读取失败：${failures.map(({ label }) => label).join('、')}。已有内容保留上次成功结果，暂未更新。` : '';
    document.getElementById('connection-state').textContent = failures.length ? '部分数据未更新' : '已连接 · 本机';
    lastSnapshot = snapshot;
    return true;
  } catch (error) {
    document.getElementById('updated').textContent = `读取失败：${error.message} · 已有内容为上次成功结果`;
    const warning = document.getElementById('connection-warning');
    warning.hidden = false;
    warning.textContent = `无法读取本机服务：${error.message}。已有内容尚未更新，连接恢复后自动重试。`;
    document.getElementById('connection-state').textContent = '连接中断';
    return false;
  } finally {
    button.disabled = false;
  }
}

function refresh() {
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = readSnapshot().finally(() => { refreshInFlight = null; });
  return refreshInFlight;
}

async function refreshAfterAction() {
  // 操作可能与旧轮询同时完成；先等旧轮询，再读取写入后的实际状态，避免旧快照覆盖成功操作。
  if (refreshInFlight) await refreshInFlight;
  const ok = await refresh();
  notifyOperation(ok ? '操作成功，已读取最新状态。' : '操作已提交，但状态核对失败。请刷新核实后再决定是否重试。', !ok);
}

function updateNavigation() {
  const sections = Array.from(document.querySelectorAll('main > .section'));
  const offset = document.querySelector('.topbar').getBoundingClientRect().height + (window.innerWidth <= 760 ? 70 : 36);
  let current = sections[0];
  sections.forEach((section) => { if (section.getBoundingClientRect().top <= offset) current = section; });
  const extent = document.documentElement.scrollHeight - window.innerHeight;
  if (extent > 0 && window.scrollY >= extent - 4) current = sections[sections.length - 1];
  document.querySelectorAll('.section-nav a').forEach((link) => {
    const active = link.hash === `#${current.id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
  });
  const percent = extent > 0 ? Math.max(0, Math.min(100, Math.round(window.scrollY / extent * 100))) : 0;
  document.getElementById('read-percent').textContent = `${percent}%`;
  document.getElementById('read-progress').style.width = `${percent}%`;
}

// 锚点使用浏览器原生历史与滚动；设置使用原生 dialog 的焦点管理和 Escape 关闭。
document.querySelectorAll('[data-dialog]').forEach((button) => button.addEventListener('click', () => {
  document.getElementById(button.dataset.dialog).showModal();
}));
document.querySelectorAll('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
document.querySelectorAll('dialog').forEach((dialog) => dialog.addEventListener('close', () => { if (autoRefreshEnabled) refresh(); }));
document.getElementById('model-services').addEventListener('input', (event) => captureProviderDrafts(event.target));
document.getElementById('agent-actions').addEventListener('change', (event) => {
  if (event.target.id === 'agent-permission-mode') permissionDraft = event.target.value;
});
document.getElementById('task-detail').addEventListener('click', (event) => {
  if (event.target.closest('[data-close-task]')) setHTML(document.getElementById('task-detail'), '');
});
document.querySelectorAll('[data-channel]').forEach((button) => button.addEventListener('click', () => {
  traceChannel = button.dataset.channel;
  document.querySelectorAll('[data-channel]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  listLimits.delete('traces');
  if (lastSnapshot) { renderTraces(lastSnapshot.traces || []); paginateList('traces'); }
}));
document.getElementById('trace-search').addEventListener('input', () => {
  listLimits.delete('traces');
  if (lastSnapshot) { renderTraces(lastSnapshot.traces || []); paginateList('traces'); }
});
document.getElementById('task-filter').addEventListener('change', () => {
  listLimits.delete('agent-tasks');
  if (lastSnapshot) { renderTasks(lastSnapshot.agent_tasks || {}); paginateList('agent-tasks'); }
});
document.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-list]');
  if (!button) return;
  listLimits.set(button.dataset.list, button.dataset.expand === 'true' ? (listLimits.get(button.dataset.list) || 8) + 12 : 8);
  paginateList(button.dataset.list);
});
document.getElementById('compact-view').addEventListener('change', (event) => {
  document.body.classList.toggle('compact', event.target.checked);
  try { localStorage.setItem('shinku-control-compact', String(event.target.checked)); } catch (_) { /* 存储禁用时仍可使用当前页。 */ }
});
try {
  const compact = localStorage.getItem('shinku-control-compact') === 'true';
  document.body.classList.toggle('compact', compact);
  document.getElementById('compact-view').checked = compact;
} catch (_) { /* 只有显示偏好落盘，业务数据与密钥不持久化到浏览器。 */ }
window.addEventListener('scroll', updateNavigation, { passive: true });
window.addEventListener('resize', updateNavigation);
document.getElementById('refresh-now').addEventListener('click', () => refresh());
document.getElementById('auto-refresh').addEventListener('click', () => {
  autoRefreshEnabled = !autoRefreshEnabled;
  const button = document.getElementById('auto-refresh');
  button.textContent = `自动刷新：${autoRefreshEnabled ? '开' : '关'}`;
  button.setAttribute('aria-pressed', String(autoRefreshEnabled));
  if (autoRefreshEnabled) refresh();
});
refresh();
setInterval(() => {
  const active = document.activeElement;
  const editing = active && active.matches('input,textarea,select') && !active.disabled && !active.readOnly;
  if (autoRefreshEnabled && !editing && !document.querySelector('dialog[open]') && !mutationsInFlight) refresh();
}, 3000);
