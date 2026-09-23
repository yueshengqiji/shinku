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
  document.getElementById('metrics').innerHTML = [
    metricCard('聊天调用', formatNumber(metricNumber(metrics, 'chat_json_calls') + metricNumber(metrics, 'chat_stream_calls'))),
    metricCard('输入 Token', formatNumber(metricNumber(metrics, 'input_tokens_chat'))),
    metricCard('输出 Token', formatNumber(metricNumber(metrics, 'output_tokens_chat'))),
    metricCard('总 Token', formatNumber(metricNumber(metrics, 'total_tokens_chat'))),
    metricCard('缓存读入', formatNumber(cacheRead)),
    metricCard('可观测命中率', cacheRate, cacheObserved > 0 ? 'value-ok' : 'value-warn'),
    metricCard('工具决策', formatNumber(metricNumber(metrics, 'native_tool_decision_sent'))),
    metricCard('最近错误', esc(errorValue), lastError.type ? 'value-bad' : 'value-ok')
  ].join('');
}

function renderRuntimeConfig(snapshot) {
  const target = document.getElementById('runtime-config');
  const diagnostics = snapshot.agent?.diagnostics || {};
  const config = diagnostics.config || {};
  const flow = diagnostics.chat_flow || {};
  const extension = diagnostics.runtime_extension || {};
  if (!Object.keys(config).length) {
    target.innerHTML = '<div class="empty">当前没有可读取的运行配置。</div>';
    return;
  }
  target.innerHTML = [
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
  ].join('');
}

function renderRecentMetrics(days) {
  const target = document.getElementById('metrics-recent');
  if (!Array.isArray(days) || !days.length) {
    target.innerHTML = '<div class="empty">当前没有持久化的日指标；运行时内存指标仍显示在上方。</div>';
    return;
  }
  const keys = ['chat_json_calls', 'chat_stream_calls', 'total_tokens_chat', 'cache_read_tokens_chat', 'native_tool_call_extracted'];
  const labels = ['聊天 JSON', '聊天流式', '聊天 Token', '缓存读入', '工具调用'];
  const header = ['日期', ...labels].map((label) => `<th>${esc(label)}</th>`).join('');
  const rows = days.slice().reverse().map((day) => {
    const values = keys.map((key) => `<td>${formatNumber(metricNumber(day, key))}</td>`).join('');
    return `<tr><td>${esc(day.day || day.date || '—')}</td>${values}</tr>`;
  }).join('');
  target.innerHTML = `<div class="history-scroll"><table class="history-table"><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function traceStatusClass(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'completed' || value === 'command' || value === 'finished') return 'trace-status-ok';
  if (value === 'failed' || value === 'sender_error' || value === 'partial') return 'trace-status-bad';
  return 'trace-status-warn';
}

function traceDuration(value) {
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

function renderTraces(traces) {
  const target = document.getElementById('traces');
  if (!Array.isArray(traces) || !traces.length) {
    target.innerHTML = '<div class="empty">暂无回合追踪；发送一条消息后这里会显示处理链。</div>';
    return;
  }
  target.innerHTML = `<div class="trace-list">${traces.slice(0, 30).map((trace) => {
    const status = String(trace.status || 'finished');
    const stages = Array.isArray(trace.stages) ? trace.stages : [];
    const conversation = [trace.conversation_type, trace.conversation_id].filter(Boolean).join(' / ') || '—';
    const stageHtml = stages.length
      ? `<details class="trace-stages"><summary>阶段 ${stages.length} 个</summary>${stages.map((stage) => {
        const details = { ...stage };
        delete details.name;
        delete details.elapsed_ms;
        return `<div class="trace-stage"><span class="trace-stage-name">${esc(stage.name || 'stage')}</span><span class="trace-stage-time">${traceDuration(stage.elapsed_ms)}</span><span class="trace-stage-details">${esc(traceDetails(details))}</span></div>`;
      }).join('')}</details>`
      : '<div class="trace-stages muted">没有阶段记录</div>';
    const reason = trace.reason ? `<div class="trace-reason">原因：${esc(trace.reason)}</div>` : '';
    return `<article class="trace">
      <div class="trace-head"><strong>${esc(String(trace.trace_id || '').slice(0, 12) || '—')}</strong><span class="trace-status ${traceStatusClass(status)}">${esc(status)}</span></div>
      <div class="trace-meta">
        <span>${esc(conversation)}</span>
        <span>耗时 ${traceDuration(trace.elapsed_ms)}</span>
        <span>输入 ${esc(trace.input_chars ?? 0)} 字</span>
        <span>输出 ${esc(trace.reply_chars ?? 0)} 字</span>
        <span>Agent ${esc(trace.agent_status || '—')}</span>
        <span>投递 ${esc(trace.delivery_status || '—')}</span>
      </div>${reason}${stageHtml}
    </article>`;
  }).join('')}</div>`;
}

function renderQQDiagnostics(payload) {
  const target = document.getElementById('qq-diagnostics');
  const checks = Array.isArray(payload?.checks) ? payload.checks : [];
  if (!checks.length) {
    target.innerHTML = '<div class="empty">QQ 诊断接口未装配或暂时不可用。</div>';
    return;
  }
  const summary = payload.ok
    ? '<span class="value-ok">通过</span>'
    : '<span class="value-warn">有项目需要注意</span>';
  target.innerHTML = `<div class="memory-summary">总体状态：${summary}</div><div class="check-list">${checks.map((check) => `
    <div class="check-row"><strong>${check.ok ? '<span class="value-ok">✓</span>' : '<span class="value-bad">✕</span>'} ${esc(check.name || 'check')}</strong><span class="check-detail">${esc(check.detail || '')}</span></div>
  `).join('')}</div>`;
}

function renderCapabilities(payload) {
  const target = document.getElementById('capabilities');
  const rows = Array.isArray(payload?.capabilities) ? payload.capabilities : [];
  if (!rows.length) {
    target.innerHTML = '<div class="empty">当前没有可展示的工具，或能力目录尚未装配。</div>';
    return;
  }
  target.innerHTML = `<div class="capability-list">${rows.map((item) => {
    const risk = String(item.risk || 'unknown').toLowerCase();
    const riskClass = ['low', 'medium', 'high', 'critical'].includes(risk) ? risk : '';
    const schema = item.input_schema && Object.keys(item.input_schema).length
      ? `<details class="trace-stages"><summary>输入结构</summary><pre>${esc(traceDetails(item.input_schema))}</pre></details>`
      : '';
    return `<article class="capability">
      <div class="capability-head"><strong>${esc(item.name || 'unnamed')}</strong><span class="risk-${riskClass}">${esc(risk)}</span></div>
      <div class="capability-meta"><span>授权：${item.requires_approval === true ? '需要' : item.requires_approval === false ? '不需要' : '未知'}</span><span>允许：${item.allowlisted === false ? '否' : '是'}</span></div>
      <div class="capability-description">${esc(item.description || '没有描述')}</div>${schema}
    </article>`;
  }).join('')}</div>`;
}

function renderTasks(payload) {
  const target = document.getElementById('agent-tasks');
  const rows = Array.isArray(payload?.tasks) ? payload.tasks : [];
  if (!rows.length) {
    target.innerHTML = '<div class="empty">当前没有后台任务，或任务协调器尚未装配。</div>';
    return;
  }
  target.innerHTML = `<div class="task-list">${rows.slice(0, 20).map((task) => {
    const status = String(task.status || 'unknown');
    const statusClass = status === 'completed' ? 'value-ok' : ['failed', 'canceled'].includes(status) ? 'value-bad' : 'value-warn';
    return `<article class="task"><div class="task-head"><strong>${esc(task.task_id || '—')}</strong><span class="${statusClass}">${esc(status)}</span></div><div class="task-meta">${esc(task.goal || '未填写任务目标')} · 第 ${esc(task.attempt ?? 1)} 次</div>${task.error ? `<div class="error-mark">${esc(task.error)}</div>` : ''}<div class="task-actions"><button type="button" data-task-id="${esc(task.task_id || '')}">查看详情</button></div></article>`;
  }).join('')}</div><div id="task-detail"></div>`;
}

function renderTaskDetail(payload) {
  const target = document.getElementById('task-detail');
  if (!target) return;
  if (!payload || payload.ok === false || !payload.task) {
    target.innerHTML = `<div class="task-detail error-mark">读取任务失败：${esc(payload?.reason || 'task_not_found')}</div>`;
    return;
  }
  const task = payload.task;
  target.innerHTML = `<div class="task-detail"><div class="kv">
    <div>任务号</div><div>${text(task.task_id)}</div>
    <div>状态</div><div>${esc(task.status || '—')}</div>
    <div>目标</div><div>${esc(task.goal || '—')}</div>
    <div>结果</div><div>${esc(task.result_summary || '—')}</div>
    <div>会话</div><div>${esc(task.session_id || '—')}</div>
  </div>${task.metadata && Object.keys(task.metadata).length ? `<details class="trace-stages"><summary>元数据</summary><pre>${esc(traceDetails(task.metadata))}</pre></details>` : ''}</div>`;
}

function renderApproval(payload) {
  const target = document.getElementById('approval-result');
  if (!payload || payload.ok === false) {
    target.innerHTML = `<div class="error-mark">读取失败：${esc(payload?.reason || 'approval_endpoint_unavailable')}</div>`;
    return;
  }
  if (!payload.pending || !payload.approval) {
    target.innerHTML = '<div class="empty">当前会话没有待授权工具调用。</div>';
    return;
  }
  const approval = payload.approval;
  const tools = Array.isArray(approval.tools) && approval.tools.length
    ? approval.tools
    : [{ tool: approval.tool, risk: approval.risk, arguments: approval.arguments }];
  target.innerHTML = `<div class="approval-box">
    <div class="memory-summary">请求：${esc(approval.request_id || '—')} · 风险：${esc(approval.risk || 'unknown')} · 任务：${esc(approval.task_id || '—')}</div>
    <div>${esc(approval.question || '等待用户确认')}</div>
    <div class="approval-tools">${tools.map((item) => `<div class="approval-tool"><strong>${esc(item.tool || 'unnamed')}</strong> · ${esc(item.risk || 'unknown')}<pre>${esc(traceDetails(item.arguments))}</pre></div>`).join('')}</div>
  </div>`;
}

function renderMemoryResult(payload) {
  const target = document.getElementById('memory-result');
  if (!payload || payload.ok === false) {
    target.innerHTML = `<div class="error-mark">读取失败：${esc(payload?.reason || 'memory_endpoint_unavailable')}</div>`;
    return;
  }
  const rows = Array.isArray(payload.facts) ? payload.facts : Array.isArray(payload.records) ? payload.records : [];
  const mode = payload.mode || 'facts';
  const summary = `作用域：${payload.scope || '—'} · 模式：${mode} · ${rows.length} 条`;
  if (!rows.length) {
    target.innerHTML = `<div class="memory-summary">${esc(summary)}</div><div class="empty">没有找到记录。</div>`;
    return;
  }
  target.innerHTML = `<div class="memory-summary">${esc(summary)}</div><div class="memory-list">${rows.map((record) => {
    const created = Number(record.created_at);
    const when = Number.isFinite(created) && created > 0 ? new Date(created * 1000).toLocaleString() : '时间未知';
    return `<article class="memory-item"><strong>${esc(record.layer || 'record')} · ${esc(when)}</strong><div class="memory-meta">${esc(record.source_id || record.record_id || '')}</div><pre>${esc(record.text || '')}</pre></article>`;
  }).join('')}</div>`;
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

  document.getElementById('service').innerHTML = [
    row('状态', health.status === 'ok' ? '<span class="value-ok">运行中</span>' : '<span class="value-bad">异常</span>'),
    row('项目', text(health.project)),
    row('PID', text((snapshot.process || {}).pid)),
    row('端口', text(health.port)),
    row('Python', text(health.python))
  ].join('');

  document.getElementById('agent').innerHTML = [
    row('状态', agent.enabled ? '<span class="value-ok">已装配</span>' : '<span class="value-warn">未装配</span>'),
    row('进程', text(`${agent.process?.service || 'backend'} / PID ${agent.process?.pid || snapshot.process?.pid || '—'}（${agent.process?.mode || '—'}）`)),
    row('运行状态', controls.responses_enabled === false ? '<span class="value-warn">已暂停</span>' : '<span class="value-ok">可接收</span>'),
    row('模型', text(liveModels.chat || diagnostics.chat_model_name)),
    row('发送', yesNo(diagnostics.send_enabled)),
    row('人设', text((agent.diagnostics || {}).persona?.path)),
    row('工具', text((bridge.tool_names || []).join(', ') || '无')),
    row('原生工具', runtimeHealth.native_tools_supported === true ? '<span class="value-ok">支持</span>' : runtimeHealth.native_tools_supported === false ? '<span class="value-bad">不支持</span>' : '<span class="value-warn">未知</span>'),
    row('轮数上限', text(bridge.max_rounds))
  ].join('');

  document.getElementById('memory').innerHTML = [
    row('回合', text(memory.turns || 0)),
    row('活动事件', text(memory.active_events || 0)),
    row('长期事实', text(memory.active_facts || 0)),
    row('遗忘记录', text(memory.forgotten || 0)),
    row('记忆路由审计', text(memory.routing_audits || 0)),
    row('数据库', text((agent.memory || {}).path))
  ].join('');

  document.getElementById('napcat').innerHTML = [
    row('已配置', yesNo(napcat.configured)),
    row('回合调度', yesNo(napcat.dispatcher_attached)),
    row('传输', text(napcat.transport)),
    row('Webhook', text((napcat.diagnostics || {}).webhook_path))
  ].join('');

  renderAgentActions(controls);
  renderMetrics(snapshot);
  renderRuntimeConfig(snapshot);
  renderRecentMetrics(snapshot.metrics_recent || []);
  renderTraces(snapshot.traces || []);
  renderQQDiagnostics(snapshot.qq_diagnostics || {});
  renderCapabilities(snapshot.capabilities || {});
  renderTasks(snapshot.agent_tasks || {});
  renderModelServices(snapshot.model_services || {});

  const logs = document.getElementById('logs');
  const items = snapshot.logs || [];
  logs.innerHTML = items.length
    ? items.map((item) => `<div class="log-title"><span>${text(item.name)}</span><span>${esc(item.updated_at)}</span></div><pre>${esc((item.lines || []).join('\n'))}</pre>`).join('<hr style="border:0;border-top:1px solid var(--line);margin:14px 0">')
    : '<div class="empty">当前日志目录没有可读日志</div>';
  document.getElementById('updated').textContent = `最近刷新：${new Date(snapshot.observed_at || Date.now()).toLocaleString()}（每 3 秒刷新）`;
}

function providerKey(id) {
  return `provider_${String(id || '').replace(/[^a-zA-Z0-9_-]/g, '_')}`;
}

function renderAgentActions(controls) {
  const target = document.getElementById('agent-actions');
  if (!controls || controls.agent_enabled === undefined) {
    target.innerHTML = '';
    return;
  }
  target.innerHTML = [
    '<button data-agent-action="start">恢复回复</button>',
    '<button data-agent-action="stop">暂停回复</button>',
    '<button data-agent-action="on">启用 Agent</button>',
    '<button data-agent-action="off">关闭 Agent</button>'
  ].join('');
}

const providerDrafts = new Map();

function captureProviderDrafts() {
  document.querySelectorAll('.provider[data-provider]').forEach((element) => {
    const providerId = element.dataset.provider || '';
    if (!providerId) return;
    const inputs = element.querySelectorAll('input');
    providerDrafts.set(providerId, {
      model: inputs[0]?.value ?? '',
      base: inputs[1]?.value ?? '',
      key: inputs[2]?.value ?? ''
    });
  });
}

function renderModelServices(services) {
  const target = document.getElementById('model-services');
  captureProviderDrafts();
  if (!services || services.available === false) {
    target.innerHTML = '<div class="empty">模型服务控制尚未装配。</div>';
    return;
  }
  if (services.status === 'error') {
    target.innerHTML = `<div class="error-mark">读取模型服务失败：${esc(services.reason)}</div>`;
    return;
  }
  const providers = services.providers || [];
  if (!providers.length) {
    target.innerHTML = '<div class="empty">暂时没有供应方目录。</div>';
    return;
  }
  target.innerHTML = `<div class="muted">当前激活：<code>${esc(services.activeProviderId || '—')}</code> / <code>${esc(services.activeModel || '—')}</code></div>` + providers.map((provider) => {
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
    return `<div class="provider" data-provider="${esc(providerId)}">
      <div class="provider-head"><strong>${esc(provider.label || provider.id)}</strong><span>${active}</span></div>
      <div class="provider-meta">${esc(provider.id)} · ${esc(provider.protocol)} · ${esc(provider.baseUrl || '未配置')}</div>
      <div class="provider-form">
        <input id="${key}_model" value="${esc(model)}" placeholder="模型名">
        <input id="${key}_base" value="${esc(baseUrl)}" placeholder="Base URL">
        <input id="${key}_key" type="password" value="${esc(apiKey)}" placeholder="${provider.hasApiKey ? '已保存密钥，留空保持' : 'API Key'}">
      </div>
      <div class="provider-actions">
        <select id="${key}_models"${models.length ? '' : ' disabled'}><option value="">${models.length ? '选择已发现的模型' : '尚未刷新模型列表'}</option>${options}</select>
        <button data-model-action="save" data-provider="${esc(provider.id)}">保存并激活</button>
        <button data-model-action="select" data-provider="${esc(provider.id)}">激活当前模型</button>
        <button data-model-action="probe" data-provider="${esc(provider.id)}">刷新模型列表</button>
      </div>${probeError}
    </div>`;
  }).join('');
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
  button.disabled = true;
  try {
    await postJson('/control-center/agent/control', { action: button.dataset.agentAction });
    await refresh();
  } catch (error) {
    alert(`Agent 控制失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
});

document.getElementById('model-services').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-model-action]');
  if (!button) return;
  const providerId = button.dataset.provider;
  const key = providerKey(providerId);
  const action = button.dataset.modelAction;
  button.disabled = true;
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
        protocol: button.closest('.provider').querySelector('.provider-meta').textContent.split('·')[1].trim(),
        baseUrl: document.getElementById(`${key}_base`).value.trim(),
        apiKey: document.getElementById(`${key}_key`).value,
        chatModel: document.getElementById(`${key}_model`).value.trim(),
        activate: true
      });
      providerDrafts.delete(providerId);
    }
    await refresh();
  } catch (error) {
    alert(`模型服务操作失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
});

document.getElementById('model-services').addEventListener('change', (event) => {
  const select = event.target.closest('select[id$="_models"]');
  if (!select || !select.value) return;
  const input = document.getElementById(`${select.id.slice(0, -7)}_model`);
  if (input) input.value = select.value;
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
    document.getElementById('memory-result').innerHTML = '<div class="error-mark">请先填写会话 ID。</div>';
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

async function optionalJson(response, fallback) {
  if (!response || !response.ok) return fallback;
  try {
    return await response.json();
  } catch (_) {
    return fallback;
  }
}

async function fetchOptional(url) {
  try {
    return await fetch(url, { cache: 'no-store' });
  } catch (_) {
    return null;
  }
}

let refreshInFlight = false;
let autoRefreshEnabled = true;

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const [stateResponse, metricsResponse, tracesResponse, diagnosticsResponse, capabilitiesResponse, tasksResponse] = await Promise.all([
      fetch('/debug/hub/state', { cache: 'no-store' }),
      fetchOptional('/metrics/recent?days=7'),
      fetchOptional('/debug/hub/traces?limit=30'),
      fetchOptional('/api/qq/self-check'),
      fetchOptional('/capabilities'),
      fetchOptional('/agent-work/tasks?limit=20')
    ]);
    if (!stateResponse.ok) throw new Error(`HTTP ${stateResponse.status}`);
    const snapshot = await stateResponse.json();
    if (metricsResponse?.ok) {
      const recent = await metricsResponse.json();
      snapshot.metrics_recent = Array.isArray(recent.days) ? recent.days : [];
    } else {
      snapshot.metrics_recent = [];
    }
    if (tracesResponse?.ok) {
      const traces = await tracesResponse.json();
      snapshot.traces = Array.isArray(traces.traces) ? traces.traces : [];
    } else {
      snapshot.traces = [];
    }
    snapshot.qq_diagnostics = await optionalJson(diagnosticsResponse, {});
    snapshot.capabilities = await optionalJson(capabilitiesResponse, {});
    snapshot.agent_tasks = await optionalJson(tasksResponse, {});
    render(snapshot);
  } catch (error) {
    document.getElementById('updated').textContent = `读取失败：${error}`;
  } finally {
    refreshInFlight = false;
  }
}

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
  if (autoRefreshEnabled) refresh();
}, 3000);
