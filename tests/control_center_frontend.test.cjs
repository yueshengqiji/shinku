// 使用 Node 标准库检查轮询与视图状态规则；真实 DOM 交互由同目录隔离夹具验收。
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../frontend/control-center/app.js'), 'utf8');

function harness() {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      innerHTML: '', textContent: '', hidden: false, disabled: false, dataset: {},
      attributes: {}, setAttribute(name, value) { this.attributes[name] = value; },
      listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; }, querySelectorAll() { return []; }, querySelector() { return null; },
      closest() { return null; }, contains() { return false; }, classList: { toggle() {} }
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    document: { getElementById: element, activeElement: null, querySelector: (selector) => element(selector) },
    setTimeout, clearTimeout, AbortController, URLSearchParams, crypto: require("node:crypto").webcrypto,
    fetch: async () => ({ok: true, json: async () => ({})})
  });
  // 不启动实际定时器，不加载浏览器，也不请求本机运行中的后端。
  vm.runInContext(source.split('// 锚点使用浏览器原生历史与滚动')[0], context);
  vm.runInContext('render = (snapshot) => { globalThis.rendered = snapshot; };', context);
  return { context, element, run: (code) => vm.runInContext(code, context) };
}

test('阶段统计保留未知、截断与异常口径并转义标签，不当作业务完成', () => {
  const {run, context} = harness();
  context.performanceFixture = {available:true, truncated:true, phases:[{
    label:'<img src=x onerror=alert(1)>', totalMs:null, measured:0, samples:1,
    p50Ms:null, p95Ms:null, requests:1, requestMs:null, knownInputTokens:null,
    knownOutputTokens:0, incompleteUsage:1, failed:0, requestFailed:1, unsettled:1, requestUnsettled:0
  }]};
  const html = run('phasePerformanceHTML(performanceFixture)');
  assert.match(html, /&lt;img/); assert.doesNotMatch(html, /<img/);
  assert.match(html, /未知/); assert.match(html, /记录已截断/);
  assert.match(html, /不可相加/); assert.match(html, /不代表业务完成/);
  assert.equal(run('traceDuration(null)'), '—');
  assert.equal(run('traceDuration(0)'), '0.0 ms');
});

test('同一轮刷新复用请求，接口失败保留上次结果并给出陈旧提示，恢复后清除提示', async () => {
  const {context, element, run} = harness();
  let mode = 'normal'; const requests = [];
  context.fetch = async (url) => {
    requests.push(url);
    if (mode === 'offline' || (mode === 'partial' && url.includes('/traces'))) return {ok: false, status: 503};
    return {ok: true, json: async () => url.includes('/traces') ? {traces: [{trace_id: 'stable'}]} : {}};
  };
  const first = run('refresh()');
  assert.equal(first, run('refresh()'));
  await first;
  // 列举需要读取的端点，同时检查每个端点只有一次请求，保留并发去重约束。
  assert.deepEqual(requests.slice().sort(), [
    '/debug/hub/state', '/metrics/recent?days=7', '/debug/hub/traces?limit=30',
    '/debug/hub/attention?limit=80', '/debug/hub/memory-routing?limit=60',
    '/api/qq/connection', '/api/qq/self-check', '/capabilities',
    '/agent-work/tasks?limit=20', '/memory/features'
  ].sort());
  mode = 'partial'; await run('refresh()');
  assert.equal(context.rendered.traces[0].trace_id, 'stable');
  assert.equal(element('connection-warning').hidden, false);
  assert.match(element('connection-warning').textContent, /处理链/);
  mode = 'offline'; assert.equal(await run('refresh()'), false);
  assert.equal(element('connection-state').textContent, '连接中断');
  assert.equal(run('lastSnapshot.traces[0].trace_id'), 'stable');
  mode = 'normal'; await run('refresh()');
  assert.equal(element('connection-warning').hidden, true);
  assert.equal(element('refresh-now').disabled, false);
});

test('首次失败不伪装为空列表，损坏 JSON 不影响其余区域，QQ 自检失败仍显示诊断数据', async () => {
  const {context, element, run} = harness();
  context.fetch = async (url) => ({ok: true, json: async () => {
    if (url.includes('/traces')) throw new Error('invalid JSON');
    return url.includes('self-check') ? {ok: false, checks: [{name: 'failed-check'}]} : {};
  }});
  await run('refresh()');
  assert.match(element('traces').innerHTML, /暂时读取失败/);
  assert.equal(context.rendered.qq_diagnostics.checks[0].name, 'failed-check');
  assert.equal(context.rendered.traces, undefined);
});

test('写操作等待旧轮询结束并再次读取状态，不能用写入前快照确认成功', async () => {
  const {context, run} = harness();
  let release; let calls = 0;
  const old = new Promise((resolve) => { release = resolve; });
  context.fetch = async (url) => {
    if (url === '/debug/hub/state') {
      calls++;
      if (calls === 1) await old;
      return {ok: true, json: async () => ({revision: calls})};
    }
    return {ok: true, json: async () => ({})};
  };
  const refresh = run('refresh()');
  const verify = run('refreshAfterAction()');
  assert.equal(calls, 1);
  release(); await refresh; await verify;
  assert.equal(calls, 2);
  assert.equal(context.rendered.revision, 2);
});

test('不变化的片段不替换节点，输入焦点和忙碌控件不会被重绘', () => {
  const {context, element, run} = harness();
  const target = element('target'); let writes = 0; let html = '';
  Object.defineProperty(target, 'innerHTML', { get: () => html, set: (value) => { writes++; html = value; } });
  context.target = target;
  run("setHTML(target, 'one'); setHTML(target, 'one');");
  assert.equal(writes, 1);
  target.contains = () => true;
  context.document.activeElement = {matches: () => true};
  run("setHTML(target, 'two');"); assert.equal(html, 'one');
  target.contains = () => false; target.querySelector = () => ({});
  run("setHTML(target, 'two');"); assert.equal(html, 'one');
  target.querySelector = () => null;
  run("setHTML(target, 'two');"); assert.equal(html, 'two');
});

test('新增记录插入时展开状态跟随记录 ID 而不是旧索引', () => {
  const {context, element, run} = harness();
  const target = element('target');
  function detail(id, open) {
    const record = {dataset: {record: id}, querySelectorAll: () => [item]};
    const item = {open, closest: () => record};
    return item;
  }
  let details = [detail('old', true)];
  target.querySelectorAll = (selector) => selector === 'details' ? details : [];
  Object.defineProperty(target, 'innerHTML', {set: () => {details = [detail('new', false), detail('old', false)];}});
  context.target = target; run("setHTML(target, 'new markup');");
  assert.equal(details[0].open, false); assert.equal(details[1].open, true);
});


test('完全访问和 Ultra 必须保留确认取消分支与明确的风险确认字段', async () => {
  for (const mode of ['full', 'ultra']) {
    const {context, element, run} = harness();
    const actions = element('agent-actions');
    const button = {dataset: {agentAction: 'permission'}, closest: () => actions};
    const event = {target: {closest: () => button}};
    element('agent-permission-mode').value = mode;
    context.window = {confirm: () => false};
    const posted = [];
    context.record = (url, body) => posted.push({url, body});
    run('postJson = async (url, body) => record(url, body); refreshAfterAction = async () => {};');
    await actions.listeners.click(event);
    assert.equal(posted.length, 0);
    assert.equal(actions.dataset.busy, undefined);
    context.window.confirm = () => true;
    await actions.listeners.click(event);
    assert.equal(posted.length, 1);
    assert.equal(posted[0].body.acknowledge_risk, true);
    assert.equal(posted[0].body.mode, mode);
  }
});


test('读取正文悬挂也会被超时中止，并解除刷新锁', async () => {
  const {context, element, run} = harness();
  context.setTimeout = (callback) => { queueMicrotask(callback); return 1; };
  context.clearTimeout = () => {};
  context.fetch = async (_url, {signal}) => ({ok: true, json: () => new Promise((resolve, reject) => {
    if (signal.aborted) reject(new Error('aborted'));
    else signal.addEventListener('abort', () => reject(new Error('aborted')));
  })});
  assert.equal(await run('refresh()'), false);
  assert.equal(run('refreshInFlight'), null);
  assert.equal(element('refresh-now').disabled, false);
});

test('角色标签来自实例，连接状态过期后不继续显示在线，未知发送状态不冒充关闭', () => {
  const {context, element, run} = harness();
  context.snapshot = {
    character: {id: 'child', name: '合成女儿'},
    agent: {diagnostics: {config: {send_enabled: true}}},
    napcat: {dispatcher_attached: true},
    qq_connection: {status: 'online', observed_at: new Date().toISOString(), received_events: 2}
  };
  run('renderOverview(snapshot)');
  assert.equal(context.document.title, '合成女儿 / 控制中枢');
  assert.equal(element('console-chat-title').textContent, '和合成女儿对话');
  assert.equal(element('.brand-name').textContent, '合成女儿');
  assert.match(element('instance-connection').textContent, /QQ 已登录，账号匹配.*消息接入：已装配.*QQ 发送：开启.*已接收事件：2/);
  context.snapshot.character = {id: 'parent', name: '合成母亲'};
  context.snapshot.qq_connection.observed_at = new Date(Date.now() - 31000).toISOString();
  context.snapshot.agent = {};
  run('renderOverview(snapshot)');
  assert.equal(element('character-code').textContent, 'PARENT');
  assert.equal(element('console-chat-title').textContent, '和合成母亲对话');
  assert.match(element('instance-connection').textContent, /QQ 连接状态待核对.*QQ 发送：未知/);
  assert.doesNotMatch(element('instance-connection').textContent, /QQ 已登录/);
});
