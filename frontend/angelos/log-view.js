'use strict';
/* 完整执行日志面板：只读展示已采集输出，不提交、不取消、不重跑任务。
 *
 * 与 activity.js 的分工：
 * - activity.js 负责每轮的工具调用列表（分组入口、状态、参数与回执摘要）；
 * - 本模块负责某一次调用的逐行日志：分页增量加载、搜索、复制、导出、
 *   跟随最新输出、向上翻阅时保持位置并提示新增、选区不被刷新破坏。
 *
 * 状态归属：
 * - 日志内容只来自服务端只读接口（分页 + 搜索 + 导出），本模块不缓存可写副本；
 * - 展示偏好（是否默认展开、面板宽度）由 layout.js 统一持久化；
 * - 页内展开/收起状态按"轮次:调用:流"记录在内存，刷新后按偏好恢复。
 */
window.AngelosLog = {
  create({root, api, esc, selectionInside, onChange}) {
    const $ = (id) => document.getElementById(id);
    // 每次调用的流式代理：键为 turnId::callId，值为 {streams:{}, complete, unknown}。
    const cache = new Map();
    let current = null; // {cid, turnId, callId}：cid 为会话 id，日志接口按会话归属读取
    let busy = false;

    function key(turnId, callId) { return turnId + '::' + callId; }

    function entry(turnId, callId) {
      const id = key(turnId, callId);
      if (!cache.has(id)) cache.set(id, {streams: {}, complete: null, unknown: [], status: '', exit: null, error: ''});
      return cache.get(id);
    }

    /* 把服务端返回的日志页合并进缓存；按 (stream, 行号) 去重，重连或重复拉取不重复追加。
     * 两个易错点（独立验收已用真实 Chromium 复现过）：
     * 1) 只有本次请求的那一个 stream 能写入自己的行。此前遍历 log.streams 对每个流都写入同一页，
     *    导致只有 stdout 内容时，stderr 缓存里也出现了 stdout 的行；
     * 2) 目标 entry 必须由调用方传入。若这里读全局 current，快速切换调用时旧响应会被写进新调用。 */
    function merge(log, target) {
      const item = target || entry(current.turnId, current.callId);
      item.status = log.call?.status || item.status;
      item.exit = log.call?.exit_code ?? item.exit;
      item.complete = log.complete;
      item.incompleteReasons = log.incomplete_reasons || [];
      item.unknown = log.unknown_fields || [];
      item.error = '';
      const segments = log.segments || [];
      const hasSegments = Array.isArray(log.segments);
      const stream = log.stream || '';
      // 先确保所有出现过的流都有桶（切页时下拉框要能列出全部流），但只往本页的流里写行。
      for (const name of (log.streams || hasSegments ? (log.streams || segments.map((s) => s.stream)) : [stream])) {
        if (!item.streams[name]) item.streams[name] = {lines: new Map(), total: 0, offset: 0, more: false};
      }
      if (!stream) return item;
      const bucket = item.streams[stream];
      if (!bucket) return item;
      const segment = segments.find((s) => s.stream === stream);
      bucket.total = log.total_lines ?? segment?.total_lines ?? bucket.total;
      // 分页按 offset 追加：同一 offset 重拉取时按行号覆写，不重复追加。
      for (const row of log.lines || []) bucket.lines.set(row.n, row);
      bucket.more = Boolean(log.has_more);
      bucket.offset = log.next_offset ?? bucket.lines.size;
      return item;
    }

    function ordered(stream) {
      const item = cache.get(key(current.turnId, current.callId));
      const bucket = item?.streams[stream];
      if (!bucket) return [];
      return [...bucket.lines.values()].sort((a, b) => a.n - b.n);
    }

    function activeStream() {
      const item = cache.get(key(current.turnId, current.callId));
      const picked = $('log-stream')?.value;
      if (picked && item?.streams[picked]) return picked;
      return Object.keys(item?.streams || {})[0] || '';
    }

    /* 判断用户是否贴着底部；向上翻阅时不强行滚回底部。 */
    function atBottom(node) { return node.scrollHeight - node.scrollTop - node.clientHeight < 48; }

    function nearbyElementSafe(node) {
      // 选区落在日志区域时不重绘节点，避免选中的文字被打断。
      return selectionInside(node);
    }

    function render({follow = true} = {}) {
      if (!current || !$('log-body')) return;
      const body = $('log-body');
      if (nearbyElementSafe(body)) return;
      const stream = activeStream();
      const lines = ordered(stream);
      const item = cache.get(key(current.turnId, current.callId)) || {};
      const stick = follow && atBottom(body);
      const top = body.scrollTop;
      const tabs = Object.keys(item.streams || {});
      const hasStream = tabs.length > 0;
      const bar = tabs.map((name) => `<option value="${esc(name)}"${name === stream ? ' selected' : ''}>${esc(labelOf(name))} · ${item.streams[name].lines.size}/${item.streams[name].total || '—'}</option>`).join('');
      if (!hasStream) { $('log-stream').innerHTML = '<option value="">无输出</option>'; }
      // 首次渲染时还没有任何流：不能对空 tab 列表求值，否则会读到 undefined。
      $('log-stream').innerHTML = bar || '<option value="">无输出</option>';
      const header = `<p class="log-meta">${esc(item.status || '状态未知')} · 退出码 ${item.exit === null || item.exit === undefined ? '未知' : esc(String(item.exit))}` +
        `${item.complete === false ? ' · 原件不可读，仅显示回执摘要' : item.complete ? ' · 完整原件' : ''}` +
        `${item.unknown?.length ? ' · 未知字段：' + esc(item.unknown.join('、')) : ''}</p>`;
      const rows = lines.map((row) => `<div class="log-line" data-n="${row.n}"><span class="log-gutter">${row.n}</span><span class="log-text">${esc(row.text.replace(/\n$/, ''))}</span></div>`).join('');
      body.innerHTML = header + (rows || `<p class="log-empty">${item.complete === false ? '没有可展示的输出片段。' : '暂无已采集输出；工具仍在运行或未产生输出。'}</p>`);
      $('log-more').hidden = !hasStream || !(item.streams[stream]?.more);
      $('log-count').textContent = lines.length + (hasStream && item.streams[stream]?.total ? ' / ' + item.streams[stream].total + ' 行' : ' 行');
      if (stick) body.scrollTop = body.scrollHeight; else body.scrollTop = top;
      changed();
    }

    function labelOf(stream) {
      return {stdout: '标准输出', stderr: '错误输出', error: '错误信息'}[stream] || stream;
    }

    function changed() { try { onChange?.(); } catch (_) {} }

    /* 日志接口的路径必须是 /execution-log/{turnId}/{callId}：后端按会话存储解析轮次再取调用，
       缺少 turnId 会被路由层判为 404，页面就会只剩“暂无已采集输出”。三个入口共用同一形态。 */
    function logPath(suffix = '') {
      return `/conversations/${encodeURIComponent(current.cid)}/execution-log/` +
        `${encodeURIComponent(current.turnId)}/${encodeURIComponent(current.callId)}${suffix}`;
    }

    async function page({reset = false} = {}) {
      if (!current || busy) return;
      // 请求发出前先钉住身份：响应回来时 current 可能已经指向别的调用。
      const token = {cid: current.cid, turnId: current.turnId, callId: current.callId};
      const item = entry(token.turnId, token.callId);
      const stream = reset ? (Object.keys(item.streams)[0] || 'stdout') : activeStream() || 'stdout';
      const bucket = item.streams[stream];
      const offset = reset ? 0 : (bucket?.offset || 0);
      busy = true;
      try {
        const data = await api(pathFor(token, `?stream=${encodeURIComponent(stream)}&offset=${offset}`));
        // 迟到响应一律丢弃：它属于上一个调用，不能改写当前面板，也不能污染新调用的缓存。
        if (!isCurrent(token)) return;
        merge(data.log, item);
        render();
      } catch (error) {
        if (!isCurrent(token)) return;
        item.error = String(error?.message || error || '日志读取失败');
        render({follow: false});
        $('log-status').textContent = item.error.includes('call_not_found')
          ? '本次调用未在当前离线路径中记录；可能属于此前尝试，历史回看需另行开启。'
          : '日志读取失败，保留已加载内容。';
      } finally {
        busy = false;
        $('log-status').textContent = $('log-status').textContent.startsWith('日志读取失败') ? $('log-status').textContent : '';
      }
    }

    function isCurrent(token) {
      return Boolean(current) && current.cid === token.cid
        && current.turnId === token.turnId && current.callId === token.callId;
    }

    /* 路径统一由传入的目标生成，不再隐式读 current：搜索/导出/翻页都靠它避免串页。 */
    function pathFor(token, suffix = '') {
      return `/conversations/${encodeURIComponent(token.cid)}/execution-log/` +
        `${encodeURIComponent(token.turnId)}/${encodeURIComponent(token.callId)}${suffix}`;
    }

    async function search(query) {
      if (!current) return;
      const term = (query || '').trim();
      const box = $('log-search-result');
      if (!term) { box.hidden = true; box.textContent = ''; return; }
      try {
        const data = await api(logPath(`/search?q=${encodeURIComponent(term)}`));
        const hits = data.log?.hits || [];
        box.hidden = false;
        box.innerHTML = `<strong>${data.log?.total_hits || 0} 处命中</strong>${data.log?.truncated ? '（仅列出前 ' + hits.length + ' 条）' : ''}` +
          hits.slice(0, 40).map((hit) => `<button type="button" class="log-hit" data-stream="${esc(hit.stream)}" data-n="${hit.n}">${esc(hit.stream)}:${hit.n} ${esc(hit.text.trim().slice(0, 120))}</button>`).join('');
      } catch (_) {
        box.hidden = false;
        box.textContent = '搜索失败，原日志不受影响。';
      }
    }

    /* 跳转到命中行：确保该页已加载，再把目标行滚动到可视区域。 */
    async function gotoHit(stream, n) {
      if (!current) return;
      const item = entry(current.turnId, current.callId);
      const bucket = item.streams[stream];
      while (bucket && !bucket.lines.has(n) && bucket.more) {
        $('log-stream').value = stream;
        await page();
        if (!bucket.lines.has(n) && !bucket.more) break;
      }
      $('log-stream').value = stream;
      render({follow: false});
      const row = $('log-body').querySelector(`.log-line[data-n="${n}"]`);
      if (row) { row.scrollIntoView({block: 'center'}); row.classList.add('log-hit-line'); }
    }

    function textOfCurrent() {
      const item = cache.get(key(current?.turnId, current?.callId));
      if (!item) return '';
      return Object.keys(item.streams).map((stream) => `===== ${labelOf(stream)} =====\n` +
        ordered(stream).map((row) => row.text).join('')).join('\n');
    }

    async function copyAll() {
      if (!current) return;
      const text = textOfCurrent();
      try {
        await navigator.clipboard.writeText(text);
        $('log-status').textContent = '已复制当前已加载的 ' + text.split('\n').length + ' 行。';
      } catch (_) {
        // 剪贴板不可用时退回选中文本，让用户自行复制；不静默失败。
        $('log-status').textContent = '浏览器拒绝了剪贴板写入，请选中文本后手动复制。';
      }
    }

    function exportAll() {
      if (!current) return;
      // 导出走服务端接口，取回全部已采集行而不是只导出已加载页。
      const url = '/angelos/api' + logPath('/export');
      const link = document.createElement('a');
      link.href = url; link.rel = 'noopener'; link.download = '';
      document.body.appendChild(link); link.click(); link.remove();
      $('log-status').textContent = '已请求导出完整日志；内容与页面使用同一脱敏结果。';
    }

    function show(cid, turnId, callId) {
      if (!cid || !turnId || !callId) return false;
      const switched = !current || current.cid !== cid
        || current.turnId !== turnId || current.callId !== callId;
      current = {cid, turnId, callId};
      root.hidden = false;
      $('log-title').textContent = callId;
      $('log-search-result').hidden = true;
      if (switched) { $('log-search').value = ''; }
      render({follow: switched});
      void page({reset: true});
      startPolling({cid, turnId, callId});
      return true;
    }

    function hide() { stopPolling(); root.hidden = true; current = null; }

    function bind() {
      $('log-close')?.addEventListener('click', () => { hide(); changed(); });
      $('log-stream')?.addEventListener('change', () => { render({follow: false}); void page(); });
      $('log-more')?.addEventListener('click', () => void page());
      $('log-body')?.addEventListener('scroll', () => {
        const body = $('log-body');
        $('log-jump').hidden = atBottom(body);
        $('log-follow').checked = atBottom(body);
      });
      $('log-jump')?.addEventListener('click', () => {
        const body = $('log-body');
        body.scrollTop = body.scrollHeight;
        $('log-new').hidden = true; $('log-jump').hidden = true;
        $('log-follow').checked = true;
      });
      $('log-search-button')?.addEventListener('click', () => void search($('log-search').value));
      $('log-search')?.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); void search($('log-search').value); } });
      $('log-search-result')?.addEventListener('click', (event) => {
        const hit = event.target.closest('.log-hit');
        if (hit) void gotoHit(hit.dataset.stream, Number(hit.dataset.n));
      });
      $('log-copy')?.addEventListener('click', () => void copyAll());
      $('log-export')?.addEventListener('click', exportAll);
    }

    /* 新输出到达时的提示：贴底则跟随，否则只提示有新内容，绝不移动阅读位置。 */
    function notice(count) {
      const node = $('log-new');
      if (!node) return;
      if ($('log-follow')?.checked && !selectionInside($('log-body'))) { $('log-jump').click(); }
      else if (count > 0) { node.textContent = `有 ${count} 行新输出`; node.hidden = false; }
    }

    /* 执行中的增量刷新。
     * 旧实现只有首屏一次请求：notice() 写好了却没人调用，也没有任何定时或事件触发，
     * 运行中的调用永远停在第一页，用户看不到新输出。这里用一个只在前台运行的轻量轮询：
     * - 只在面板可见、已选中调用、且上次返回的状态仍是 running 时才继续；
     * - 请求本身走已有的分页 offset，不会重复追加；
     * - 重新打开面板/切换调用会重启计时，关闭就清掉，不在后台空转；
     * - 不能只看 status：有些适配器在结束时才回写终态，所以未拿到终态就继续拉；
     *   拿到终态之前不会反复刷新已结束的调用。 */
    const POLL_MS = 2000;
    let poller = null;

    function shouldKeepPolling(token) {
      if (!isCurrent(token) || root.hidden) return false;
      const item = cache.get(key(token.turnId, token.callId));
      if (!item) return true;
      const status = String(item.status || '').toLowerCase();
      if (['ok', 'completed', 'failed', 'canceled', 'cancelled', 'not_executed', 'error'].includes(status)) return false;
      return true;
    }

    function startPolling(token) {
      stopPolling();
      poller = setInterval(async () => {
        if (!shouldKeepPolling(token) || busy) return;
        const item = cache.get(key(token.turnId, token.callId));
        const stream = $('log-stream')?.value || (item ? Object.keys(item.streams)[0] : '') || 'stdout';
        const bucket = item?.streams?.[stream];
        const before = bucket ? bucket.lines.size : 0;
        // 仍有未读页时优先补齐（more=true），否则去拉同一 offset 探新行。
        const offset = bucket && bucket.more ? bucket.offset : (bucket ? bucket.offset : 0);
        try {
          const data = await api(pathFor(token, `?stream=${encodeURIComponent(stream)}&offset=${offset}`));
          if (!isCurrent(token)) return;
          merge(data.log, item);
          const after = item.streams[stream]?.lines.size || 0;
          render();
          notice(Math.max(0, after - before));
        } catch (_) { /* 轮询失败不盖掉已加载内容，也不弹错；下一拍再试。 */ }
      }, POLL_MS);
    }

    function stopPolling() {
      if (poller) { clearInterval(poller); poller = null; }
    }

    bind();
    return {show, hide, render, merge, page, notice, search, copyAll, exportAll, current: () => current, entry, key,
            pollInterval: POLL_MS, startPolling, stopPolling};
  }
};
