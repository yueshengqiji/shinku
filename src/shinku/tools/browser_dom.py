"""有界页面观察和真实元素身份：不把 CSS 下标当成操作授权。

Playwright locator 用于发现可见语义控件（含开放 Shadow DOM）；本次观察保存
ElementHandle，动作前验证它仍属于原文档且语义/表单未变。iframe 独立授权。
网页内容、标签和链接均为不可信资料，不读取密码值，不执行模型给出的脚本。
"""
from dataclasses import dataclass, field
import hashlib
import json
import uuid


CONTROLS = ('button,input:not([type=hidden]),select,textarea,a[href],summary,'
            '[role=button],[role=link],[role=checkbox],[role=radio],[role=combobox],'
            '[role=textbox],[role=tab],[role=menuitem],[contenteditable=true]')
MAX_TARGETS = 2000
MAX_FRAMES = 32

# _guard 只保存在宿主，模型只收到短预览。表单所有字段共同约束提交按钮，
# 避免观察后被外部编辑的表单仍以旧授权提交；超大表单不假装已完整核对。
ELEMENT_INFO = r"""(e, withGuard=true) => {
 const text = v => String(v||'').replace(/\s+/g,' ').trim();
 const root = e.getRootNode();
 const named = (e.getAttribute('aria-labelledby')||'').split(/\s+/)
   .map(id=>root.getElementById?.(id)?.textContent||'').join(' ');
 const label = text(named || e.getAttribute('aria-label') ||
   [...(e.labels||[])].map(n=>n.textContent).join(' ') || e.innerText ||
   e.getAttribute('placeholder') || e.getAttribute('title'));
 const tag=e.tagName.toLowerCase(), type=e.getAttribute('type')||'';
 const value=n=>n.type==='password'?'[redacted]':n.type==='file'?
   [...(n.files||[])].map(f=>({name:f.name,size:f.size})):String(n.value||'');
 const form=e.form||e.closest('form');
 const guard=withGuard!==false?JSON.stringify({tag,type,label,role:e.getAttribute('role')||'',
   href:e.href||'',value:value(e),checked:!!e.checked,disabled:!!e.disabled,
   form:form?{action:form.action,method:form.method,
     fields:[...form.elements].filter(n=>n.type!=='hidden').map(n=>
       ({name:n.name,type:n.type,value:value(n),checked:!!n.checked,disabled:!!n.disabled}))}:null}):'';
 const style=getComputedStyle(e);
 return {tag,type,role:e.getAttribute('role')||({button:'button',a:'link',textarea:'textbox',select:'combobox'}[tag]||''),
   label:label.slice(0,200),value:String(value(e)).slice(0,200),href:String(e.href||'').slice(0,2000),
   disabled:!!e.disabled,checked:!!e.checked,visible:!!e.getClientRects().length && style.visibility!=='hidden' && style.display!=='none',
   _guard:guard.length<=65536?guard:'',guard_complete:guard.length<=65536};
}"""
PAGE_INFO = r"""() => ({url:location.href,title:document.title,
 text:document.body?.innerText||'',
 status:[...document.querySelectorAll('output,[role=status],[role=alert]')]
   .filter(e=>e.getClientRects().length).map(e=>e.innerText).join('\n')})"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class Target:
    handle: object
    frame: object
    guard: str
    data: dict

    def resolve(self, *, strict=True):
        try:
            if not self.handle.evaluate('e => e.isConnected'):
                raise ValueError('browser_target_changed')
            current = self.handle.evaluate(ELEMENT_INFO)
            if strict and (not current['guard_complete'] or digest(current['_guard']) != self.guard):
                raise ValueError('browser_target_changed')
            return self.handle
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('browser_target_changed') from exc


@dataclass
class Snapshot:
    page: object
    document: object
    data: dict
    status: str
    targets: dict = field(default_factory=dict)
    binding: tuple = ()

    def check_document(self):
        try:
            if self.page.url != self.data['url'] or not self.document.evaluate('e => e.isConnected'):
                raise ValueError('browser_document_changed_observe_again')
        except Exception as exc:
            raise ValueError('browser_document_changed_observe_again') from exc

    def dispose(self):
        for handle in [self.document, *(t.handle for t in self.targets.values())]:
            try:
                handle.dispose()
            except Exception:
                pass


def capture(page, *, frames, check, target_offset=0, text_offset=0, frame_id='', query=''):
    """返回分页预览和仅存于宿主的句柄；越界或中途变化不签发完整快照。"""
    available = [(identifier, frame) for identifier, frame in frames if not frame.is_detached()]
    if frame_id and frame_id not in {ident for ident, _ in available}:
        raise ValueError('browser_frame_not_observed')
    selected = next((frame for ident, frame in available if ident == frame_id), page.main_frame)
    for _, frame in available:
        check(frame.url)
    document = page.query_selector('html')
    if document is None:
        raise ValueError('browser_document_unavailable')
    result = Snapshot(page, document, {}, '')
    try:
        info = selected.evaluate(PAGE_INFO)
        if text_offset > len(info['text']):
            raise ValueError('browser_text_offset_out_of_range')
        # 分帧保留状态输出摘要，普通时钟/正文变化不影响目标操作。
        statuses, candidates, complete = [], [], len(available) <= MAX_FRAMES
        for ident, frame in available[:MAX_FRAMES]:
            statuses.append((ident, frame.evaluate(PAGE_INFO)['status']))
            if frame_id and ident != frame_id:
                continue
            locator = frame.locator(CONTROLS)
            rows = locator.evaluate_all('(es) => es.slice(0,' + str(MAX_TARGETS + 1) + ').map(e=>(' + ELEMENT_INFO + ')(e,false))')
            if len(rows) > MAX_TARGETS:
                complete = False
            for index, row in enumerate(rows[:MAX_TARGETS]):
                if row['visible'] and (not query or query.casefold() in row['label'].casefold()):
                    candidates.append((ident, frame, locator, index, row))
            if len(candidates) > MAX_TARGETS:
                complete = False
                candidates = candidates[:MAX_TARGETS]
                break
        if target_offset > len(candidates):
            raise ValueError('browser_target_offset_out_of_range')
        public = []
        for offset, (ident, frame, locator, index, row) in enumerate(candidates[target_offset:target_offset + 100]):
            handle = locator.nth(index).element_handle(timeout=1000)
            if handle is None:
                raise ValueError('browser_target_changed')
            ref = 'e' + str(target_offset + offset + 1)
            data = {k: v for k, v in row.items() if k != '_guard'} | {'ref': ref, 'frame_id': ident}
            actual = handle.evaluate(ELEMENT_INFO)
            if any(actual[k] != row[k] for k in row if k not in {'_guard', 'guard_complete'}):
                handle.dispose()
                raise ValueError('browser_target_changed')
            data['guard_complete'] = actual['guard_complete']
            target = Target(handle, frame, digest(actual['_guard']), data)
            result.targets[ref] = target
            target.resolve(strict=False)
            public.append(data)
        target_end = target_offset + len(public)
        text = info['text'][text_offset:text_offset + 16000]
        text_end = text_offset + len(text)
        result.status = digest(statuses)
        result.data = {'state_id': uuid.uuid4().hex, 'url': page.url, 'title': page.title(),
            'text': text, 'text_offset': text_offset, 'total_text_chars': len(info['text']),
            'next_text_offset': text_end if text_end < len(info['text']) else None,
            'page_text_complete': text_offset == 0 and text_end == len(info['text']),
            'targets': public, 'target_offset': target_offset, 'total_targets': len(candidates),
            'targets_complete': complete and target_offset == 0 and target_end == len(candidates),
            'target_scan_complete': complete,
            'next_target_offset': target_end if target_end < len(candidates) else None,
            'frames': [{'frame_id': ident, 'url': frame.url, 'name': frame.name} for ident, frame in available[:MAX_FRAMES]],
            'frames_complete': len(available) <= MAX_FRAMES,
            'selected_frame_id': next(ident for ident, frame in available if frame is selected)}
        result.check_document()
        return result
    except Exception:
        result.dispose()
        raise


def status_digest(frames):
    return digest([(ident, frame.evaluate(PAGE_INFO)['status']) for ident, frame in frames[:MAX_FRAMES]
                   if not frame.is_detached()])
