"""在真实隔离文件系统验证分行、搜索、差异、版本冲突和原修改保留。"""
from copy import deepcopy
import hashlib

import pytest

from shinku.agent.request_scope import RequestScope
from shinku.tools.execution import ExecutionPolicy, execute_invocation
from shinku.tools.invocation import ToolInvocation
from shinku.tools.workspace import build_workspace_tools


def context(session='session', primary='turn', mode='full'):
    return {'agent_request_scope': RequestScope('owner', session, primary, ((primary, 'edit fixture'),)),
            'agent_permission_mode': mode}


@pytest.fixture
def setup(tmp_path):
    tools = build_workspace_tools(tmp_path, authorize=lambda ctx, write: ctx.get('agent_permission_mode') == 'full'
                                  or not write and ctx.get('agent_permission_mode') == 'read')
    return tmp_path, {t.tool_type: t for t in tools}


def run(tools, op, args, ctx=None, policy=None):
    _, envelope = execute_invocation(ToolInvocation('workspace_' + op, args), handlers=tools,
                                    context=ctx or context(), policy=policy)
    return envelope


def updates(envelope):
    assert envelope.status == 'ok', envelope.model_feedback
    return envelope.data['state_updates']


def preview(tools, path, old, new):
    read = updates(run(tools, 'read_lines', {'path': path}))
    return updates(run(tools, 'preview_patch', {'path': path, 'expected_sha256': read['sha256'],
                                               'edits': [{'old': old, 'new': new}]}))


def test_read_lines_preserves_bytes_and_partial_is_not_full(setup):
    from shinku.agent.local_answer import eligible_source
    root, tools = setup
    assert eligible_source(tools['workspace_read_lines']) and not eligible_source(tools['workspace_search'])
    raw = '已有修改\r\n中文🙂\r\n尾部\r\n'.encode()
    (root / 'x.py').write_bytes(raw)
    partial = updates(run(tools, 'read_lines', {'path': 'x.py', 'start_line': 2, 'line_count': 1}))
    assert partial['text'] == '中文🙂\r\n' and partial['start_line'] == partial['end_line'] == 2
    assert not partial['read_complete'] and partial['sha256'] == hashlib.sha256(raw).hexdigest()
    assert not tools['workspace_read_lines'].agent_verify_result(receipt=partial, context=context())['result_valid']
    full = updates(run(tools, 'read_lines', {'path': 'x.py'}))
    assert full['read_complete'] and tools['workspace_read_lines'].agent_verify_result(receipt=full, context=context())['result_valid']
    assert (root / 'x.py').read_bytes() == raw


def test_absolute_read_has_same_canonical_receipt_without_mutating_arguments(setup):
    root, tools = setup
    (root / 'nested').mkdir()
    path = root / 'nested' / '中文.py'
    path.write_bytes('原文\r\n第二行\r\n'.encode())
    args = {'path': str(path), 'start_line': 1, 'line_count': 400}
    original = deepcopy(args)
    absolute = updates(run(tools, 'read_lines', args, context(mode='read')))
    relative = updates(run(tools, 'read_lines', {**args, 'path': 'nested/中文.py'}, context(mode='read')))
    assert args == original and absolute == relative
    assert absolute['path'] == 'nested/中文.py' and absolute['source_path'] == str(path)
    assert absolute['completion_targets'] == ['nested/中文.py', str(path)]
    handler = tools['workspace_read_lines']
    assert handler.agent_verify_result(receipt=absolute, context=context(mode='read'))['result_valid']
    partial = updates(run(tools, 'read_lines', {**args, 'line_count': 1}))
    assert not handler.agent_verify_result(receipt=partial, context=context())['result_valid']
    path.write_text('外部修改')
    assert not handler.agent_verify_result(receipt=absolute, context=context())['result_valid']


@pytest.mark.parametrize('kind', ['sibling_prefix', 'parent', 'root', 'git', 'file_link', 'directory_link', 'denied'])
def test_absolute_read_keeps_scope_symlink_and_permission_boundaries(setup, kind):
    root, tools = setup
    sibling = root.parent / (root.name + '-sibling')
    sibling.mkdir(); (sibling / 'secret').write_text('OUTSIDE_SECRET')
    (root / 'safe').write_text('INSIDE_SECRET')
    (root / '.git').mkdir(); (root / '.git/config').write_text('GIT_SECRET')
    (root / 'link').symlink_to(sibling / 'secret')
    (root / 'linked').symlink_to(sibling, target_is_directory=True)
    paths = {'sibling_prefix': str(sibling / 'secret'), 'parent': str(root) + '/../' + sibling.name + '/secret',
             'root': str(root), 'git': str(root / '.git/config'), 'file_link': str(root / 'link'),
             'directory_link': str(root / 'linked/secret'), 'denied': str(root / 'safe')}
    result = run(tools, 'read_lines', {'path': paths[kind]}, context(mode='none' if kind == 'denied' else 'read'))
    assert result.status == 'error' and 'SECRET' not in str(result)


def test_absolute_directory_swapped_to_symlink_after_canonicalization_is_denied(setup, monkeypatch):
    root, tools = setup
    (root / 'nested').mkdir(); (root / 'nested/file').write_text('original')
    outside = root.parent / (root.name + '-external')
    outside.mkdir(); (outside / 'file').write_text('OUTSIDE_SECRET')
    access = tools['workspace_read_lines'].access
    canonicalize = access.relative_read_path
    def swapped(path):
        relative = canonicalize(path)
        (root / 'nested').rename(root / 'moved')
        (root / 'nested').symlink_to(outside, target_is_directory=True)
        return relative
    monkeypatch.setattr(access, 'relative_read_path', swapped)
    result = run(tools, 'read_lines', {'path': str(root / 'nested/file')})
    assert result.status == 'error' and 'OUTSIDE_SECRET' not in str(result)


def test_absolute_read_does_not_expand_preview_or_apply_protocol(setup):
    root, tools = setup
    path = root / 'x.py'; path.write_text('value=1')
    plan = preview(tools, 'x.py', 'value=1', 'value=2')
    for op, extra in [('preview_patch', {'edits': [{'old': 'value=1', 'new': 'value=2'}]}),
                      ('apply_patch', {'preview_id': plan['preview_id']})]:
        result = run(tools, op, {'path': str(path), 'expected_sha256': plan['sha256'], **extra})
        assert result.status == 'error' and path.read_text() == 'value=1'


def test_search_paging_versions_exclusions_and_invalidated_cursor(setup):
    root, tools = setup
    (root / 'a.py').write_text('hit\nhit\n中文\n')
    (root / 'b.py').write_text('hit\n')
    (root / 'private').mkdir(); (root / 'private/secret').write_text('hit SECRET')
    (root / '.env').write_text('hit SECRET')
    first = updates(run(tools, 'search', {'query': 'hit', 'limit': 1}))
    assert first['scan_complete'] and first['next_offset'] == 1
    second = updates(run(tools, 'search', {'query': 'hit', 'offset': 1, 'snapshot': first['snapshot']}))
    assert len(second['matches']) == 2 and all('SECRET' not in str(r) for r in second['matches'])
    (root / 'b.py').write_text('different')
    assert run(tools, 'search', {'query': 'hit', 'offset': 1, 'snapshot': first['snapshot']}).status == 'error'


def test_observation_cannot_be_forged_or_reused_in_another_request(setup):
    root, tools = setup
    (root/'a').write_text('hit')
    data = updates(run(tools, 'search', {'query':'hit'}))
    handler = tools['workspace_search']
    assert handler.agent_verify_result(receipt=data, context=context())['result_valid']
    assert not handler.agent_verify_result(receipt=data, context=context(primary='other'))['result_valid']
    forged = deepcopy(data)
    forged['observation']['self_state']['data_sha256'] = 'forged'
    assert not handler.agent_verify_result(receipt=forged, context=context())['result_valid']


def test_directory_swap_to_symlink_during_search_does_not_enumerate_outside(setup, monkeypatch):
    root, tools = setup
    (root/'nested').mkdir(); (root/'nested'/'inside').write_text('fixture')
    outside = root.parent / ('external-' + root.name)
    outside.mkdir(); (outside/'SECRET_FILENAME').write_text('fixture')
    access = tools['workspace_search'].access
    original = access.entries
    def entries(parts):
        rows = original(parts)
        if parts == ():
            (root/'nested').rename(root/'moved')
            (root/'nested').symlink_to(outside, target_is_directory=True)
        return rows
    monkeypatch.setattr(access, 'entries', entries)
    result = updates(run(tools, 'search', {'query':'SECRET', 'mode':'path'}))
    assert result['matches'] == [] and not result['scan_complete'] and 'SECRET_FILENAME' not in str(result)


def test_patch_preserves_existing_changes_and_non_target_file(setup):
    root, tools = setup
    (root / 'x.py').write_bytes(b'user edit\r\nvalue=1\r\nend\r\n')
    (root / 'other').write_bytes(b'unrelated')
    p = preview(tools, 'x.py', 'value=1', 'value=2')
    assert '-value=1' in p['diff'] and '+value=2' in p['diff'] and not p['applied']
    assert b'value=1' in (root / 'x.py').read_bytes()
    args = {'path': 'x.py', 'expected_sha256': p['sha256'], 'preview_id': p['preview_id']}
    receipt = updates(run(tools, 'apply_patch', args))
    assert receipt['applied'] and (root / 'x.py').read_bytes() == b'user edit\r\nvalue=2\r\nend\r\n'
    assert (root / 'other').read_bytes() == b'unrelated'
    assert tools['workspace_apply_patch'].agent_verify_result(receipt=receipt, context=context())['result_valid']
    assert run(tools, 'apply_patch', args).status == 'error'
    (root / 'x.py').write_text('changed after write')
    assert not tools['workspace_apply_patch'].agent_verify_result(receipt=receipt, context=context())['result_valid']


def test_multifile_partial_failure_keeps_independent_receipts_and_user_changes(setup):
    root, tools = setup
    for name in ('a.py','b.py'):
        (root/name).write_text('# existing edit\ntimeout=10\n')
    plans = {name:preview(tools,name,'timeout=10','timeout=30') for name in ('a.py','b.py')}
    (root/'b.py').write_text('# concurrent user edit\ntimeout=20\n')
    results = {name:run(tools,'apply_patch', {'path':name,'expected_sha256':p['sha256'],'preview_id':p['preview_id']})
               for name,p in plans.items()}
    assert results['a.py'].status == 'ok' and results['b.py'].status == 'error'
    assert (root/'a.py').read_text() == '# existing edit\ntimeout=30\n'
    assert (root/'b.py').read_text() == '# concurrent user edit\ntimeout=20\n'


@pytest.mark.parametrize('change', ['concurrent', 'wrong_session', 'new_turn', 'revoked', 'blocked'])
def test_apply_rejects_changed_version_scope_permission_and_dispatch_policy(setup, change):
    root, tools = setup
    (root / 'x').write_text('old')
    p = preview(tools, 'x', 'old', 'new')
    args = {'path': 'x', 'expected_sha256': p['sha256'], 'preview_id': p['preview_id']}
    ctx, policy = context(), None
    if change == 'concurrent': (root / 'x').write_text('user concurrent edit')
    if change == 'wrong_session': ctx = context(session='other')
    if change == 'new_turn': ctx = context(primary='other')
    if change == 'revoked': ctx = context(mode='read')
    if change == 'blocked': policy = ExecutionPolicy(blocked=frozenset({'workspace_apply_patch'}))
    before = (root / 'x').read_bytes()
    assert run(tools, 'apply_patch', args, ctx, policy).status == 'error'
    assert (root / 'x').read_bytes() == before


@pytest.mark.parametrize('path', ['../outside', '/etc/passwd', '.git/config', 'link', 'dirlink/file'])
def test_no_path_or_symlink_escape(setup, tmp_path, path):
    root, tools = setup
    outside = root.parent / ('outside-' + root.name)
    outside.mkdir(); (outside / 'file').write_text('SECRET')
    (root / 'link').symlink_to(outside / 'file'); (root / 'dirlink').symlink_to(outside, target_is_directory=True)
    result = run(tools, 'read_lines', {'path': path})
    assert result.status == 'error' and 'SECRET' not in str(result)
    found = updates(run(tools, 'search', {'query': 'SECRET'}))
    assert found['matches'] == []


def test_multiple_matches_and_non_utf8_do_not_overwrite(setup):
    root, tools = setup
    (root / 'x').write_text('same same')
    before = (root / 'x').read_bytes()
    with pytest.raises(AssertionError): preview(tools, 'x', 'same', 'new')
    assert (root / 'x').read_bytes() == before
    (root / 'x').write_bytes(b'\xff\x00')
    assert run(tools, 'read_lines', {'path': 'x'}).status == 'error'


def test_real_agent_service_completion_requires_current_written_artifact(setup, monkeypatch):
    from shinku.agent_runtime.demo import demo_service
    from shinku.agent_runtime import AgentRequest, AgentScope
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from dataclasses import replace
    from test_agent_runtime_boundary import settled
    root, tools = setup
    (root / 'x.py').write_text('# user edit\nvalue=1\n')
    service, _, runtime = demo_service(root)
    service.executor.tool_host = ToolHost(ToolRegistry(tools))
    original_authorize = service.authorize
    service.authorize = lambda *args: replace(original_authorize(*args), permission_mode='full',
        allowed_tools=frozenset(tools), policy=ExecutionPolicy(risk_by_tool={n:'low' for n in tools}))
    counter = []
    def respond(**kwargs):
        counter.append(kwargs)
        if len(counter) == 1:
            return {'tool_call': {'type': 'workspace_read_lines', 'path': 'x.py'}}
        if len(counter) == 2:
            return {'tool_call': {'type': 'workspace_preview_patch', 'path': 'x.py',
                    'expected_sha256': hashlib.sha256((root/'x.py').read_bytes()).hexdigest(),
                    'edits': [{'old': 'value=1', 'new': 'value=2'}]}}
        if len(counter) == 3:
            identifier, plan = next(iter(tools['workspace_apply_patch'].access.previews.items()))
            return {'tool_call': {'type': 'workspace_apply_patch', 'path': 'x.py', 'expected_sha256': plan[2], 'preview_id': identifier}}
        return {'kind': 'final', 'text': '已保留原有注释并更新值，实际文件核对通过。', 'task_outcome': 'completed'}
    monkeypatch.setattr(runtime, 'call_chat_json', respond)
    port = service.bind(AgentScope('owner', 'local:fixture'))
    contract = {'goal':'edit fixture','deliverables':['x.py'],'scope':[str(root)],
                'requirements':[{'id':'written','description':'value updated','effect':'write',
                                 'targets':[str(root/'x.py')], 'checks':{'contains':'value=2'}}]}
    try:
        handle = port.submit(AgentRequest('edit', '把x.py的value改为2，保留原注释', contract))
        view = settled(port, handle.task_id)
        result = port.result(handle.task_id)
        assert view['status'] == 'completed' and result['verification']['accepted'], (view, result)
        assert (root/'x.py').read_text() == '# user edit\nvalue=2\n' and len(counter) == 4
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()
