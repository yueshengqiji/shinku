"""搜索扫描完整性及真实文件授权回归；使用合成项目，不访问正式数据。"""
import hashlib
import json

import pytest

from test_workspace_tools import setup, run, updates, context


def test_large_workspace_and_file_do_not_stop_before_target(setup):
    root, tools = setup
    # 一个超过旧4MiB单文件限制的文件，再加超过旧16MiB总量的资料。
    for n in range(5):
        (root / f'a{n}.txt').write_text('x' * (4 * 1024 * 1024) + '\n')
    target = root / 'z_project' / 'src' / 'learning'
    target.mkdir(parents=True)
    (target / 'plugin.py').write_text('class LearningPlugin: pass\n')
    result = updates(run(tools, 'search', {'query': 'learningplugin'}))
    assert result['scan_complete'] and not result['scan_bounded']
    assert result['scanned_bytes'] > 20 * 1024 * 1024
    assert result['matches'][0]['path'] == 'z_project/src/learning/plugin.py'
    assert result['skipped_counts'] == {}


def test_directory_enumeration_has_no_hidden_10001_cutoff(setup):
    root, tools = setup
    for n in range(10003):
        (root / f'{n:05}.txt').touch()
    result = updates(run(tools, 'search', {'query': '.txt', 'mode': 'path', 'limit': 1}))
    assert result['total_matches'] == 10003 and result['scan_complete']
    last = updates(run(tools, 'search', {'query': '.txt', 'mode': 'path', 'offset': 10002,
                                        'snapshot': result['snapshot']}))
    assert last['matches'] == [{'path': '10002.txt', 'kind': 'file'}]
    assert last['next_offset'] is None


def test_more_than_2000_matches_are_pageable_across_files(setup):
    root, tools = setup
    for name in ('a', 'b'):
        (root / name).write_text('hit\n' * 1500)
    first = updates(run(tools, 'search', {'query': 'hit', 'limit': 100}))
    assert first['total_matches'] == 3000 and len(first['matches']) == 100
    middle = updates(run(tools, 'search', {'query': 'hit', 'offset': 1499, 'limit': 3,
                                          'snapshot': first['snapshot']}))
    assert [(m['path'], m['line']) for m in middle['matches']] == [('a', 1500), ('b', 1), ('b', 2)]
    last = updates(run(tools, 'search', {'query': 'hit', 'offset': 2999, 'snapshot': first['snapshot']}))
    assert len(last['matches']) == 1 and last['next_offset'] is None


def test_large_line_chunk_boundary_unicode_and_line_numbers(setup, monkeypatch):
    from shinku.tools import workspace_search
    monkeypatch.setattr(workspace_search, 'CHUNK', 7)
    root, tools = setup
    raw = ('123456\r\n中文STRAßE\rthird\n' + 'x' * 1000 + 'STRASSE').encode()
    (root / 'unicode.txt').write_bytes(raw)
    result = updates(run(tools, 'search', {'query': 'strasse'}))
    assert [m['line'] for m in result['matches']] == [2, 4]
    assert result['matches'][1]['omitted_chars'] == 607
    assert all(m['sha256'] == hashlib.sha256(raw).hexdigest() for m in result['matches'])


@pytest.mark.parametrize('absolute', [False, True])
def test_subdirectory_case_and_path_discovery(setup, absolute):
    root, tools = setup
    (root / 'project' / 'Learning').mkdir(parents=True)
    (root / 'project' / 'Learning' / 'Plugin.py').write_text('LearningPlugin\n')
    (root / 'outside').write_text('LearningPlugin\n')
    path = str(root / 'project') if absolute else 'project'
    args = {'path': path, 'query': 'learningplugin'}
    result = updates(run(tools, 'search', args))
    assert len(result['matches']) == 1 and result['search_scope']['path'] == 'project'
    assert not updates(run(tools, 'search', {**args, 'ignore_case': False}))['matches']
    found = updates(run(tools, 'search', {'path': path, 'mode': 'path', 'query': 'learning'}))
    assert found['matches'][0] == {'path': 'project/Learning', 'kind': 'directory'}
    assert found['matches'][1]['kind'] == 'file'


@pytest.mark.parametrize('change', ['content', 'directory', 'case', 'path', 'query', 'glob'])
def test_snapshot_binds_scope_options_and_content(setup, change):
    root, tools = setup
    (root / 'sub').mkdir()
    (root / 'sub' / 'x').write_text('hit\nhit')
    args = {'query': 'hit'}
    first = updates(run(tools, 'search', args))
    if change == 'content': (root / 'sub' / 'x').write_text('hit\nhit\nhit')
    if change == 'directory': (root / 'added').write_text('hit')
    if change == 'case': args['ignore_case'] = False
    if change == 'path': args['path'] = 'sub'
    if change == 'query': args['query'] = 'hi'
    if change == 'glob': args['glob'] = '*.py'
    result = run(tools, 'search', {**args, 'offset': 1, 'snapshot': first['snapshot']})
    assert result.status == 'error' and 'search_snapshot_changed_restart' in str(result)


@pytest.mark.parametrize('path', ['../other', '/etc', '.git', 'private', 'private/sub', '.hidden', 'missing', 'link'])
def test_search_start_keeps_exclusions_and_no_symlink_escape(setup, path):
    root, tools = setup
    for name in ('.git', 'private', '.hidden'):
        (root / name).mkdir(); (root / name / 'secret').write_text('SECRET')
    outside = root.parent / (root.name + '-outside')
    outside.mkdir(); (outside / 'secret').write_text('SECRET')
    (root / 'link').symlink_to(outside, target_is_directory=True)
    result = run(tools, 'search', {'path': path, 'query': 'SECRET'})
    assert result.status == 'error' and 'SECRET' not in str(result)


def test_skip_diagnostics_are_compact_and_do_not_hide_later_match(setup):
    root, tools = setup
    for n in range(30): (root / f'a{n}').write_bytes(b'\x00binary')
    (root / 'z').write_text('hit')
    result = updates(run(tools, 'search', {'query': 'hit'}))
    assert result['matches'][0]['path'] == 'z'
    assert result['skipped_counts'] == {'binary_file': 30}
    assert len(result['skipped_examples']) == 5 and not result['scan_complete']
    assert result['search_status'] == 'completed_with_skips'


def test_revocation_mid_file_aborts_instead_of_returning_partial_success(setup, monkeypatch):
    root, tools = setup
    (root / 'x').write_text('hit\n' * 100000)
    access = tools['workspace_search'].access
    original = access.authorize
    calls = []
    def authorize(ctx, write):
        calls.append(1)
        return len(calls) < 7 and original(ctx, write)
    monkeypatch.setattr(access, 'authorize', authorize)
    result = run(tools, 'search', {'query': 'hit'})
    assert result.status == 'error' and 'workspace_not_authorized' in str(result)
    assert 'matches' not in result.data.get('state_updates', {})


def test_invalid_case_and_large_offset_validation(setup):
    _, tools = setup
    assert run(tools, 'search', {'query': 'x', 'ignore_case': 'false'}).status == 'error'
    assert tools['workspace_search'].normalize_call({'query': 'x', 'offset': 100001}) is not None


def test_real_workbench_routes_scoped_search_then_reads_target(tmp_path):
    from test_angelos_workspace_tools import workbench, conversation, send, call, final
    from test_angelos import wait
    root = tmp_path / 'project'; root.mkdir()
    (root / 'nested').mkdir(); (root / 'nested' / 'plugin.py').write_text('class LearningPlugin: pass\n')
    def respond(kw, count, service):
        if count == 1: return call('search', path='nested', query='learningplugin', ignore_case=True)
        if count == 2:
            assert 'nested/plugin.py' in kw['user_prompt']
            return call('read_lines', path='nested/plugin.py')
        assert 'class LearningPlugin: pass' in kw['user_prompt']
        return final()
    with workbench(tmp_path, respond) as (client, service, host, calls):
        cid = conversation(client, root); send(client, cid)
        result = wait(client, cid)['turns'][0]['result']
        assert result['execution']['status'] == 'completed'
        assert [e['tool'] for e in result['evidence']] == ['workspace_search', 'workspace_read_lines']
        assert len(calls) == 3 and host.permissions.level == 'he'
