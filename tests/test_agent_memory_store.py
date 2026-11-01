"""独立数据库、FTS、策略和忘记边界，合成资料不能污染 Bot。"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sqlite3

import pytest

from shinku.agent_memory.store import AgentMemoryStore
from shinku.agent_memory.retrieval import AgentMemoryPolicy


def put(store, text='白鸽客户端 2.1.7 proxy', *, owner='owner', workspace='w', key='one'):
    return store.put(principal=owner, workspace=workspace, key=key,
                     kind='user_statement', text=text, source={'type': 'synthetic'})


def test_agent_and_bot_files_indices_and_mutations_are_independent(tmp_path):
    bot = tmp_path/'bot.sqlite3'
    with sqlite3.connect(bot) as db:
        db.execute('CREATE TABLE bot_memory(text)'); db.execute("INSERT INTO bot_memory VALUES ('BOT_ONLY_91')")
    before = bot.read_bytes()
    agent = AgentMemoryStore(tmp_path/'agent'/'memory')
    row = put(agent)
    assert agent.search('owner', 'w', 'BOT_ONLY_91') == []
    assert agent.search('owner', 'w', '白鸽')[0]['id'] == row['id']
    agent.forget('owner', 'w', row['id'])
    assert bot.read_bytes() == before
    with sqlite3.connect(bot) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='memory_fts'").fetchone() is None


@pytest.mark.parametrize('owner,workspace', [('other','w'), ('owner','other')])
def test_scope_is_sql_filter_not_post_search_guess(tmp_path, owner, workspace):
    store = AgentMemoryStore(tmp_path/'memory'); row = put(store)
    assert store.search(owner, workspace, '白鸽') == []
    assert store.get(owner, workspace, row['id']) is None
    with pytest.raises(ValueError, match='not_found'):
        store.forget(owner, workspace, row['id'])
    assert store.get('owner','w',row['id'])


@pytest.mark.parametrize('kind', ['foreign', 'symlink', 'hardlink', 'permissions', 'parent_symlink'])
def test_wrong_database_never_opened_as_agent_or_repaired(tmp_path, kind):
    root = tmp_path/'memory'; root.mkdir()
    path = root/'agent-memory.sqlite3'
    foreign = tmp_path/'bot.sqlite3'
    with sqlite3.connect(foreign) as db:
        db.execute('CREATE TABLE bot_private(value)'); db.execute("INSERT INTO bot_private VALUES ('BOT_SECRET')")
    foreign.chmod(0o600); before = foreign.read_bytes()
    if kind == 'symlink': path.symlink_to(foreign)
    elif kind == 'hardlink': os.link(foreign,path)
    elif kind == 'parent_symlink':
        alias = tmp_path/'alias';alias.symlink_to(root,target_is_directory=True);root=alias
    else:
        path.write_bytes(before);path.chmod(0o644 if kind == 'permissions' else 0o600)
    with pytest.raises(ValueError): AgentMemoryStore(root)
    assert foreign.read_bytes() == before
    if kind == 'foreign': assert path.read_bytes() == before


def test_restart_idempotence_and_forget_tombstone(tmp_path):
    root=tmp_path/'memory';store=AgentMemoryStore(root); first=put(store)
    again=AgentMemoryStore(root);assert put(again)['status']=='unchanged'
    assert again.search('owner','w','白鸽')[0]['id']==first['id']
    again.forget('owner','w',first['id'])
    assert put(AgentMemoryStore(root))['status']=='forgotten'
    assert again.search('owner','w','白鸽')==[]
    with sqlite3.connect(again.path) as db:
        assert db.execute('SELECT payload FROM memories').fetchone()[0]=='{}'


@pytest.mark.parametrize('query', ['" OR *', '白鸽" OR NEAR(secret)', '2.1.7', 'PROXY', '白鸽'])
def test_fts_input_is_data_and_case_folded(tmp_path,query):
    store=AgentMemoryStore(tmp_path/'memory');put(store)
    rows=store.search('owner','w',query)
    assert isinstance(rows,list)
    if query in ['2.1.7','PROXY','白鸽']:assert rows


@pytest.mark.parametrize('value', [0,True,-1,100000])
def test_policy_bounds_are_separate_and_validated(value):
    with pytest.raises(ValueError):AgentMemoryPolicy(max_results=value)


def test_no_recent_window_hides_matching_old_record(tmp_path):
    store=AgentMemoryStore(tmp_path/'memory');put(store,text='ancient_special_73',key='old')
    for i in range(70):put(store,text=f'unrelated_{i}',key=f'n{i}')
    assert store.search('owner','w','ancient_special_73')[0]['text']=='ancient_special_73'


def test_concurrent_writes_and_read_does_not_refresh_retention(tmp_path):
    store=AgentMemoryStore(tmp_path/'memory')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda i:put(store,text='client',key=str(i)),range(20)))
    before=store.path.read_bytes()
    assert store.search('owner','w','client')
    assert store.path.read_bytes()==before
    assert len({r['id'] for r in rows})==20


def test_record_tampering_is_rejected(tmp_path):
    store=AgentMemoryStore(tmp_path/'memory');row=put(store)
    with sqlite3.connect(store.path) as db:db.execute("UPDATE memories SET payload='{}'")
    with pytest.raises(ValueError,match='integrity'):store.search('owner','w','白鸽')
    with pytest.raises(ValueError,match='integrity'):store.get('owner','w',row['id'])


def test_bot_directory_cannot_be_used_as_default_agent_location(tmp_path):
    root=tmp_path/'bot-memory';root.mkdir();(root/'bot.sqlite3').write_bytes(b'BOT_DATA')
    with pytest.raises(ValueError,match='not_dedicated'):AgentMemoryStore(root)
    assert not (root/'agent-memory.sqlite3').exists()
