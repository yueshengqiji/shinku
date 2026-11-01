"""工作区真实磁盘变化、版本冲突和文件边界；原件/编辑回执独立于 Agent。"""
import os
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
import pytest
from shinku.angelos.store import WorkbenchStore
from shinku.angelos.files import WorkspaceFiles,digest
from shinku.angelos.permissions import PermissionControls
from shinku.angelos.app import create_angelos_app


@pytest.fixture
def files(tmp_path):
    root=tmp_path/'project';root.mkdir()
    store=WorkbenchStore(tmp_path/'data'/'workbench.sqlite3')
    workspace=store.workspace({'name':'测试','path':str(root)})
    permissions=PermissionControls(tmp_path/'permissions.json',default='waw')
    return WorkspaceFiles(store,permissions),workspace['id'],root


def test_snapshot_add_change_delete_and_edit(files):
    f,w,r=files;(r/'a.py').write_text('class Old:\n    pass\n');(r/'deleted.txt').write_text('gone\n')
    assert f.changes(w)['files']==[]
    (r/'a.py').write_text('class New:\n    def call(self):\n        return 1\n')
    (r/'added.js').write_text('const x = 1;\n');(r/'deleted.txt').unlink()
    rows={x['path']:x for x in f.changes(w)['files']}
    assert rows['added.js']['added']==1 and rows['deleted.txt']['removed']==1
    assert rows['a.py']['added']==3 and rows['a.py']['removed']==2
    old=f.open(w,'a.py');assert old['original']=='class Old:\n    pass\n'
    new=f.save(w,'a.py','class Final:\n    pass\n',old['version'])
    assert new['content']==(r/'a.py').read_text()
    assert f.store.list('manual_file_edit')[0]['before']==old['content']


def test_conflict_preserves_external_edit(files):
    f,w,r=files;(r/'a.py').write_text('one')
    old=f.open(w,'a.py');(r/'a.py').write_text('external')
    with pytest.raises(ValueError,match='file_version_conflict'):
        f.save(w,'a.py','overwrite',old['version'])
    assert (r/'a.py').read_text()=='external'


def test_workspace_boundary_and_nonregular_files(files,tmp_path):
    f,w,r=files;outside=tmp_path/'outside';outside.write_text('private')
    (r/'link').symlink_to(outside)
    (r/'directory-link').symlink_to(tmp_path,target_is_directory=True)
    for name in ('../outside',str(outside),'link','directory-link/outside','.git/config'):
        with pytest.raises((ValueError,OSError)):
            f.open(w,name)
    (r/'binary').write_bytes(b'a\0b')
    with pytest.raises(ValueError,match='file_binary_or_oversized'):
        f.open(w,'binary')
    (r/'big').write_bytes(b'a'*(512*1024+1))
    with pytest.raises(ValueError,match='file_binary_or_oversized'):
        f.open(w,'big')


def test_hardlink_and_permission_downgrade_cannot_write(files,tmp_path):
    f,w,r=files;(r/'a').write_text('one');os.link(r/'a',tmp_path/'alias')
    old=f.open(w,'a')
    with pytest.raises(ValueError,match='file_hardlink_not_editable'):
        f.save(w,'a','two',old['version'])
    f.permissions.select('he',f.permissions.revision)
    with pytest.raises(PermissionError,match='file_edit_not_permitted'):
        f.save(w,'a','two',old['version'])


def test_git_changes_are_against_head_including_staged_and_untracked(files):
    f,w,r=files
    def git(*args):
        return subprocess.run(['git','-C',str(r),*args],check=True,capture_output=True)
    git('init','-q');(r/'a.py').write_text('x=1\n')
    git('add','a.py');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','base')
    assert f.changes(w)['baseline']=='HEAD'
    (r/'a.py').write_text('x=2\ny=3\n');git('add','a.py');(r/'new file.py').write_text('pass\n')
    changes=f.changes(w)
    assert changes['added']==3 and changes['removed']==1
    assert f.open(w,'a.py')['original']=='x=1\n'


def test_file_http_origin_token_and_oversized_body(tmp_path):
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    service,_,_=demo_service(tmp_path)
    permissions=PermissionControls(tmp_path/'permissions.json',default='waw')
    host=ServiceHost(service,permissions=permissions)
    app=create_angelos_app(tmp_path/'data',host=host)
    r=tmp_path/'project';r.mkdir();(r/'source.py').write_text('original')
    w=app.state.angelos.store.workspace({'name':'编辑 API','path':str(r)})['id']
    url='/angelos/api/workspaces/'+w+'/file'
    try:
        with TestClient(app) as client:
            row=client.get(url,params={'path':'source.py'}).json()['file']
            payload={'path':'source.py','content':'# '+('x'*100000),'version':row['version']}
            assert client.post(url,json=payload).status_code==403
            client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
            assert client.post(url,json=payload,headers={'origin':'https://evil.invalid'}).status_code==403
            saved=client.post(url,json=payload)
            assert saved.status_code==200, saved.text
            assert (r/'source.py').read_text()==payload['content']
            assert client.post(url,json=payload).json()['reason']=='file_version_conflict'
            assert client.get(url,params={'path':'../permissions.json'}).status_code==409
            assert client.get('/angelos/api/workspaces/'+w+'/changes').status_code==200
    finally:
        service.coordinator.shutdown(wait=True)
