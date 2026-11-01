"""附件原任务相对路径与旧版本校验；不以当前工作区或新文件替代旧回执。"""
import hashlib
import pytest
from shinku.angelos.artifact_file import read_artifact


def test_relative_artifact_and_versions(tmp_path):
    path=tmp_path/'workspace.js';path.write_bytes(b'new')
    receipt={'path':'workspace.js','sha256':hashlib.sha256(b'new').hexdigest()}
    assert read_artifact(receipt,{'path':str(tmp_path)})==('workspace.js',b'new')
    old={**receipt,'sha256':hashlib.sha256(b'old').hexdigest()}
    with pytest.raises(ValueError,match='artifact_changed_or_oversized'):read_artifact(old,{'path':str(tmp_path)})
    with pytest.raises(ValueError,match='artifact_unavailable'):read_artifact(receipt,{})
    assert read_artifact({**receipt,'path':str(path)}, {})==('workspace.js',b'new')


@pytest.mark.parametrize('name',['../workspace.js','sub/../../workspace.js','link','directory/workspace.js'])
def test_relative_path_escape_and_links_rejected(tmp_path,name):
    root=tmp_path/'root';root.mkdir();(tmp_path/'workspace.js').write_bytes(b'content')
    (root/'link').symlink_to(tmp_path/'workspace.js');(root/'directory').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='artifact_unavailable'):
        read_artifact({'path':name,'sha256':hashlib.sha256(b'content').hexdigest()},{'path':str(root)})
