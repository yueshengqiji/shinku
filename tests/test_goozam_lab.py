"""实验副本必须来自固定公开提交，不能携带本机秘密或连接正式数据。"""
import importlib.util
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location('prepare_goozam_lab',
    Path(__file__).resolve().parents[1] / 'scripts/prepare_goozam_lab.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    for name in lab.EXPORTS:
        target = root / name
        if name in {'src', 'frontend'}:
            target.mkdir()
            target = target / 'fixture.txt'
        target.write_text('committed baseline')
    subprocess.run(['git', '-C', str(root), 'add', *lab.EXPORTS], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    return root


def dest(repo):
    return repo / 'private/experiments/goozam/angelos2'


def test_committed_snapshot_excludes_private_and_uncommitted(repo):
    (repo / 'src/fixture.txt').write_text('uncommitted parallel work')
    (repo / 'src/untracked-secret').write_text('synthetic secret')
    (repo / 'private').mkdir()
    (repo / 'private/production.json').write_text('synthetic production config')
    result = lab.prepare(repo, dest(repo))
    assert result['state'] == 'prepared_not_running'
    assert not result['model_connected']
    assert (dest(repo) / 'runtime/src/fixture.txt').read_text() == 'committed baseline'
    assert not (dest(repo) / 'runtime/src/untracked-secret').exists()
    assert not (dest(repo) / 'runtime/private').exists()
    assert all(not list((dest(repo) / name).iterdir()) for name in lab.EMPTY_DIRS)
    assert (repo / 'src/fixture.txt').read_text() == 'uncommitted parallel work'


def test_existing_instance_never_overwritten(repo):
    lab.prepare(repo, dest(repo))
    marker = dest(repo) / 'data/keep.txt'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        lab.prepare(repo, dest(repo))
    assert marker.read_text() == 'keep'


@pytest.mark.parametrize('name', ['private/data/angelos', 'private/experiments/goozam',
                                   'private/experiments/goozam/../escape'])
def test_rejects_nonexperimental_destinations(repo, name):
    with pytest.raises(ValueError, match='experimental_directory_required'):
        lab.prepare(repo, repo / name)


def test_rejects_linked_destination(repo, tmp_path):
    (repo / 'private').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink_not_allowed'):
        lab.prepare(repo, dest(repo))


def test_rejects_links_in_source_archive(repo):
    (repo / 'src/link').symlink_to('/unavailable-production-data')
    subprocess.run(['git', '-C', str(repo), 'add', 'src/link'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'link'], check=True)
    with pytest.raises(ValueError, match='unsafe_archive_entry'):
        lab.prepare(repo, dest(repo))
    assert not dest(repo).exists()


def test_new_snapshot_pins_learning_extension(repo):
    package = repo / 'src/goozam'
    package.mkdir()
    (package / '__init__.py').write_text('"""合成学习扩展。"""\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'src/goozam'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'extension'], check=True)
    result = lab.prepare(repo, dest(repo))
    assert result['extension_source'] == 'runtime'
    assert 'src/goozam/__init__.py' in result['files']


def test_old_snapshot_keeps_explicit_external_extension(repo):
    assert lab.prepare(repo, dest(repo))['extension_source'] == 'external'
