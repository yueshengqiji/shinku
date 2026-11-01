"""发布输入与本机旧证据分离，合成目录/真实zip验证媒体检查不能被范围过滤绕过。"""
import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts.audit_shinku_release import audit_release
from scripts.build_shinku_release_manifest import build_manifest,iter_release_files
from scripts.build_shinku_release_bundle import build_bundle
from scripts.audit_shinku_artifact import audit_artifact
from scripts.check_shinku_cutover import assess


@pytest.fixture
def tree(tmp_path):
    root=tmp_path/'tree';root.mkdir()
    files={'README.md':'Synthetic release fixture', 'LICENSE':'Apache License\nVersion 2.0',
        'NOTICE':'原作与外部素材边界\n第三方依赖',
        'pyproject.toml':'[project]\nname="fixture"\nversion="0.1"\nlicense = { file = "LICENSE" }\ndependencies=[]\n',
        'docs/THIRD_PARTY_LICENSES.md':'No dependencies in synthetic fixture.', 'docs/MCP_SCOPE.md':'Scope',
        'requirements-cleanenv.lock':'', 'requirements-build.lock':'',
        'src/shinku/__init__.py':'', 'frontend/angelos/index.html':'<main>fixture</main>',
        'scripts/fixture.py':'VALUE = "fixture"', 'tests/test_fixture.py':'',
        'setup.py':'# synthetic build entry', 'MANIFEST.in':'include setup.py',
        '.env.example':'SYNTHETIC=true', '.gitignore':'*.png\nfrontend/\ntmp/\n'}
    for relative,text in files.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    return root


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_old_top_level_evidence_is_preserved_and_not_in_release(tree,tmp_path):
    evidence=[]
    for relative in ('tmp/old-run/silence.wav','tmp/old-run/browser.png','private/evidence/sound.flac'):
        path=tree/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic old evidence');evidence.append(path)
    before={p:digest(p)for p in evidence}
    result=audit_release(tree);manifest=build_manifest(tree);bundle=build_bundle(tree,tmp_path/'clean.zip')
    assert result['ok'] and manifest['ok'] and bundle['ok']
    assert not result['media_files']
    with zipfile.ZipFile(bundle['artifact'])as archive:
        assert not any(n.startswith(('tmp/','private/'))for n in archive.namelist())
    assert before=={p:digest(p)for p in evidence}
    cutover=assess(tree,old_url='http://old',new_url='http://new',fetcher=lambda url:{'status':'ready'if url.endswith('/health/ready')else'ok'})
    assert cutover['ready_for_manual_cutover'] and cutover['manual_confirmation_required']


@pytest.mark.parametrize('relative',['src/icon.PNG','frontend/angelos/icon.svg','docs/sound.flac',
    'tests/sample.woff2','scripts/sample.mp4','src/tmp/silence.wav'])
def test_media_in_release_inputs_is_rejected_even_if_gitignored(tree,tmp_path,relative):
    path=tree/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic media')
    before=digest(path)
    assert path in list(iter_release_files(tree))  # 不能先筛掉媒体再让审计变绿。
    audit=audit_release(tree);manifest=build_manifest(tree);bundle=build_bundle(tree,tmp_path/'blocked.zip')
    assert not audit['ok'] and 'media_in_tree:'+relative in audit['findings']
    assert not manifest['ok'] and 'media_file:'+relative in manifest['findings']
    assert not bundle['ok'] and 'media_file:'+relative in bundle['findings']
    with zipfile.ZipFile(bundle['artifact'])as archive:assert relative not in archive.namelist()
    assert digest(path)==before


def test_manifest_and_bundle_share_all_candidates_including_frontend_and_build_inputs(tree,tmp_path):
    manifest=build_manifest(tree);first=tmp_path/'a.zip';second=tmp_path/'b.zip'
    a=build_bundle(tree,first);b=build_bundle(tree,second)
    expected={row['path']for row in manifest['files']}
    assert expected==set(a['files'])==set(b['files'])
    assert {'frontend/angelos/index.html','setup.py','MANIFEST.in'}<=expected
    assert first.read_bytes()==second.read_bytes()
    assert audit_artifact(first)['ok']


@pytest.mark.parametrize('name',['tmp/old-run/silence.wav','private/evidence/photo.png',
    'frontend/angelos/icon.svg','unlisted-root/sound.flac'])
def test_actual_archive_media_is_always_rejected_regardless_of_source_scope(tree,tmp_path,name):
    artifact=tmp_path/'injected.zip';assert build_bundle(tree,artifact)['ok']
    # 模拟第三方打包器或包被补入文件；实际包审计绝不按源码ROOTS过滤条目。
    with zipfile.ZipFile(artifact,'a')as archive:archive.writestr(name,b'synthetic injected media')
    result=audit_artifact(artifact)
    assert not result['ok'] and 'media_file:'+name in result['findings']


def test_secret_in_release_scope_is_not_hidden_by_manifest_filter(tree,tmp_path):
    path=tree/'src/.env';path.write_text('SYNTHETIC=fixture')
    assert 'secret_file:src/.env' in audit_release(tree)['findings']
    assert 'secret_file:src/.env' in build_manifest(tree)['findings']
    assert not build_bundle(tree,tmp_path/'blocked.zip')['ok'] and path.exists()


def test_actual_wheel_media_is_checked_without_source_scope_filter(tmp_path):
    artifact=tmp_path/'synthetic-0.1-py3-none-any.whl'
    with zipfile.ZipFile(artifact,'w')as archive:
        archive.writestr('shinku/__init__.py','')
        archive.writestr('synthetic-0.1.dist-info/METADATA','Name: synthetic\nVersion: 0.1\n')
    assert audit_artifact(artifact)['ok']
    with zipfile.ZipFile(artifact,'a')as archive:
        archive.writestr('private/old-evidence/silence.wav',b'synthetic media')
    result=audit_artifact(artifact)
    assert not result['ok'] and 'media_file:private/old-evidence/silence.wav' in result['findings']
