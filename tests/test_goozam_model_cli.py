"""显式启动配置的离线边界；默认命令只校验，不创建代理或访问上游。"""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_goozam_model_gateway import route, limits
from goozam.sandbox import plain_tree

script = Path(__file__).resolve().parents[1] / 'scripts/run_goozam_model_probe.py'
spec = importlib.util.spec_from_file_location('goozam_probe_cli', script)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def config(tmp_path):
    key = tmp_path / 'key.txt'
    key.write_text('synthetic-key-not-a-real-credential')
    key.chmod(0o600)
    data = asdict(route())
    data.pop('fixture_http')
    value = dict(route=data, limits=asdict(limits()), key_file=str(key))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def test_default_cli_is_validation_only(tmp_path, config):
    result = subprocess.run([sys.executable, '-B', str(script), '--config', str(config),
        '--lab', str(tmp_path / 'lab'), '--batch-dir', str(tmp_path / 'batch')],
        capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(result.stdout)['status'] == 'validated_not_executed'
    assert not (tmp_path / 'batch').exists()
    assert 'synthetic-key' not in result.stdout + result.stderr


def test_parent_traversal_cannot_hide_a_key_inside_mounted_lab(tmp_path, config):
    lab = tmp_path / 'lab'
    lab.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    key = lab / 'key.txt'
    key.write_text('fixture')
    key.chmod(0o600)
    value = json.loads(config.read_text())
    value['key_file'] = str(outside / '../lab/key.txt')
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='parent_traversal'):
        cli.settings(config, lab)
    with pytest.raises(ValueError, match='parent_traversal'):
        plain_tree(outside / '../lab')


@pytest.mark.parametrize('change', ['public_config', 'public_key', 'key_in_lab', 'fixture_switch', 'missing_price'])
def test_invalid_secret_or_route_configuration_is_rejected(tmp_path, config, change):
    value = json.loads(config.read_text())
    lab = tmp_path / 'lab'
    if change == 'public_config':
        config.chmod(0o644)
    elif change == 'public_key':
        Path(value['key_file']).chmod(0o644)
    elif change == 'key_in_lab':
        lab.mkdir()
        key = lab / 'key.txt'
        key.write_text('synthetic')
        key.chmod(0o600)
        value['key_file'] = str(key)
    elif change == 'fixture_switch':
        value['route']['fixture_http'] = True
    else:
        value['route']['price'] = {}
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cli.settings(config, lab)


def test_local_proxy_config_is_explicit_and_validation_does_not_call(tmp_path,config):
    value=json.loads(config.read_text())
    value['route'].update(endpoint='http://127.0.0.1:8317/v1/chat/completions',local_proxy=True,
        billing='no_incremental_charge',billing_evidence='synthetic permission only',price={})
    config.write_text(json.dumps(value))
    selected,_,_=cli.settings(config,tmp_path/'lab')
    assert selected.local_proxy and selected.billing=='no_incremental_charge'
    refs=tmp_path/'refs.json'
    refs.write_text(json.dumps([dict(session='angelos:fixture',task_id='fixture')]))
    result=subprocess.run([sys.executable,'-B',str(script),'--config',str(config),'--lab',str(tmp_path/'lab'),
        '--batch-dir',str(tmp_path/'batch'),'--action','learn-propose','--sources',str(refs)],
        capture_output=True,text=True,timeout=15,check=True)
    assert json.loads(result.stdout)['status']=='validated_not_executed'
    assert not (tmp_path/'batch').exists()
