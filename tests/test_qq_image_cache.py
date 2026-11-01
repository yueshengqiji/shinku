"""CDN失效到真实NapCat适配器/容器缓存的隔离回归；不调用模型或发送QQ。"""
import base64
from email.message import Message
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from shinku.qq.config import NapCatConnectionConfig
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatHttpActionCaller, NapCatImageMaterializer, NapCatVisualInputBridge


URL = 'https://image.invalid/synthetic.jpg'
MEDIA = 'a' * 32 + '.jpg'
JPEG = b'\xff\xd8\xff\xe0synthetic-image'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Synthetic image acceptance must not connect to a real socket')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


class Response:
    def __init__(self, body, mime='application/json'):
        self.body, self.closed = body, False
        self.headers = Message()
        self.headers['Content-Type'] = mime
    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]
    def close(self):
        self.closed = True


def expired(request, **kwargs):
    raise HTTPError(URL, 400, 'synthetic expired CDN reference', {}, None)


def event(kind='private'):
    return {'post_type': 'message', 'message_type': kind, 'user_id': 'synthetic-owner',
        'self_id': 'synthetic-bot', 'group_id': 'synthetic-group', 'message_id': 'image-source',
        'message': [{'type': 'image', 'data': {'file': MEDIA, 'url': URL}}]}


def image_loader(tmp_path, response_override=None, roots=()):
    root = tmp_path/'cache'; root.mkdir(exist_ok=True)
    (root/'stored.jpg').write_bytes(JPEG)
    calls = []
    raw = response_override or {'status': 'ok', 'retcode': 0, 'data': {
        'file': '/container/images/stored.jpg', 'file_name': MEDIA, 'file_size': len(JPEG), 'url': URL}}
    def opener(request, **kwargs):
        calls.append((request.full_url.rsplit('/', 1)[-1], json.loads(request.data)))
        assert calls[-1][0] == 'get_image'  # 任何发送动作都必须使验收失败。
        return Response(json.dumps(raw).encode())
    caller = NapCatHttpActionCaller('http://napcat.invalid', opener=opener)
    materializer = NapCatImageMaterializer(url_opener=expired, media_loader=caller.get_image,
        path_mappings=(('/container/images', str(root)),), allowed_roots=roots)
    return materializer, calls, root


def test_http_failure_uses_existing_media_loader():
    # 只用旧API，原版应在URL失败后直接返回None；固定原故障而非仅测试新参数。
    calls = []
    materializer = NapCatImageMaterializer(url_opener=expired,
        media_loader=lambda ident: calls.append(ident) or JPEG)
    result = materializer({'kind': 'image', 'url': URL, 'media_id': MEDIA, 'mime_type': 'image/jpeg'})
    assert result == 'data:image/jpeg;base64,' + base64.b64encode(JPEG).decode()
    assert calls == [MEDIA]


def test_successful_url_does_not_call_napcat():
    calls = []
    materializer = NapCatImageMaterializer(url_opener=lambda *a, **k: Response(JPEG, 'image/jpeg'),
        media_loader=lambda ident: calls.append(ident))
    assert materializer({'url': URL, 'media_id': MEDIA}).startswith('data:image/jpeg;base64,')
    assert calls == []


def test_image_without_content_type_preserves_extension_inference():
    response = Response(JPEG); del response.headers['Content-Type']
    materializer = NapCatImageMaterializer(url_opener=lambda *a, **k: response)
    assert materializer({'url': URL}).startswith('data:image/jpeg;base64,')
    assert response.closed


@pytest.mark.parametrize('kind', ['private', 'group'])
def test_real_adapter_and_visual_bridge_keep_source_identity(tmp_path, kind):
    materializer, calls, root = image_loader(tmp_path)
    batch = NapCatVisualInputBridge(materialize=materializer).build(IncomingMessage.from_event(event(kind)))
    assert batch.ready and not batch.pending and len(batch.model_images()) == 1
    item = batch.inputs[0]
    assert item.source_message_id == 'image-source' and item.source_sender_id == 'synthetic-owner'
    assert item.reference == URL and item.reference_kind == 'direct'
    assert base64.b64decode(item.data_url.split(',', 1)[1]) == JPEG
    assert calls == [('get_image', {'file': MEDIA})]


@pytest.mark.parametrize('kind', ['private', 'group'])
def test_previous_image_selection_can_reread_expired_url_from_cache(tmp_path, kind):
    from test_qq_visual_provenance import Harness, message
    from shinku.qq.agent_bridge import QQAgentBridge
    materializer, calls, root = image_loader(tmp_path)
    bridge = NapCatVisualInputBridge(materialize=materializer)
    h = Harness()
    h.dispatcher.set_visual_loader(bridge.build)
    h.receive(IncomingMessage.from_event(event(kind)))
    current = h.receive(message('query-image', 'synthetic-owner', '看一下这张图', kind=kind,
        group='synthetic-group'))
    selected = h.choose(current, 'image-source')
    assert selected.visual_context_inherited and len(QQAgentBridge._images(selected)) == 1
    sources = selected.visuals_for('image-source')
    assert len(sources) == 1
    assert sources[0].source_message_id == 'image-source'
    assert sources[0].source_sender_id == 'synthetic-owner'
    assert calls == [('get_image', {'file': MEDIA})]


@pytest.mark.parametrize('path', ['/container/images/../outside.jpg', '/container/images-other/x.jpg',
    '/etc/secret.jpg', 'relative.jpg'])
def test_unmapped_and_traversal_cache_paths_are_not_read(tmp_path, path):
    raw = {'status': 'ok', 'data': {'file': path, 'file_name': MEDIA}}
    materializer, calls, root = image_loader(tmp_path, raw)
    assert materializer({'url': URL, 'media_id': MEDIA}) is None


def test_symlink_escape_and_explicit_read_root_are_enforced(tmp_path):
    materializer, calls, root = image_loader(tmp_path)
    outside = tmp_path/'outside.jpg'; outside.write_bytes(JPEG)
    (root/'stored.jpg').unlink(); (root/'stored.jpg').symlink_to(outside)
    assert materializer({'url': URL, 'media_id': MEDIA}) is None
    (root/'stored.jpg').unlink(); (root/'stored.jpg').write_bytes(JPEG)
    blocked, _, _ = image_loader(tmp_path, roots=(str(tmp_path/'other-allowed-root'),))
    assert blocked({'url': URL, 'media_id': MEDIA}) is None


def test_native_cache_needs_explicit_root_and_cannot_be_an_arbitrary_file(tmp_path):
    path = tmp_path/'native.jpg'; path.write_bytes(JPEG)
    loader = lambda ident: {'file': str(path)}
    assert NapCatImageMaterializer(media_loader=loader)({'media_id': MEDIA}) is None
    assert NapCatImageMaterializer(media_loader=loader, allowed_roots=(str(tmp_path),))(
        {'media_id': MEDIA}).startswith('data:image/jpeg;base64,')
    path = tmp_path/'settings.json'; path.write_text('synthetic credential')
    assert NapCatImageMaterializer(media_loader=loader, allowed_roots=(str(tmp_path),))(
        {'media_id': MEDIA, 'mime_type': 'image/jpeg'}) is None


def test_cache_size_limit_and_old_data_url_loader_contract(tmp_path):
    materializer, calls, root = image_loader(tmp_path)
    materializer.max_bytes = 4
    assert materializer({'url': URL, 'media_id': MEDIA}) is None
    result = 'data:image/png;base64,c3ludGhldGlj'
    assert NapCatImageMaterializer(media_loader=lambda ident: {'data_url': result, 'file': '/ignored'})({
        'media_id': MEDIA}) == result


@pytest.mark.parametrize('mime', ['text/html', 'application/json'])
def test_http_200_error_document_uses_cache_instead_of_faking_an_image(tmp_path, mime):
    materializer, calls, root = image_loader(tmp_path)
    response = Response(b'synthetic error page', mime)
    materializer._url_opener = lambda *a, **k: response
    result = materializer({'url': URL, 'media_id': MEDIA})
    assert base64.b64decode(result.split(',', 1)[1]) == JPEG
    assert response.closed and calls == [('get_image', {'file': MEDIA})]


@pytest.mark.parametrize('raw', [{'status': 'failed', 'retcode': 1}, {'status': 'async', 'retcode': 0},
    {'status': 'ok', 'retcode': 'invalid'}, {'status': 'ok', 'data': {'file': '/container/images/x.jpg',
        'file_name': 'different-source.jpg'}}, {'status': 'ok', 'data': None}])
def test_invalid_image_receipt_is_not_materialized(tmp_path, raw):
    materializer, calls, root = image_loader(tmp_path, raw)
    assert materializer({'url': URL, 'media_id': MEDIA}) is None
    assert calls == [('get_image', {'file': MEDIA})]


@pytest.mark.parametrize('ident', ['../private.jpg', '/etc/file.jpg', 'https://image.invalid/x', 'C:\\file.jpg', ''])
def test_get_image_rejects_paths_without_dispatch(ident):
    calls = []
    caller = NapCatHttpActionCaller('http://napcat.invalid', opener=lambda *a, **kw: calls.append(a))
    assert caller.get_image(ident) is None and calls == []


@pytest.mark.parametrize('value', ['[]', '{invalid', '{"/":"/srv/images"}',
    '{"relative":"/srv/images"}', '{"/container/images":"relative"}',
    '{"/container/../images":"/srv/images"}'])
def test_invalid_mapping_configuration_fails_closed(value):
    with pytest.raises(ValueError, match='invalid_napcat_image_path_mappings'):
        NapCatConnectionConfig.from_env({'SHINKU_NAPCAT_IMAGE_PATH_MAPPINGS': value})


def test_cli_wires_read_only_image_fallback_without_starting_server(tmp_path, monkeypatch):
    from test_cli import isolated_env
    from shinku import cli
    import shinku.qq.napcat as napcat
    root = tmp_path/'cache'; root.mkdir(); (root/'stored.jpg').write_bytes(JPEG)
    env = {**isolated_env(tmp_path), 'SHINKU_NAPCAT_BASE_URL': 'http://napcat.invalid',
        'SHINKU_NAPCAT_WEBHOOK_ENABLED': 'true', 'SHINKU_QQ_AGENT_ENABLED': 'false',
        'SHINKU_NAPCAT_IMAGE_PATH_MAPPINGS': json.dumps({'/container/images': str(root)})}
    calls = []
    def opener(request, **kwargs):
        if request.full_url == URL:
            return expired(request)
        action = request.full_url.rsplit('/', 1)[-1]; calls.append(action)
        assert action == 'get_image'
        return Response(json.dumps({'status': 'ok', 'data': {
            'file': '/container/images/stored.jpg', 'file_name': MEDIA}}).encode())
    def server(app, **kwargs):
        assert calls == []  # 构造不联网。
        assert app.state.napcat_host.handle_event(event()).visual.ready
    monkeypatch.setattr(napcat, 'urlopen', opener)
    monkeypatch.setattr('uvicorn.run', server)
    with patch.dict(os.environ, env, clear=True):
        assert cli.main(['serve', '--service', 'backend']) == 0
    assert calls == ['get_image']
