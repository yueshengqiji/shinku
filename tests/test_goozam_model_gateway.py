"""只用合成价格/响应；验证固定出口、失败留证和批次额度，不访问真实模型。"""
from concurrent.futures import ThreadPoolExecutor
import json
import httpx
import pytest
from goozam.model_gateway import BatchLimits, GatewayStopped, ModelGateway, ModelRoute


def route(**overrides):
    value = dict(endpoint='https://model.example.invalid/v1/chat/completions', model='fixture-model',
        context_tokens=65536, max_output_tokens=4096,
        price=dict(source='synthetic fixture', effective_at='fixture', model='fixture-model', period='fixture',
            hit_cny_per_million='1', miss_cny_per_million='2', output_cny_per_million='3'))
    value.update(overrides)
    return ModelRoute(**value)


def limits(**overrides):
    value = dict(authorization_ref='offline-fixture-only', requests=4, tokens=1000000, seconds=60, cny='10')
    value.update(overrides)
    return BatchLimits(**value)


def response(**overrides):
    value = dict(id='fixture', object='chat.completion', created=1, model='fixture-model',
        choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant',
            content=json.dumps(dict(kind='final', text='静态分析推导代码行为；实际运行执行程序并产生真实回执。', task_outcome='completed'), ensure_ascii=False)))],
        usage=dict(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=60, prompt_cache_miss_tokens=40))
    value.update(overrides)
    return value


def request():
    return dict(model='fixture-model', messages=[dict(role='user', content='synthetic task')])


def gateway(tmp_path, callback=None, **changes):
    sent = []
    def handle(req):
        sent.append(req)
        return callback(req) if callback else httpx.Response(200, json=response())
    obj = ModelGateway(tmp_path / 'batch', route(), limits(**changes), 'synthetic-credential-for-test',
                       transport=httpx.MockTransport(handle))
    return obj, sent


def test_records_originals_cache_and_cost_without_output_cap(tmp_path):
    obj, sent = gateway(tmp_path)
    assert obj.complete(request())['model'] == 'fixture-model'
    body = json.loads(sent[0].content)
    assert not {'max_tokens', 'max_completion_tokens', 'api_key', 'base_url'} & body.keys()
    assert body['thinking'] == {'type': 'disabled'}
    assert sent[0].url == obj.route.endpoint
    assert sent[0].headers['authorization'] == 'Bearer synthetic-credential-for-test'
    summary = obj.ledger.snapshot(obj.binding, expected_requests=1)
    row = summary['calls'][0]
    assert row['request']['messages'] == request()['messages']
    assert json.loads(row['response_text']) == response()
    assert row['estimated_cny'] == '0.0002'
    assert summary['cache_hit_ratio'] == .6 and summary['ledger_complete']
    assert 'synthetic-credential-for-test' not in (tmp_path / 'batch/calls.jsonl').read_text()


@pytest.mark.parametrize('extra', [dict(model='other'), dict(base_url='http://127.0.0.1:8770'),
    dict(stream=True), dict(max_tokens=1), dict(thinking={'type':'enabled'}), dict(reasoning_effort='high'),
    dict(messages=[dict(role='user', content=[dict(type='image_url', image_url={'url':'http://local'})])]),
    dict(tools=[dict(type='web_search')])])
def test_caller_cannot_override_route_policy_or_fetch_urls(tmp_path, extra):
    obj, sent = gateway(tmp_path)
    with pytest.raises(ValueError):
        obj.complete({**request(), **extra})
    assert not sent and not (obj.root / 'calls.jsonl').exists()


@pytest.mark.parametrize('change', [dict(requests=1), dict(tokens=70000), dict(cny='0.21')])
def test_reservation_covers_worst_case_before_sending(tmp_path, change):
    obj, sent = gateway(tmp_path, **change)
    obj.complete(request())
    while True:
        try:
            obj.complete(request())
        except GatewayStopped:
            break
    assert len(sent) == obj.requests and obj.requests <= obj.limits.requests
    assert obj.tokens == 120 * obj.requests
    before = len(sent)
    with pytest.raises(GatewayStopped):
        obj.complete(request())
    assert len(sent) == before


@pytest.mark.parametrize('kind', ['http', 'redirect', 'missing_usage', 'bad_cache', 'model_drift', 'bad_json', 'over_cap', 'timeout'])
def test_failures_preserve_first_response_and_stop_without_retry(tmp_path, kind):
    def reply(req):
        if kind == 'timeout':
            raise httpx.ReadTimeout('synthetic timeout', request=req)
        if kind == 'http':
            return httpx.Response(500, text='synthetic upstream failure')
        if kind == 'redirect':
            return httpx.Response(302, headers={'location':'http://127.0.0.1:8770/'})
        if kind == 'bad_json':
            return httpx.Response(200, text='{"cutoff":')
        value = response()
        if kind == 'missing_usage':
            value.pop('usage')
        elif kind == 'bad_cache':
            value['usage']['prompt_cache_hit_tokens'] = 200
        elif kind == 'model_drift':
            value['model'] = 'different-model'
        elif kind == 'over_cap':
            value['usage']['completion_tokens'] = 5000
        return httpx.Response(200, json=value)
    obj, sent = gateway(tmp_path, reply)
    for _ in range(2):
        with pytest.raises(GatewayStopped):
            obj.complete(request())
    assert len(sent) == 1
    snapshot = obj.ledger.snapshot(obj.binding, expected_requests=1)
    assert snapshot['requests'] == 1 and snapshot['calls'][0]['status'] in {'failed', 'timeout'}
    assert snapshot['calls'][0]['response_text'] is not None
    assert (obj.root / 'stopped.json').exists()


def test_concurrent_admission_does_not_overspend(tmp_path):
    obj, sent = gateway(tmp_path, requests=1)
    def invoke(_):
        try:
            obj.complete(request())
            return True
        except GatewayStopped:
            return False
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(invoke, range(4))).count(True) == 1
    assert len(sent) == 1


@pytest.mark.parametrize('change', [dict(tokens=1000), dict(cny='0.0001')])
def test_insufficient_full_request_reservation_never_contacts_upstream(tmp_path, change):
    obj, sent = gateway(tmp_path, **change)
    with pytest.raises(GatewayStopped, match='reservation_exhausted'):
        obj.complete(request())
    assert not sent


def test_chinese_input_counts_utf8_and_still_rejects_oversized_input(tmp_path):
    obj, sent = gateway(tmp_path)
    value = request()
    value['messages'][0]['content'] = '中文' * 6000
    obj.complete(value)
    value['messages'][0]['content'] = '中文' * 12000
    with pytest.raises(ValueError, match='reservation_exceeded'):
        obj.complete(value)
    assert len(sent) == 1


@pytest.mark.parametrize('reason', ['revoke', 'deadline'])
def test_revocation_and_expiry_prevent_new_requests(tmp_path, reason):
    obj, sent = gateway(tmp_path)
    if reason == 'revoke':
        (obj.root / 'revoked').touch()
    else:
        obj.started -= 100
    with pytest.raises(GatewayStopped):
        obj.complete(request())
    assert not sent


def test_journal_failure_never_sends_unrecorded_request(tmp_path, monkeypatch):
    obj, sent = gateway(tmp_path)
    monkeypatch.setattr(obj, 'ledger', lambda row: (_ for _ in ()).throw(OSError('fixture')))
    with pytest.raises(OSError):
        obj.complete(request())
    assert not sent


def test_revocation_during_request_preserves_response_and_usage(tmp_path):
    def revoke(req):
        (tmp_path / 'batch/revoked').touch()
        return httpx.Response(200, json=response())
    obj, sent = gateway(tmp_path, revoke)
    with pytest.raises(GatewayStopped, match='revoked'):
        obj.complete(request())
    snapshot = obj.ledger.snapshot(obj.binding, expected_requests=1)
    assert len(sent) == 1 and snapshot['calls'][0]['status'] == 'completed'
    assert snapshot['totals']['input_tokens'] == 100
    assert json.loads(snapshot['calls'][0]['response_text']) == response()


@pytest.mark.parametrize('changes', [dict(endpoint='http://example.invalid/chat/completions'),
    dict(endpoint='https://user:key@example.invalid/chat/completions'),
    dict(endpoint='https://example.invalid/chat/completions?target=x'),
    dict(price={}), dict(max_output_tokens=0), dict(thinking='disabled', effort='high')])
def test_route_configuration_requires_explicit_valid_bounds(changes):
    with pytest.raises(ValueError):
        route(**changes)


@pytest.mark.parametrize('endpoint', ['http://localhost:8317/v1/chat/completions',
    'http://192.168.1.2:8317/v1/chat/completions', 'http://127.0.0.1/v1/chat/completions',
    'http://127.0.0.1:8317/other/chat/completions'])
def test_local_proxy_requires_explicit_numeric_loopback(endpoint):
    with pytest.raises(ValueError): route(endpoint=endpoint, local_proxy=True)


def test_local_proxy_alias_and_declared_billing_preserve_unknown_cache(tmp_path):
    config=route(endpoint='http://127.0.0.1:8317/v1/chat/completions',local_proxy=True,
        reported_model='upstream-fixture',billing='no_incremental_charge',
        billing_evidence='synthetic authorization only',price={})
    sent=[]
    def respond(req):
        sent.append(req)
        return httpx.Response(200,json=response(model='upstream-fixture',
            usage=dict(prompt_tokens=100,completion_tokens=20)))
    obj=ModelGateway(tmp_path/'batch',config,limits(), 'synthetic-key',transport=httpx.MockTransport(respond))
    obj.complete(request())
    summary=obj.ledger.snapshot(obj.binding,expected_requests=1)
    assert summary['totals']['input_tokens']==100 and summary['cache_hit_ratio'] is None
    assert summary['calls'][0]['estimated_cny']=='0'
    assert json.loads(sent[0].content)['model']=='fixture-model'
    assert summary['calls'][0]['reported_model']=='upstream-fixture'
    assert config.public()['route_fingerprint']!=route().public()['route_fingerprint']


@pytest.mark.parametrize('change', [dict(billing_evidence=''),dict(price={'fake':0}),dict(local_proxy=False)])
def test_free_route_is_never_inferred(change):
    with pytest.raises(ValueError):
        route(**(dict(endpoint='http://127.0.0.1:8317/v1/chat/completions',local_proxy=True,
            billing='no_incremental_charge',billing_evidence='fixture',price={})|change))


def test_declared_free_route_still_stops_on_unknown_usage(tmp_path):
    config=route(endpoint='http://127.0.0.1:8317/v1/chat/completions',local_proxy=True,
        billing='no_incremental_charge',billing_evidence='fixture',price={})
    obj=ModelGateway(tmp_path/'batch',config,limits(),'synthetic-key',
        transport=httpx.MockTransport(lambda req:httpx.Response(200,json=response(usage={}))))
    with pytest.raises(GatewayStopped,match='usage_or_cost_unknown'): obj.complete(request())
