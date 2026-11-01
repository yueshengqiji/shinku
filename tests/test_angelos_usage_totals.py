"""实例累计以物理请求回执加权，跨会话但不跨用量库；无真实模型调用。"""
from shinku.angelos.usage_view import UsageView
from shinku.angelos.usage_totals import read_instance_usage
from test_angelos_usage import event, response


def test_weighted_instance_totals_and_restart(tmp_path):
    path=tmp_path/'usage.sqlite3'; view=UsageView(path)
    view(event('a','1',reply=response(100,20,90)))
    view(event('b','2',reply=response(900,80,450)))
    view(event('a','1',reply=response(100,20,90)))
    s=read_instance_usage(view)['summary']
    assert s['requests']==2 and s['totalTokens']==1100
    assert s['cacheHitRatio']==.54  # 不能平均90%和50%得到70%。
    assert read_instance_usage(UsageView(path))['summary']==s
    assert read_instance_usage(UsageView())['summary']['requests']==0


def test_missing_and_invalid_usage_are_visible_not_zero():
    view=UsageView();view(event('a','good',reply=response(100,20,80)))
    view(event('b','pending',status='started',reply={}))
    view(event('b','bad',reply={'usage':{'prompt_tokens':100,'completion_tokens':2,'prompt_cache_hit_tokens':120}}))
    s=read_instance_usage(view)['summary']
    assert s['totalTokens'] is None and s['knownTotalTokens']==222
    assert s['unsettledRequests']==1 and s['incompleteRequests']==1
    assert s['cacheHitRatio']==.8 and s['cacheCoverageRequests']==1 and s['excludedCacheRequests']==2


def test_cached_snapshot_is_independent_and_refreshes(monkeypatch):
    import shinku.angelos.usage_totals as module
    clock=[10.0];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    view=UsageView();view(event())
    first=read_instance_usage(view);first['summary']['requests']=999
    view(event('b','2'))
    assert read_instance_usage(view)['summary']['requests']==1
    clock[0]=25.0
    assert read_instance_usage(view)['summary']['requests']==2


def test_empty_and_zero_hit_are_different():
    view=UsageView();s=read_instance_usage(view)['summary']
    assert s['cacheHitRatio'] is None and s['totalTokens'] is None
    view(event(reply=response(100,20,0)));view._instance_usage_cache=None
    s=read_instance_usage(view)['summary']
    assert s['cacheHitRatio']==0 and s['totalTokens']==120


def test_instance_api_empty_conversation_and_origin_guard(tmp_path):
    from fastapi.testclient import TestClient
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    from shinku.angelos.app import create_angelos_app
    service, tool, runtime=demo_service(tmp_path)
    host=ServiceHost(service,reasoning_path=tmp_path/'reasoning.sqlite3')
    host.usage_view(event('other-conversation','prior',reply=response(100,20,80)))
    app=create_angelos_app(tmp_path/'ui',host=host)
    try:
        with TestClient(app) as client:
            boot=client.get('/angelos/api/bootstrap').json()
            cid=client.post('/angelos/api/conversations',headers={'x-angelos-token':boot['token']},json={}).json()['conversation']['id']
            url='/angelos/api/conversations/'+cid+'/usage'
            value=client.get(url).json()['usage']
            assert value['summary']['requests']==0
            assert value['instance']['summary']['requests']==1
            assert value['instance']['summary']['cacheHitRatio']==.8
            assert client.get(url,headers={'origin':'https://untrusted.invalid'}).status_code==403
            assert not tool.calls
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
