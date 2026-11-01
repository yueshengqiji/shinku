"""用量展示的真实 SDK 观测边界、隔离 HTTP 与浏览器定时行为（全部离线）。"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

from fastapi.testclient import TestClient
import pytest

from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.usage_view import UsageView, summarize
from shinku.llm.request_ledger import bind_provider_calls, observed_completion
from shinku.agent_runtime.evaluation import provider_usage


def response(inp=100, out=20, hit=80):
    return {'usage': {'prompt_tokens': inp, 'completion_tokens': out,
                      'prompt_cache_hit_tokens': hit, 'prompt_cache_miss_tokens': inp-hit}}


def event(binding='t', request_id='one', status='completed', reply=None):
    return {'binding': binding, 'request_id': request_id, 'status': status,
            'usage': provider_usage(response() if reply is None else reply), 'elapsed_seconds': 1}


def test_dedup_terminal_late_start_restart_and_binding(tmp_path):
    path = tmp_path/'usage.sqlite3'; view = UsageView(path)
    start = event(status='started', reply={})
    view(start); view(event()); view(event()); view(start)
    view(event('other', 'two', reply=response(9999)))
    view(event('t', 'unfinished', status='started', reply={}))
    fresh = UsageView(path)
    rows = fresh.read(['t', 't'])
    result = summarize(rows)
    assert result['requests'] == 2 and result['known']['input_tokens'] == 100
    assert result['totals']['input_tokens'] is None and result['totalTokens'] is None
    assert result['unsettledRequests'] == 1 and result['knownTotalTokens'] == 120
    assert result['cacheHitRatio'] is None
    assert path.stat().st_mode & 0o777 == 0o600


def test_actual_boundary_failure_retry_and_stream_not_per_chunk():
    delivered = []; view = UsageView(downstream=delivered.append)
    with bind_provider_calls('t', view):
        with pytest.raises(TimeoutError):
            observed_completion(lambda **kw: (_ for _ in ()).throw(TimeoutError()), {})
        observed_completion(lambda **kw: response(), {})
        assert len(list(observed_completion(lambda **kw: iter([response(200), response(300)]),
                                            {'stream': True}))) == 2
    rows = view.read(['t']); result = summarize(rows)
    assert result['requests'] == 3 and len(delivered) == 6
    assert result['known']['input_tokens'] == 400 and result['knownTotalTokens'] == 440
    assert result['incompleteRequests'] == 1 and result['totalTokens'] is None
    assert result['cacheHitRatio'] is None


def test_unknown_zero_and_reasoning_do_not_fake_totals():
    assert summarize([])['totalTokens'] is None
    row = event(reply={'usage': {'prompt_tokens': 0, 'completion_tokens': 0,
                               'prompt_cache_hit_tokens': 0}})
    zero = summarize([row])
    assert zero['totalTokens'] == 0 and zero['cacheHitRatio'] is None
    reply = response()
    reply['usage']['completion_tokens_details'] = {'reasoning_tokens': 15}
    result = summarize([event(reply=reply)])
    assert result['totalTokens'] == 120 and result['cacheHitRatio'] == .8
    result = summarize([event(reply={'usage': {'prompt_tokens': 100, 'completion_tokens': 20}})])
    assert result['totalTokens'] == 120 and result['totals']['cache_hit_tokens'] is None
    assert result['known']['cache_hit_tokens'] is None and result['cacheHitRatio'] is None


def test_inconsistent_cache_does_not_gain_plausible_ratio_from_other_calls():
    rows = [event(reply=response(100, 20, 120)), event(request_id='two', reply=response(1000, 20, 0))]
    result = summarize(rows)
    assert result['invalidCache'] and result['cacheHitRatio'] is None


@pytest.mark.parametrize('bad', [-1, True, '10', 1.5])
def test_invalid_counter_marks_incomplete_without_altering_external_sink(bad):
    delivered = []; view = UsageView(downstream=delivered.append)
    row = event(); row['usage']['input_tokens'] = bad
    view(row)
    assert not view.healthy and view.read(['t']) == [] and delivered == [row]


def test_view_write_failure_does_not_change_model_result_or_drop_external_sink():
    delivered = []; view = UsageView(downstream=delivered.append); view.db.close()
    calls = []
    with bind_provider_calls('t', view):
        result = observed_completion(lambda **kw: calls.append(1) or response(), {})
    assert result == response() and len(calls) == 1 and len(delivered) == 2
    assert not view.healthy


def test_http_scope_readonly_and_actual_agent_roundtrip(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    host = ServiceHost(service, reasoning_path=tmp_path/'reasoning.sqlite3')
    calls = []
    def reply(**kwargs):
        calls.append(1)
        answer = ({'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
                  if len(calls) == 1 else {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
                  if not tool.calls else {'kind': 'final', 'text': '已核验文件。', 'task_outcome': 'completed'})
        # 仅替换 SDK 返回，使用生产 observed_completion 的请求事件与真实 AgentService。
        return observed_completion(lambda **kw: dict(response(), answer=answer), {})['answer']
    runtime.call_chat_json = reply
    app = create_angelos_app(tmp_path/'ui', host=host)
    try:
        with TestClient(app) as client:
            boot = client.get('/angelos/api/bootstrap').json()
            assert boot['agent']['usageStats']
            headers = {'x-angelos-token': boot['token']}
            cid = client.post('/angelos/api/conversations', headers=headers, json={}).json()['conversation']['id']
            endpoint = '/angelos/api/conversations/'+cid
            client.post(endpoint+'/messages', headers=headers, json={'text': '创建并核验测试文件', 'request_key': 'usage'})
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                turn = client.get(endpoint).json()['conversation']['turns'][0]
                if turn['stage'] == 'finished': break
                time.sleep(.01)
            assert not turn['view']['verification']['verified'] and tool.path.read_text() == 'standalone agent verified'
            assert turn['result']['verification']['evidence'][-1]['receipt_valid'] is True
            before = copy.deepcopy(app.state.angelos.store.read('conversation', cid))
            first = client.get(endpoint+'/usage').json()['usage']
            assert first['summary']['requests'] == 3 and first['summary']['totalTokens'] == 360
            assert first['summary']['cacheHitRatio'] == .8 and first['unrecordedTurns'] == 0
            assert first['turns'][turn['id']]['totalTokens'] == 360
            # 同一真实 AgentService 链路：阶段计时与实际 SDK 请求关联，不增加调用。
            timing = first['turns'][turn['id']]['performance']
            phases = {r['phase']: r for r in timing['phases']}
            assert {'execution', 'planning', 'tool'} <= phases.keys()
            # 直接交付不再运行整项语义验收阶段，也不伪造其耗时样本。
            assert 'verification' not in phases
            assert phases['tool']['measured'] >= 1
            assert sum(r['requests'] for r in phases.values()) == 3
            assert phases['planning']['knownInputTokens'] == 300
            for _ in range(3): client.get(endpoint+'/usage')
            assert len(calls) == 3 and tool.calls == 1
            assert before == app.state.angelos.store.read('conversation', cid)
            assert client.get(endpoint+'/usage', headers={'origin':'https://example.invalid'}).status_code == 403
            assert client.get('/angelos/assets/usage.js').status_code == 200
            other = client.post('/angelos/api/conversations', headers=headers, json={}).json()['conversation']
            assert client.get('/angelos/api/conversations/'+other['id']+'/usage').json()['usage']['summary']['requests'] == 0
            with pytest.raises(ValueError, match='usage_conversation_mismatch'):
                host.read_usage(other['id'], before)
            other['turns'] = [turn]
            assert host.read_usage(other['id'], other)['turns'][turn['id']] == {'available': False}
            # 原准备绑定与任务绑定都保留；没有采集记录的历史轮明确列出。
            host.usage_view(event('prepare:'+cid+':'+turn['id'], 'prep'))
            old = copy.deepcopy(before); old['turns'].append({'id': 'legacy', 'stage': 'replied'})
            stats = host.read_usage(cid, old)
            assert stats['summary']['totalTokens'] == 480 and stats['unrecordedTurns'] == 1
            fresh = UsageView(tmp_path/'usage-view.sqlite3')
            assert len(fresh.read([turn['task_id']])) == 3
    finally:
        service.coordinator.shutdown(wait=True)


def test_frontend_interval_hidden_failures_and_late_conversation_response(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable: interval behavior not verified')
    source = Path(__file__).resolve().parents[1]/'frontend/angelos/usage.js'
    # 用虚拟时间驱动真实前端模块，断言网络请求次数与快照归属，而非查源码字符串。
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const settle=()=>new Promise(resolve=>setImmediate(resolve));
let tick,period,clock=0,visible=true,changes=0,requests=[];
const context={window:{},setTimeout:(fn,ms)=>{tick=fn;period=ms;return 1},clearTimeout:()=>{}};
vm.runInNewContext(fs.readFileSync(process.argv[2],'utf8'),context);
const display=context.window.AngelosUsage.create({now:()=>clock,visible:()=>visible,
 changed:()=>changes++,api:path=>new Promise((resolve,reject)=>requests.push({path,resolve,reject}))});
const summary={available:true,requests:1,totals:{input_tokens:100,output_tokens:20,cache_hit_tokens:80},
 known:{input_tokens:100,output_tokens:20,cache_hit_tokens:80},totalTokens:120,
 knownTotalTokens:120,cacheHitRatio:.8,incompleteRequests:0,unsettledRequests:0};
summary.performance={available:true,truncated:true,phases:[{label:'<img src=x>',samples:1,measured:0,
 totalMs:null,p50Ms:null,p95Ms:null,requests:1,requestMs:null,knownInputTokens:null,knownOutputTokens:0,
 incompleteUsage:1,failed:0,requestFailed:1,unsettled:1,requestUnsettled:0}]};
summary.cacheGroups={ordinary:summary,summary:{...summary,cacheHitRatio:.1},post_compaction:summary};
summary.recentRequests=[{category:'summary',cacheHitRatio:.1,prefix:{available:true,baseline:true,
 matchedMessages:3,firstChangedMessage:3,changedEnvelope:['tool_choice','<img src=x>']}}];
const data=id=>({usage:{available:true,conversationId:id,sampledAt:100,summary,turns:{t:summary}}});
(async()=>{
 display.select('a',true);assert.equal(requests.length,1);
 requests[0].resolve(data('a'));await settle();
 assert.equal(period,15000);
 assert.equal(display.statusHTML(),'会话命中 80.0%');assert.equal(display.instanceStatusHTML(),'累计命中未提供');
 assert.match(display.panel('session').summary,/120/);
 assert.match(display.panel('turn','new-turn').summary,/等待下一次采样/);
 assert.doesNotMatch(display.panel('turn','new-turn').summary,/80.0%/);
 assert.match(display.panel('instance').summary,/全局累计暂不可读/);
 assert.match(display.panel('session').advanced,/阶段耗时/);
 assert.equal(requests.length,1);
 assert.match(display.summaryHTML(),/80.0%/);assert.match(display.turnHTML('t'),/120/);
 assert.match(display.summaryHTML(),/摘要调用/);assert.match(display.summaryHTML(),/压缩后首轮/);
 assert.match(display.summaryHTML(),/10.0%/);assert.match(display.turnHTML('t'),/相同前 3 条消息/);
 assert.match(display.turnHTML('t'),/第 4 条起变化/);assert.match(display.turnHTML('t'),/tool_choice/);
 assert.match(display.turnHTML('t'),/阶段耗时/);assert.match(display.turnHTML('t'),/不可相加/);
 assert.match(display.turnHTML('t'),/未知/);assert.match(display.turnHTML('t'),/&lt;img/);
 assert.doesNotMatch(display.turnHTML('t'),/<img/);assert.match(display.summaryHTML(),/采集不完整/);
 assert.match(display.compactHTML('t'),/缓存命中率 <b>80.0%/);
 assert.match(display.compactHTML('t'),/输入 <b>100/);
 assert.match(display.compactHTML('t'),/输出 <b>20/);
 assert.match(display.compactHTML('new-turn'),/等待下一次用量采样/);
 assert.doesNotMatch(display.compactHTML('new-turn'),/80.0%/);
 for(clock=2500;clock<15000;clock+=2500){display.select('a',true);await tick()}
 assert.equal(requests.length,1);
 const pending=tick();assert.equal(requests.length,2);
 clock=30000;await tick();assert.equal(requests.length,2); // 在途不重叠
 requests[1].reject(new Error('offline'));await pending;
 assert.match(display.statusHTML(),/待更新/);
 assert.match(display.panel('session').summary,/更新失败/);
 assert.match(display.summaryHTML(),/更新失败/);assert.match(display.summaryHTML(),/120/);
 assert.match(display.compactHTML('t'),/更新失败/);assert.match(display.compactHTML('t'),/80.0%/);
 visible=false;clock=45000;await tick();assert.equal(requests.length,2);
 visible=true;const old=tick();assert.equal(requests.length,3);
 display.select('b',true);assert.equal(requests.length,4);
 assert.equal(display.statusHTML(),'');
 assert.doesNotMatch(display.compactHTML('t'),/80.0%/);
 const b=data('b');b.usage.summary={...summary,totalTokens:240};b.usage.instance={available:true,summary:{...summary,cacheHitRatio:.97,totalTokens:9000}};
 requests[3].resolve(b);await settle();
 requests[2].resolve(data('a'));await old;
 assert.match(requests[3].path,/conversations\/b\/usage/);
 assert.equal(display.instanceStatusHTML(),'累计命中 97.0%');assert.match(display.panel('instance').summary,/97.0%/);assert.match(display.panel('instance').summary,/9,000/);assert.doesNotMatch(display.statusHTML(),/97.0%/);
 assert.match(display.summaryHTML(),/240/);assert.doesNotMatch(display.summaryHTML(),/合计 <b>120/);
 display.select('c',true);assert.equal(requests.length,5);
 const unknown={...summary,requests:1,totals:{input_tokens:null,output_tokens:null,cache_hit_tokens:null},
 known:{input_tokens:null,output_tokens:null,cache_hit_tokens:null},totalTokens:null,knownTotalTokens:null,
 cacheHitRatio:null,incompleteRequests:1};
 requests[4].resolve({usage:{...data('c').usage,summary:unknown}});
 await settle();
 assert.match(display.summaryHTML(),/未提供/);assert.doesNotMatch(display.summaryHTML(),/合计 <b>0/);
 display.select('missing',true);
 requests[5].resolve({usage:{...data('missing').usage,summary:unknown,turns:{t:unknown}}});await settle();
 assert.match(display.compactHTML('t'),/未提供/);assert.doesNotMatch(display.compactHTML('t'),/>0</);
 display.select('zero',true);
 const zero={...summary,cacheHitRatio:0,totals:{...summary.totals,cache_hit_tokens:0}};
 requests[6].resolve({usage:{...data('zero').usage,turns:{t:zero}}});await settle();
 assert.match(display.compactHTML('t'),/缓存命中率 <b>0.0%/);
 display.select('old',false);clock=60000;await tick();assert.equal(requests.length,7);
 assert.equal(display.summaryHTML(),'');assert.equal(display.compactHTML('t'),'');display.stop();
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    path = tmp_path/'timing.cjs'; path.write_text(script)
    result = subprocess.run([node, str(path), str(source)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_usage_statusbar_is_outside_composer_and_scrolling_messages():
    from html.parser import HTMLParser
    class Structure(HTMLParser):
        def __init__(self): super().__init__(); self.stack=[]; self.parents={}
        def handle_starttag(self, tag, attrs):
            attrs=dict(attrs)
            if attrs.get('id'): self.parents[attrs['id']]=list(self.stack)
            if tag not in {'input','img','meta','link','br','hr','use'}:
                self.stack.append((tag, attrs.get('id',''),attrs.get('class','')))
        def handle_endtag(self, tag):
            for i in range(len(self.stack)-1,-1,-1):
                if self.stack[i][0]==tag: del self.stack[i:]; break
    dom=Structure()
    dom.feed((Path(__file__).resolve().parents[1]/'frontend/angelos/index.html').read_text())
    ancestors=dom.parents['usage-panel']
    assert any(tag=='footer' and 'workbench-statusbar' in classes.split() for tag,_,classes in ancestors)
    assert all('composer-shell' not in classes.split() for _,_,classes in ancestors)
    assert all(ident!='conversation-scroll' for _,ident,_ in ancestors)
    assert any(ident=='usage-popover' for _,ident,_ in dom.parents['usage-summary'])
    assert all('composer-shell' not in classes.split() for _,_,classes in dom.parents['usage-summary'])
    assert any(ident=='usage-details' for _,ident,_ in dom.parents['usage-advanced'])
