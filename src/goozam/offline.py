"""确定性离线演练：真实任务账本/记忆/工作台，明确不代表真实模型学习质量。

probe验证OS边界；exercise在沙箱内运行小闭环并归档。两者均无模型客户端，
不得把此响应器标为DeepSeek或自动切换到生产模型。
"""
import json
import os
from pathlib import Path
import socket
import sys
import time

from .host import experiment
from .learning import Assessment
from shinku.angelos.task_continuity import workspace_binding
from shinku.agent_runtime import AgentScope


def probe():
    denied = {}
    for port in (8317, 8770, 9998, 17897, 6099):
        with socket.socket() as sock:
            sock.settimeout(.3)
            denied[str(port)] = sock.connect_ex(('127.0.0.1', port)) != 0
    write_denied = False
    try:
        Path('/runtime/.write-probe').write_text('should be denied')
    except OSError:
        write_denied = True
    status = Path('/proc/self/status').read_text()
    facts = {'home_hidden': not Path('/home').exists(), 'runtime_readonly': write_denied,
             'host_ports_unreachable': denied, 'unprivileged': os.geteuid() != 0,
             'caps_empty': 'CapEff:\t0000000000000000' in status,
             'no_new_privs': 'NoNewPrivs:\t1' in status,
             'session_bus_absent': 'DBUS_SESSION_BUS_ADDRESS' not in os.environ,
             'model_credentials_absent': not any('API_KEY' in k or 'TOKEN' in k for k in os.environ)}
    if not all(v if isinstance(v, bool) else all(v.values()) for v in facts.values()):
        raise RuntimeError('goozam_isolation_failed:' + json.dumps(facts))
    return facts


class OfflineRuntime:
    """只用于验证数据流的固定响应器；经验收益由留出样例检查，非模型实测。"""
    def __init__(self):
        self.text = '这是离线合成回答。'
        self.calls = []
        self.responses = []

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {'kind': 'final', 'text': self.text, 'task_outcome': 'completed'}


def send(lab, wid, text, output):
    """每个样例新会话；用同一工作区，避免历史泄漏造成伪对照收益。"""
    runtime = lab.runtime
    runtime.text = output
    conversation = lab.workbench.store.conversation(wid)
    cid = conversation['id']
    lab.workbench.send(cid, {'request_key': 'offline-turn', 'text': text})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = lab.workbench.refresh(cid)
        turn = current['turns'][0]
        if turn['stage'] in {'finished', 'failed', 'replied'}:
            if not turn.get('task_id'):
                raise RuntimeError('goozam_missing_real_task:' + str(turn.get('error')))
            return {'session': 'angelos:' + cid, 'task_id': turn['task_id']}, turn
        time.sleep(.02)
    raise TimeoutError('goozam_offline_task_timeout')


def exercise(root):
    """验证候选不自动生效、留出对照、重启持久化与撤销；原任务只读保存。"""
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        # 合成演练显式打开实验经验；不是正式实例默认值，也不授权真实调用。
        previous = lab.controls.snapshot()
        lab.controls.select('use', previous['revision'])
        try:
            lab.runtime = runtime
            wid = lab.workbench.store.workspace({'name': 'goozam离线样例-' + str(time.time_ns()),
                'path': str(lab.workspace), 'description': '仅离线合成案例'})['id']
            seed, _ = send(lab, wid, '解释合成循环A的输出', '循环A输出0和1。')
            _, scope = workspace_binding(lab.workbench.store, seed['session'])
            original = lab.host.service.bind(AgentScope(
                lab.host.requester, seed['session'])).result(seed['task_id'])
            key = lab.learning.batch(scope, [seed], lambda source: dict(
                situation='静态代码解释', conditions='没有执行工具回执',
                advice='代码解释应注明静态推导，不声称已经运行。', exceptions='已有真实运行回执时按实际回执报告。'))[0]
            assert not lab.learning.recall(scope, '代码解释')['items']
            baseline, bturn = send(lab, wid, '解释合成循环B的输出', '循环B输出2和3。')
            def evaluator(candidate):
                assert '静态推导' in candidate['advice']
                trial, turn = send(lab, wid, '解释合成循环B的输出', '静态推导：循环B输出2和3；未实际运行。')
                assert turn['result']['execution']['status'] == 'completed'
                def score(value):
                    text = value['result']['execution_report']['text']
                    return float('静态推导' in text and '未实际运行' in text)
                return Assessment(baseline, trial, score(bturn), score(turn),
                    not turn['result']['evidence'], '合成留出规则：标明推导及未运行；不代表模型质量。')
            assert lab.learning.evaluate(scope, key, evaluator) == 'evaluated'
            lab.learning.activate(scope, key)
            assert lab.learning.recall(scope, '代码解释')['items'][0]['id'] == key
            from .learning import LearningStore
            reopened = LearningStore(Path(root) / 'learning', lab.learning.resolve)
            assert reopened.recall(scope, '代码解释')['items'][0]['id'] == key
            # 通过真实Agent工具调度及观察签名读取经验，不只直接调用Python检索函数。
            contract = {'goal': '读取实验代码解释经验', 'deliverables': ['经验建议及限制'],
                'scope': ['当前实验工作区'], 'requirements': [{'id': 'experience',
                    'description': '实际读取经验', 'effect': 'tool:read_goozam_experiences',
                    'targets': [], 'checks': {}}]}
            runtime.responses = [
                {'tool_call': {'type': 'declare_task_contract', 'task_contract': contract}},
                {'tool_call': {'type': 'read_goozam_experiences', 'query': '代码解释'}},
                {'kind': 'final', 'text': '已读取经验：无执行回执时注明静态推导；这不证明程序实际运行。',
                 'task_outcome': 'completed'}]
            _, retrieved = send(lab, wid, '读取当前工作区的代码解释经验', '')
            assert retrieved['result']['execution']['status'] == 'completed'
            assert any(e['tool'] == 'read_goozam_experiences' and e['execution_status'] == 'ok'
                       for e in retrieved['result']['evidence'])
            reopened.revoke(scope, key)
            assert not lab.learning.recall(scope, '代码解释')['items']
            port = lab.host.port(seed['session'].removeprefix('angelos:'))
            assert port.result(seed['task_id']) == original
            return {'kind': 'offline_fixture_not_model_quality', 'source': seed, 'holdout_source': baseline, 'candidate': key,
                    'proposed_evaluated_activated_reopened_revoked': True,
                    'original_result_preserved': True, 'real_task_count': 4,
                    'experience_read_via_agent_tool': True,
                    'model_http_calls': 0, 'permission': lab.host.permissions.level,
                    'delegation_disabled': lab.host.delegation is None,
                    'diagnostic_services_disabled': lab.host.service_logs is None}
        finally:
            current = lab.controls.snapshot()
            # 只撤回本演练设置；如操作者期间已改档，不覆盖新选择。
            expected = previous['revision'] + (previous['mode'] != 'use')
            if current['revision'] == expected:
                lab.controls.select(previous['mode'], current['revision'])


if __name__ == '__main__':
    action = sys.argv[1]
    result = probe()
    if action == 'exercise':
        result = {'isolation': result, 'exercise': exercise(Path('/lab'))}
        Path('/lab/evaluations/' + str(time.time_ns()) + '.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
    elif action != 'probe':
        raise ValueError('goozam_unknown_action')
    print(json.dumps(result, ensure_ascii=False, indent=2))
