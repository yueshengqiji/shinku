"""仅通过显式代理运行一个合成 Agent 任务；结果仍按原宿主验收并保留原稿。

入口不自动学习、不启用候选，也不证明质量提升。宿主独立调用账本用于判断
本次是否真正发生模型请求；进程成功/任务完成本身不能证明真实上游或费用完整。
"""
import json
from pathlib import Path
import sys
import time

from shinku.agent_runtime import ExecutionProfile
from .host import experiment
from .model_runtime import relay_runtime
from .offline import probe


def run(profile):
    isolation = probe()
    with relay_runtime('/run/goozam-model.sock', profile) as runtime:
        execution = ExecutionProfile(runtime, profile_id='goozam-relay-v1',
            runtime_meters_budget=True, native_tool_history=True)
        with experiment(Path('/lab'), runtime, profile=execution) as lab:
            wid = lab.workbench.store.workspace(dict(name='goozam模型探针-' + str(time.time_ns()),
                path=str(lab.workspace), description='只含合成请求的独立实验'))['id']
            cid = lab.workbench.store.conversation(wid)['id']
            lab.workbench.send(cid, dict(request_key='model-probe',
                text='用两句话说明静态代码分析与实际运行程序的区别。无需调用工具，不要声称已运行任何程序。'))
            until = time.monotonic() + 120
            while time.monotonic() < until:
                turn = lab.workbench.refresh(cid)['turns'][0]
                if turn['stage'] in {'finished', 'failed', 'replied'}:
                    result = dict(isolation=isolation, kind='model_channel_probe_not_learning_gain',
                        conversation=cid, task_id=turn.get('task_id'), turn=turn)
                    destination = Path('/lab/evaluations') / ('model-probe-' + str(time.time_ns()) + '.json')
                    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    return result
                time.sleep(.05)
            raise TimeoutError('goozam_model_probe_task_timeout')


if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1])), ensure_ascii=False, indent=2))
