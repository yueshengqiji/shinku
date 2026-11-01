"""沙箱中的显式提案入口：只读取所选实验任务，生成待评估候选后退出。"""
import json
from pathlib import Path
import sys

from shinku.agent_runtime import ExecutionProfile
from shinku.angelos.task_continuity import workspace_binding
from .host import experiment
from .model_runtime import relay_runtime
from .offline import probe


def run(profile, refs):
    isolation = probe()
    with relay_runtime('/run/goozam-model.sock', profile) as runtime:
        execution = ExecutionProfile(runtime, profile_id='goozam-relay-v1',
            runtime_meters_budget=True, native_tool_history=True)
        with experiment(Path('/lab'), runtime, profile=execution) as lab:
            if not isinstance(refs, list) or not 1 <= len(refs) <= 3:
                raise ValueError('goozam_batch_limit')
            scopes = [workspace_binding(lab.workbench.store, ref['session'])[1] for ref in refs]
            if any(scope != scopes[0] for scope in scopes):
                raise ValueError('goozam_mixed_source_scopes')
            results = lab.plugin.propose(scopes[0], refs, runtime, profile)
            return dict(kind='candidates_not_learning_gain', isolation=isolation, proposals=results)


if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]), json.loads(sys.argv[2])), ensure_ascii=False))
