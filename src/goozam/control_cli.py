"""沙箱内的可信操作者开关入口；不创建 Agent、模型客户端或学习请求。"""
import json
import sys
from .plugin import Controls


def run(action):
    controls = Controls('/lab/learning')
    state = controls.snapshot()
    if action != 'plugin-status':
        state = controls.select({'plugin-off': 'off', 'plugin-learn': 'learn', 'plugin-use': 'use'}[action],
                                state['revision'])
    return dict(plugin='goozma', state=state, model_calls=0)


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1]), ensure_ascii=False))
