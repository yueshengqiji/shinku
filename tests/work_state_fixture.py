"""持久计划测试共用的合成提案；不依赖私用宿主或真实环境。"""
from shinku.agent.work_state import input_binding


def plan(requirements=(), *, status='in_progress', references=()):
    return {'reason': '按当前任务安排工作', 'current_step': 'inspect', 'next_action': '读取并核实当前输入',
        'blocker': '', 'findings': [], 'hypotheses': ['具体原因需要证据核实'], 'steps': [
            {'id': 'inspect', 'goal': '核实原问题并交付具体结果', 'depends': [],
             'requirements': list(requirements), 'status': status, 'references': list(references)}]}


def record(task, value, revision=1):
    return {'schema_version': 1, **{k: task[k] for k in ('task_id', 'requester', 'session')},
        'revision': revision, 'attempt': task['execution_attempt'], 'task_version': task['revision'] + 1,
        'binding': input_binding(task), 'plan': value, 'references': [], 'created': 10}
