"""本机工作台的累计预算选择，不负责权限、费用计价或模型上下文。

启动工厂在本机会话授权通过后调用；远程委托/实验仍用自身显式有限预算。
旧任务按账本与当前宿主取较严限制，绝不因升级自动释放旧额度或清掉停止原因。
"""
from shinku.agent.budget import TaskBudget

LIMITS = ('max_requests', 'max_tokens', 'max_tools', 'max_active_seconds')


def local_task_limits(task, ceilings, *, mode='metered'):
    """mode仅来自宿主配置，不能读取用户消息或模型任务契约中的同名字段。"""
    if mode not in {'metered', 'bounded'}:
        raise ValueError('invalid_angelos_budget_mode')
    bounded = TaskBudget(**dict(ceilings))
    if bounded.metered_only:
        raise ValueError('finite_experiment_limits_required')
    budget = TaskBudget(**{k: None for k in LIMITS}) if mode == 'metered' else bounded
    prior = task.get('budget')
    if prior is not None:
        budget.absorb_prior(prior)
        if budget.reason == 'task_budget_snapshot_invalid':
            raise ValueError('invalid_prior_budget')
    return {k: getattr(budget, k) for k in LIMITS}
