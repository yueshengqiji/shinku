"""经验的历史观察与本轮授权上下文；仅描述适用线索，不拥有执行权限。

旧 anchors 是原任务结果，不是新任务前置条件。兼容投影只读，不迁移原库，
也不改来源指纹、评估记录或启用状态。执行仍由原宿主检查当前权限。
"""
from dataclasses import dataclass


def task_observations(result):
    """从原账本抽取有界观察，不复制回执正文或模型自填字段。"""
    if not isinstance(result, dict):
        return {}
    observed = {}
    status = (result.get('execution') or {}).get('status')
    if isinstance(status, str) and status:
        observed['execution_status'] = status[:32]
    tools, failed = set(), set()
    for row in result.get('evidence') or ():
        if not isinstance(row, dict):
            continue
        name = row.get('tool')
        if isinstance(name, str) and 0 < len(name) <= 64:
            tools.add(name)
            if row.get('execution_status') in {'error', 'failed', 'blocked'}:
                failed.add(name)
    for key, values in [('tools', tools), ('failed_tools', failed)]:
        if values:
            observed[key] = sorted(values)[:16]
    return observed


def source_observations(payload):
    """读取v2观察或旧无版本anchors；未知格式不能冒充可理解的历史。"""
    version = payload.get('observation_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('goozam_observation_version_unsupported')
    raw = payload.get('source_observations' if version == 2 else 'anchors', {})
    if not isinstance(raw, dict):
        raise ValueError('goozam_observations_invalid')
    # 旧数据只投影已知观察；原始payload原封不动留在库中。
    return {key: value for key, value in raw.items()
            if key in ('execution_status', 'tools', 'failed_tools')}


@dataclass(frozen=True)
class ExperienceContext:
    """仅宿主从当前grant构造；workspace必须与检索作用域一致。"""
    workspace: str
    allowed_tools: frozenset[str]

    @classmethod
    def from_grant(cls, workspace, grant):
        # 不截取授权工具前16项；源经验观察的长度限制不应改变当前工具集合。
        return cls(workspace, frozenset(name for name in grant.allowed_tools
                                       if grant.policy.accepts(name)))

    def valid_for(self, scope):
        return (isinstance(self.workspace, str) and bool(self.workspace)
                and self.workspace == scope and isinstance(self.allowed_tools, frozenset)
                and all(isinstance(name, str) and bool(name) for name in self.allowed_tools))

    def project(self, row):
        """工具重合只作说明，不评分、不筛掉经验，也不承诺执行获准。"""
        observed = source_observations(row)
        tools = set(observed.get('tools') or ())
        return dict(row, source_observations=observed, tool_context={
            'available_source_tools': sorted(tools & self.allowed_tools),
            'unavailable_source_tools': sorted(tools - self.allowed_tools),
            'is_execution_authorization': False})
