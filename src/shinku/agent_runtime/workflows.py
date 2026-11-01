"""由模型按需读取的三份工作参考；不分类用户请求、不规划额外模型调用。

GUIDES 是可调整的参考资料，绝不是授权、硬契约或固定答复。工具清单每次
从当前宿主授权及已注册元数据派生，只表示注册/获准，不保证依赖或业务健康。
原执行器仍负责工具资格、材料门、效果核验和最终交付。
"""
import copy

from shinku.agent.work_state import digest


GUIDES = {
    'diagnosis': {
        'when': '用户要求定位已发生的执行失败、卡住或结果异常；按实际问题调整步骤。',
        'inputs': ['具体任务/进程/日志及版本范围', '预期行为、已观察行为和保留原文'],
        'steps': ['定位同一作用域的原任务和首个具体错误，保留失败原件，不先重跑。',
            '沿真实调用链核对输入、运行版本、权限和依赖；列出事实、假设及其验证方法。',
            '优先使用当前获准只读工具验证关键假设；未知副作用先检查，不把健康检查当业务成功。',
            '获准修复时做必要修改并复现原场景；否则交付具体原因、已做检查和缺口。'],
        'artifacts': ['原失败及具体错误定位', '原因依据、验证结果和未核实假设'],
        'acceptance': '回答原故障问题；修复成功需真实独立结果，不能只认退出码或模型陈述。',
        'preferred_groups': ['observe'],
    },
    'project_analysis': {
        'when': '用户要求理解项目、分析设计或提出优化方向；不默认要求穷举审计。',
        'inputs': ['用户当前目标、范围和硬条件', '入口、相关实现及已有读取记录'],
        'steps': ['确认范围，读取真实入口，沿调用关系核对状态归属和关键边界。',
            '复用仍有效的已读材料，按具体疑问补读相关实现；没有新疑问时收束。',
            '区分事实、成立条件下的推断和建议；不从同名/同值推断耦合，不把旧需求当当前缺陷。',
            '交付有依据的判断、优先顺序、可行改法和代价；未知数字和收益明确未测。'],
        'artifacts': ['实际分析正文', '对应来源、建议依据及未核实部分'],
        'acceptance': '真实读取和用户硬条件严格；普通分析局部引用/格式另列质量问题，不虚构修改或测试。',
        'preferred_groups': ['read'],
    },
    'modify_validate': {
        'when': '用户已明确授权修改并要求可验证结果；步骤可随真实环境调整。',
        'inputs': ['原要求、范围、不可破坏条件和当前权限', '原文件版本、已有修改、执行环境和验收方法'],
        'steps': ['核对环境、原文件和已有工作，保存必要原稿；确认真实写入目标与验收条件。',
            '按依赖实施最小充分修改；长任务记录当前步骤，失败时保留原错误再调整。',
            '运行与改动相称的验证，检查真实产物和用户界面；成功工具调用不代替业务验收。',
            '交付具体改动、验证证据和剩余事项；源码提交、运行加载、发送和部署分别判断。'],
        'artifacts': ['可审查的实际改动及必要原稿', '测试/产物回执和明确的交付状态'],
        'acceptance': '权限、输入版本、效果与原硬条件由宿主严格检查；参考流程不能授权安装、登录、外发或付费。',
        'preferred_groups': ['write', 'verify'],
    },
}


def workflow_reference(name, handlers, fresh):
    """动态列出可用路径和缺口，不把首选接口缺失推导为绝对能力缺失。"""
    guide = copy.deepcopy(GUIDES[name])
    groups = {key: [] for key in ('observe', 'read', 'write', 'command', 'verify')}
    for tool, handler in sorted(handlers.items()):
        if (tool not in fresh.allowed_tools or not fresh.policy.accepts(tool)
                or getattr(handler, 'agent_execution_evidence', True) is False):
            continue
        effects = getattr(handler, 'agent_completion_effects', ()) or ()
        effects = {effects} if isinstance(effects, str) else set(effects)
        if (getattr(handler, 'agent_permission_class', '') == 'read'
                and not effects & {'write', 'delete', 'send', 'process_stop', 'command_run'}):
            groups['observe'].append(tool)
            if 'read' in effects:
                groups['read'].append(tool)
        if effects & {'write', 'delete'}:
            groups['write'].append(tool)
        if 'command_run' in effects:
            groups['command'].append(tool)
        if callable(getattr(handler, 'agent_verify_result', None)):
            groups['verify'].append(tool)
    return {'view': 'workflow', 'workflow': name, 'schema_version': 1, 'guide_sha256': digest(guide),
        'guide': guide, 'current_tools': groups,
        'missing_preferred_groups': [g for g in guide['preferred_groups'] if not groups[g]],
        'capability_boundary': '首选接口缺失时按已获准通用工具和真实schema判断替代路径；清单不证明依赖或业务实测。',
        'is_execution_evidence': False, 'grants_permissions': False, 'adaptable_reference': True}
