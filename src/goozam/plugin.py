"""Goozma 可选插件：实例开关门控提案和经验读取，不拥有 Agent 执行状态。

阅读顺序：Controls 持久状态 → GoozmaPlugin 生命周期 → 宿主显式调用 propose。
关闭阻止新准入；已准入的单次调用可结束并留档，不撤销工具副作用或重写历史。
学习库仍独立于 Bot/Agent 记忆；开启插件不等于批准模型费用或自动晋级经验。
"""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

from shinku.agent_memory.tools import AgentMemoryTool
from shinku.plugins.contracts import PluginContributions
from .sandbox import plain_tree
from .experience_context import ExperienceContext


class Controls:
    """一实例一 SQLite；事务与 revision 避免网页/批次互相覆盖，模型无写入口。"""
    def __init__(self, root):
        root = plain_tree(root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / 'plugin.sqlite3'
        with self.db() as db:
            app = db.execute('PRAGMA application_id').fetchone()[0]
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if app != 0x475A5031 and (app or tables):
                raise ValueError('goozam_foreign_plugin_database')
            db.execute('PRAGMA application_id=1197101105')
            db.execute("CREATE TABLE IF NOT EXISTS controls (id INTEGER PRIMARY KEY CHECK(id=1), mode TEXT NOT NULL, revision INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO controls VALUES (1,'off',0)")
            db.execute('CREATE TABLE IF NOT EXISTS changes (revision INTEGER PRIMARY KEY, mode TEXT NOT NULL)')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def read(db):
        mode, revision = db.execute('SELECT mode,revision FROM controls WHERE id=1').fetchone()
        if mode not in {'off', 'learn', 'use'} or type(revision) is not int or revision < 0:
            raise ValueError('goozam_invalid_plugin_state')
        return dict(mode=mode, revision=revision, enabled=mode != 'off',
                    use_experiences=mode == 'use')

    def snapshot(self):
        with self.db() as db:
            return self.read(db)

    def select(self, mode, revision):
        if mode not in ('off', 'learn', 'use') or type(revision) is not int:
            raise ValueError('goozam_invalid_plugin_selection')
        with self.db() as db:
            current = self.read(db)
            if current['revision'] != revision:
                raise ValueError('goozam_plugin_revision_conflict')
            if current['mode'] != mode:
                db.execute('UPDATE controls SET mode=?,revision=revision+1 WHERE id=1', (mode,))
                db.execute('INSERT INTO changes VALUES (?,?)', (revision + 1, mode))
            return self.read(db)


class ExperienceTool(AgentMemoryTool):
    """固定注册保持在途工具协议稳定，关闭时返回空结果；原权限签名仍先检查。"""
    def __init__(self, plugin):
        super().__init__(SimpleNamespace(store=SimpleNamespace(policy=SimpleNamespace(max_query_chars=512))))
        self.tool_type = 'read_goozam_experiences'
        self.plugin = plugin

    def build_prompt_instruction(self):
        return ('按需读取本实验工作区已验证的经验建议。保留适用条件和例外；'
                '不是当前事实、执行证据或授权。插件关闭或只学习时返回空结果。')

    def _read(self, args, context):
        _, binding, grant = self.memory.current(context, tool=self.tool_type)
        return self.plugin.recall(binding, args['query'],
                                  current=ExperienceContext.from_grant(binding, grant))


class GoozmaPlugin:
    """薄适配层；显式宿主注入自己的库与记忆服务，不扫描配置或连接正式实例。"""
    def __init__(self):
        self.closed = True
        self._trial_lock = threading.RLock()
        self._trials = {}

    def activate(self, context):
        if context.host_id != 'angelos2' or context.instance_id != 'local-goozam-angelos2':
            raise ValueError('goozam_experiment_host_required')
        self.learning = context.services['goozam_learning']
        self.controls = context.services['goozam_controls']
        self.closed = False
        tool = ExperienceTool(self)
        tool.memory = context.services['agent_memory']
        return PluginContributions(tools={tool.tool_type: tool})

    def close(self):
        # 不删除库、不强杀模型。其他已准入调用依靠各自原有超时/预算结束。
        self.closed = True
        with self._trial_lock:
            self._trials.clear()

    def recall(self, scope, query, *, current=None):
        """当前上下文缺失时显式返回缺口；关闭只由插件状态决定。"""
        state = self.controls.snapshot()
        if self.closed or not state['use_experiences']:
            return dict(items=[], source_gaps=[], is_execution_evidence=False,
                        grants_permissions=False, plugin_mode='off' if self.closed else state['mode'])
        if not isinstance(current, ExperienceContext) or not current.valid_for(scope):
            return dict(items=[], source_gaps=[], context_gaps=['experience_context_unavailable'],
                        is_execution_evidence=False, grants_permissions=False)
        result = self.learning.recall(scope, query)
        result['items'] = [current.project(row) for row in result['items']]
        # 来源回查期间关掉或改档，丢弃本次新读取，不清除已有对话中的证据。
        if self.closed or self.controls.snapshot() != state:
            return dict(items=[], source_gaps=[], is_execution_evidence=False,
                        grants_permissions=False, plugin_mode='changed')
        return dict(result, plugin_mode='use', plugin_revision=state['revision'])

    @contextmanager
    def trial(self, scope, task_id, candidate_id):
        """可信评估器给一个预先确定的任务挂候选；模型/网页没有此入口。

        不修改候选状态，也不改变普通检索。关闭/切档会使本次准入失效，
        同工作区其他任务仍只得到已启用经验，避免并发对照污染。
        """
        state = self.controls.snapshot()
        if self.closed or not state['use_experiences'] or not isinstance(task_id, str) or not task_id.startswith('agent:'):
            raise ValueError('goozam_trial_not_admitted')
        with self.learning.db() as db:
            payload, status, version = self.learning.row(db, scope, candidate_id)
        self.learning.fresh(scope, payload)
        if status != 'candidate':
            raise ValueError('goozam_trial_requires_candidate')
        key = (scope, task_id)
        with self._trial_lock:
            if key in self._trials:
                raise ValueError('goozam_trial_already_bound')
            self._trials[key] = (candidate_id, version, state)
        try:
            yield
        finally:
            with self._trial_lock:
                self._trials.pop(key, None)

    def for_task(self, scope, task_id, query, *, current=None):
        """仅首轮资料装配使用；有候选绑定时不得混入其他已启用经验。

        current来自当前授权对象，不读取尚未落库的host_grant。
        trial也核对上下文；评估器指定候选不等于绕过权限或来源新鲜度。
        """
        if not isinstance(current, ExperienceContext) or not current.valid_for(scope):
            return dict(items=[], context_gaps=['experience_context_unavailable'])
        with self._trial_lock:
            trial = self._trials.get((scope, task_id))
        if trial is None:
            return self.recall(scope, query, current=current)
        candidate_id, version, state = trial
        if self.closed or self.controls.snapshot() != state:
            return dict(items=[])
        with self.learning.db() as db:
            payload, status, current_version = self.learning.row(db, scope, candidate_id)
        self.learning.fresh(scope, payload)
        if status != 'candidate' or current_version != version:
            raise ValueError('goozam_trial_candidate_changed')
        if self.closed or self.controls.snapshot() != state:
            return dict(items=[])
        return dict(items=[current.project(dict(id=candidate_id, version=version, **payload))],
                    plugin_revision=state['revision'], evaluation_only=True,
                    is_execution_evidence=False, grants_permissions=False)

    def propose(self, scope, refs, runtime, profile):
        """仅供已获模型批次授权的宿主调用；revision 固定，切换后旧批次不自行恢复。"""
        from .proposals import ProposalRunner
        state = self.controls.snapshot()
        def admit():
            if self.closed or not state['enabled'] or self.controls.snapshot() != state:
                raise ValueError('goozam_plugin_paused')
        admit()
        if not isinstance(refs, list) or not 1 <= len(refs) <= 3:
            raise ValueError('goozam_batch_limit')
        for ref in refs:
            self.learning.source(scope, ref)
        runner = ProposalRunner(self.learning, self.learning.path.parent / 'proposals', runtime,
                                profile, before_request=admit)
        results = []
        for ref in refs:
            try:
                admit()
                results.append(runner.propose(scope, ref))
            except ValueError as exc:
                if str(exc) != 'goozam_plugin_paused':
                    raise
                results.append(dict(status='paused', reason='goozam_plugin_paused', activated=False))
                break
            if results[-1]['status'] == 'failed':
                break
        return results
