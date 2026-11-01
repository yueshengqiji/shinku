"""带来源的经验候选与小批量评估；不更改任务结论、权限或用户陈述记忆。

宿主提供来源解析器及独立评估器，本模块仅拥有实验候选/版本状态。SQLite事务
串行化晋级和撤销；模型不能直接提交通过标记。每次使用重验原任务及评估来源。
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3

from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_memory.retrieval import terms


from .experience_context import task_observations, source_observations


@dataclass(frozen=True)
class Assessment:
    baseline: dict
    candidate: dict
    baseline_score: float
    candidate_score: float
    safety_passed: bool
    notes: str


class LearningStore:
    """一实例一库；scope由宿主给定，来源resolver不接收数据库路径或任意SQL。"""
    def __init__(self, root, resolver):
        root = Path(root)
        from .sandbox import plain_tree
        plain_tree(root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path, self.resolve = root / 'learning.sqlite3', resolver
        with self.db() as db:
            known = db.execute('PRAGMA application_id').fetchone()[0]
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if known != 0x475A3031 and (known or tables):
                raise ValueError('goozam_foreign_database')
            db.execute('PRAGMA application_id=1197092913')
            db.execute('CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, scope TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS audits (seq INTEGER PRIMARY KEY, id TEXT, event TEXT, payload TEXT)')

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

    def source(self, scope, ref):
        source = self.resolve(scope, ref)
        if not isinstance(source, dict) or source.get('scope') != scope or not source.get('fingerprint'):
            raise ValueError('goozam_source_unavailable')
        # 正文和工具回执继续保留于原账本；候选只存绑定与指纹，避免检索时反复
        # 把三个任务的完整上下文灌入新任务。缺少来源时拒绝，不能靠此摘要自证。
        return {k: source[k] for k in ('scope', 'fingerprint', 'input_hash', 'conditions_hash', 'accepted')}

    def propose(self, scope, ref, *, situation, conditions, advice, exceptions, expected_source=None):
        fields = dict(situation=situation, conditions=conditions, advice=advice, exceptions=exceptions)
        if any(not isinstance(v, str) or not v.strip() or len(v) > 2000 for v in fields.values()):
            raise ValueError('goozam_invalid_proposal')
        source = self.source(scope, ref)
        if expected_source is not None and source != expected_source:
            raise ValueError('goozam_source_changed_during_proposal')
        # 同一来源版本只有一个候选。再次处理不改写原稿，也不复活撤销记录。
        key = canonical_hash([scope, ref, source['fingerprint']])
        # 观察来自同一来源版本，只用于解释旧任务；不得作为新任务完成状态要求。
        record = self.resolve(scope, ref)
        if any(record.get(k) != v for k, v in source.items()) or self.source(scope, ref) != source:
            raise ValueError('goozam_source_changed_during_proposal')
        observations = task_observations(record.get('result'))
        payload = dict(fields, ref=ref, source=source, observation_version=2,
                       source_observations=observations, assessment=None)
        with self.db() as db:
            old = db.execute('SELECT payload FROM candidates WHERE id=?', (key,)).fetchone()
            if old:
                previous = json.loads(old[0])
                if any(previous[k] != v for k, v in fields.items()):
                    raise ValueError('goozam_proposal_conflict')
                # 最早无观察字段的记录保持幂等，不要求补写当时未保存的数据。
                if (('anchors' in previous or 'observation_version' in previous)
                        and source_observations(previous) != observations):
                    raise ValueError('goozam_proposal_conflict')
                return key
            if db.execute('SELECT count(*) FROM candidates').fetchone()[0] >= 2048:
                raise ValueError('goozam_candidate_limit')
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?)',
                       (key, scope, json.dumps(payload, ensure_ascii=False), 'candidate', 1))
            self.audit(db, key, 'proposed', {'source': source['fingerprint']})
        return key

    @staticmethod
    def audit(db, key, event, value):
        db.execute('INSERT INTO audits(id,event,payload) VALUES (?,?,?)',
                   (key, event, json.dumps(value, ensure_ascii=False)))

    @staticmethod
    def row(db, scope, key):
        row = db.execute('SELECT payload,status,version FROM candidates WHERE id=? AND scope=?', (key, scope)).fetchone()
        if row is None:
            raise ValueError('goozam_candidate_not_found')
        return json.loads(row[0]), row[1], row[2]

    def fresh(self, scope, payload):
        if self.source(scope, payload['ref']) != payload['source']:
            raise ValueError('goozam_source_changed')
        assessment = payload.get('assessment')
        if assessment:
            for label in ('baseline', 'candidate'):
                if self.source(scope, assessment[label]) != assessment[label + '_source']:
                    raise ValueError('goozam_evaluation_changed')

    def evaluate(self, scope, key, evaluator):
        with self.db() as db:
            payload, state, version = self.row(db, scope, key)
            if state not in {'candidate', 'no_gain'}:
                raise ValueError('goozam_not_evaluable')
            self.fresh(scope, payload)
        # 耗时评估在事务外执行；失败保留candidate，不改原业务任务或自动补跑。
        result = evaluator(json.loads(json.dumps(payload)))
        if not isinstance(result, Assessment) or type(result.safety_passed) is not bool:
            raise ValueError('goozam_invalid_assessment')
        if any(type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1
               for s in (result.baseline_score, result.candidate_score)):
            raise ValueError('goozam_invalid_score')
        base = self.source(scope, result.baseline)
        trial = self.source(scope, result.candidate)
        if (result.baseline == result.candidate or base['input_hash'] != trial['input_hash']
                or base['input_hash'] == payload['source']['input_hash']
                or base['conditions_hash'] != trial['conditions_hash']):
            raise ValueError('goozam_holdout_required')
        assessment = dict(baseline=result.baseline, candidate=result.candidate,
            baseline_source=base, candidate_source=trial, baseline_score=result.baseline_score,
            candidate_score=result.candidate_score, safety_passed=result.safety_passed, notes=str(result.notes)[:2000])
        # 质量须有严格改善；相等保留为无收益，不把通过格式检查算作学习成功。
        state = ('evaluated' if result.safety_passed and trial.get('accepted') is True
                 and result.candidate_score > result.baseline_score else 'no_gain')
        with self.db() as db:
            current, status, current_version = self.row(db, scope, key)
            if current_version != version or status == 'revoked':
                raise ValueError('goozam_concurrent_change')
            self.fresh(scope, current)
            current['assessment'] = assessment
            self.fresh(scope, current)
            db.execute('UPDATE candidates SET payload=?,status=?,version=version+1 WHERE id=?',
                       (json.dumps(current, ensure_ascii=False), state, key))
            self.audit(db, key, state, assessment)
        return state

    def activate(self, scope, key):
        with self.db() as db:
            payload, state, _ = self.row(db, scope, key)
            self.fresh(scope, payload)
            if state == 'active':
                return
            if state != 'evaluated':
                raise ValueError('goozam_evaluation_required')
            if db.execute("SELECT count(*) FROM candidates WHERE scope=? AND status='active'", (scope,)).fetchone()[0] >= 5:
                raise ValueError('goozam_active_limit')
            db.execute("UPDATE candidates SET status='active',version=version+1 WHERE id=?", (key,))
            self.audit(db, key, 'activated', {})

    def revoke(self, scope, key):
        with self.db() as db:
            self.row(db, scope, key)
            db.execute("UPDATE candidates SET status='revoked',version=version+1 WHERE id=?", (key,))
            self.audit(db, key, 'revoked', {})

    def recall(self, scope, query):
        """仅按工作区、启用和来源新鲜度准入；原任务结果不作为新任务门槛。

        本层供可信宿主/评估器检索，不签发工具权限。任务入口另核对当前上下文；
        旧格式在返回副本中投影为来源观察，不写回、不修改历史评估。
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise ValueError('goozam_invalid_query')
        matches, gaps = [], []
        with self.db() as db:
            rows = db.execute("SELECT id,payload,version FROM candidates WHERE scope=? AND status='active' ORDER BY id", (scope,)).fetchall()
            for key, raw, version in rows:
                payload = json.loads(raw)
                try:
                    self.fresh(scope, payload)
                    observations = source_observations(payload)
                except (ValueError, KeyError):
                    gaps.append(key)
                    continue
                score = len(set(terms(query)) & set(terms(payload['situation'] + ' ' + payload['advice'])))
                if score:
                    matches.append((score, dict(dict(id=key, version=version, **payload),
                                               source_observations=observations)))
        return {'items': [v for _, v in sorted(matches, key=lambda x: -x[0])[:3]],
                'source_gaps': gaps,
                'is_execution_evidence': False, 'grants_permissions': False}

    def batch(self, scope, proposals, proposer):
        """显式一次批次最多三个来源；模型提案回调只生成候选，不可自动晋级。"""
        if not 1 <= len(proposals) <= 3:
            raise ValueError('goozam_batch_limit')
        keys = []
        for ref in proposals:
            before = self.source(scope, ref)
            case = self.resolve(scope, ref)
            if self.source(scope, ref) != before:
                raise ValueError('goozam_source_changed_during_proposal')
            proposal = proposer(case)
            keys.append(self.propose(scope, ref, **proposal, expected_source=before))
        return keys
