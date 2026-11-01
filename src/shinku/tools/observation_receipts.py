"""复用原宿主观察凭据，只证明本请求实际取得某次工具观察。

大结果原文由既有任务材料保存，这里只签发摘要。凭据不证明安装、业务成功或
模型结论；工具可在 verify 后增加资源版本/新鲜度核对。进程重启后不恢复旧凭据。
"""
import hashlib
import json


class ObservationReceipts:
    def __init__(self):
        from shinku.agent.self_state_receipts import StateObservationStore
        self.store = StateObservationStore()

    def issue(self, data, *, tool, context):
        signature = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self.store.record(snapshot={'tool': tool, 'data_sha256': signature}, scope=tool,
                                 source='fresh_provider', gaps=[], context=context)

    def verify(self, receipt, *, tool, context):
        result = self.store.resolve(updates=receipt.get('observation'), arguments={'scope': tool}, context=context)
        return bool(result and result['valid'])
