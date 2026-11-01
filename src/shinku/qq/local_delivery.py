"""桌面 HTTP / 本地终端的单轮送达回执，复用 Bot 记忆与会话终结。

传输适配器持有此对象：prepare 只接收候选；HTTP 响应体交给 ASGI 后，或终端
stdout 写入并 flush 后才 confirm。传输失败没有成功回执；成功不表示用户已阅读。
状态不跨请求共享，锁保证重复回调不会重复写入或强化记忆；transport 只由宿主设置。
"""
import threading

from .delivery import DeliveryResult
from .reply_delivery import QQDeliveryReceipt
from .reply_models import QQAgentReply


class DesktopReplyDelivery:
    def __init__(self, bridge, *, requester, transport="http"):
        if transport not in {"http", "terminal"}:
            raise ValueError("local_transport_invalid")
        self.transport = transport
        self.bridge = bridge
        self.requester = requester
        self._pending = None
        self._done = False
        self._lock = threading.Lock()

    def prepare(self, publication, dependencies):
        with self._lock:
            if self._pending is not None or self._done:
                raise ValueError("desktop_publication_already_prepared")
            self._pending = (publication, dependencies)
        return QQAgentReply(publication.agent_status, final_text=publication.final_text,
                            delivery_status="pending_local", reason=publication.reason)

    def confirm(self, returned_text):
        """确认本轮实际返回的正文；不接受客户端提供的发言或发送身份。"""
        with self._lock:
            if self._done or self._pending is None:
                return
            publication, deps = self._pending
            if returned_text != publication.final_text:
                return
            self._done = True
            self._pending = None
        ids = (("local-terminal:" if self.transport == "terminal" else "desktop-http:") + publication.turn.primary_message_id,)
        receipt = QQDeliveryReceipt(returned_text, ids, 1, True, False, ())
        reply = QQAgentReply(publication.agent_status, final_text=returned_text, sent=True,
                            delivery_status="sent", reason=publication.reason,
                            delivery=DeliveryResult(True, "sent", ids))
        try:
            # 身份撤销后停止记忆写入/强化；已经发生的传输仍需如实终结会话。
            if self.requester in self.bridge.master_user_ids:
                deps.accept_receipt(publication, receipt)
                if publication.record_memory and publication.agent_status in {"completed", "chat", "partial"}:
                    deps.finalize_usage(publication, returned_text)
        finally:
            deps.trace(publication.trace_id, "local_delivery", status="sent", source_kind="terminal_stdout" if self.transport == "terminal" else "http_response_body")
            self.bridge.trace_store.finish(publication.trace_id, reply=reply)
            self.bridge._conversation_finish(publication.trace_id, reply)

    def fail(self):
        """响应传输异常终结本轮，未知送达不得变成可跨渠道使用的发言。"""
        with self._lock:
            if self._done or self._pending is None:
                return
            publication, deps = self._pending
            self._done, self._pending = True, None
        reply = QQAgentReply("failed", delivery_status="uncertain", reason="local_transport_failed")
        deps.trace(publication.trace_id, "local_delivery", status="uncertain")
        self.bridge.trace_store.finish(publication.trace_id, reply=reply)
        self.bridge._conversation_finish(publication.trace_id, reply)
