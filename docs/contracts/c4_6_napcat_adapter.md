# C4-6：OneBot/NapCat 消息适配边界

## 目的

把 C4-4 的统一出站消息和 C4-1 的入站事件接到 OneBot/NapCat 的消息形状，验证
群聊、私聊、引用、文本和图片最终如何组成 action。网络客户端、WebSocket、token
和端口管理仍然属于外层宿主。

## 契约

- 入站只接受 `post_type=message`，再交给 `IncomingMessage.from_event()`；
- 群聊出站使用 `send_group_msg` + `group_id`；
- 私聊出站使用 `send_private_msg` + `user_id`；
- 引用、文本和图片保持同一个 `message` segment 列表；
- OneBot 错误统一转换为 `DeliveryResult` 的 rejected 状态；
- `NapCatActionTransport` 只调用注入的 `action_call`，不自己建立网络连接。

## 灰度口径

本批使用内存 action caller 和假 OneBot 事件，验证 payload 形状和错误转换。没有连接
真实 NapCat、QQ 账号、WebSocket 或 HTTP 端口。
