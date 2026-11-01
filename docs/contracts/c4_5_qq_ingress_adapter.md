# C4-5：QQ 入站适配器组合边界

## 目的

把已经独立完成的消息归一化、回复路由和注意力批处理串成一条入站链路，避免真实
QQ 适配器重复实现判断逻辑，造成一边能看见图片、一边把图片变成占位符，或一边
过滤群聊、一边又把群聊送进模型。

## 契约

`QQIngressAdapter.ingest(event, at=...)` 固定经过：

```text
原始事件 → IncomingMessage → ReplyRoutePolicy → AttentionPolicy → AttentionBatcher
```

返回 `QQIngressResult`，包含归一化消息、路由决定、注意力决定，以及因会话切换或
窗口超时而关闭的上一批消息。适配器不连接 QQ SDK、不调用模型、不发送回复；真实
NapCat/QQ 客户端只负责把事件转换为 mapping 并消费批次。

## 灰度口径

本批使用内存事件和确定性随机源，验证普通群聊过滤、@ 图片保留、连续请求合并、
主动回复开关和跨会话隔离。没有登录 QQ、连接 NapCat 或发送网络请求。


## QQ 卡片与转发资料

`IncomingMessage` 保留 `json/xml/share/contact/location/forward/node` 片段及原始 payload。
`text` / `text_without_at_mentions` 仍表示当前发言的正文，供控制与权限判断；
`card_context` 是宿主解析的外部引用资料，`readable_text` 合并正文、来源和资料供只读使用。
不接受网络事件自报的 `card_context` 或“读取成功”状态。

正式宿主的顺序是：解码 → 准入/去重 → 卡片和引用回读 → 旁观观察 → 路由/批处理。
`NapCatHost.from_http/from_config` 只向卡片读取器注入 `get_forward_msg`；收到的卡片
URL、邀请链接和转发文件路径不会成为读取器的请求目标。每个事件独享请求和内容预算，
不同账号、会话之间不共用解析缓存；重复或拒绝事件不触发回读。

聊天 transcript 和群聊短上下文接收资料；独立 Agent 通过原有 `MaterialSource`
收到同一资料及原消息来源。卡片/转发中的作者是被引用的声明，不能改变当前请求者、
权限或控制状态。字段截断、过期和正文缺失保留结构化状态，并传递资料包 `complete`。
合并转发中的图片、语音、文件只标出存在和尚未读取；视频卡片标题与简介不代表视频正文。
消息中的邀请卡片与 OneBot 平台邀请请求是不同事件，此契约不接受邀请或发送消息。

公共回归见 `tests/test_qq_cards.py`：使用合成 OneBot 事件、真实 HTTP action caller、
宿主、调度器、资料存储和 bridge，验证内容入包与控制边界；模型和平台送达均由隔离夹具替代。
