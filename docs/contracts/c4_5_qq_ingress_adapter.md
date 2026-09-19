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
