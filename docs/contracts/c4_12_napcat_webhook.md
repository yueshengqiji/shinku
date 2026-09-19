# C4-12：NapCat 入站 webhook 路由

## 目的

将 `NapCatHost` 暴露为可被独立 HTTP 服务 include 的 FastAPI router，完成真实接线前
的入站 HTTP 边界验证。

## 契约

- `create_napcat_webhook_router()` 只返回 router，不创建 app、不监听端口；
- `POST /events/napcat` 默认接收 JSON body，交给 `NapCatHost.handle_event()`；
- 可选 `event_token` 使用 `Authorization: Bearer ...` 校验，token 不回显；
- 非法 JSON、非对象 JSON 和非 message 事件返回统一 400，不暴露内部异常；
- 成功响应只返回 scheduled、visual_ready、visual_pending 摘要，不回显消息正文或图片引用；
- 出站消息仍由 `NapCatHost.send()` 处理，本路由不自动回复。

## 灰度口径

使用 FastAPI `TestClient` 覆盖成功入站、Bearer 校验、非法事件和路径配置；不启动真实
监听端口、不连接 NapCat、不调用 QQ 账号。
