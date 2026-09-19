# C4-7：NapCat HTTP action caller

## 目的

为 C4-6 的 `NapCatActionTransport` 提供一个真实可接线的 OneBot HTTP
action caller，同时把网络、超时和认证边界留在消息域之外。

## 契约

- `NapCatHttpActionCaller` 只在被调用时发起 `POST` 请求，构造对象不会联网；
- action URL 为 `base_url/<action>`，请求体是 UTF-8 JSON；
- 可选 `access_token` 只放入 `Authorization: Bearer ...` 请求头，不写日志、不拼进 URL；
- HTTP 错误和无效 JSON 转成失败响应，网络不可达转成连接异常，由 `deliver()` 统一归类为传输失败；
- caller 通过 `opener` 注入，离线测试不连接 NapCat、QQ 或任何端口；
- 具体群聊/私聊参数、引用和附件 segment 仍由 C4-6 的 `NapCatActionTransport` 负责。

## 灰度口径

本批只验证 HTTP 请求构造、认证头、超时传递、错误分类和路径边界。测试使用内存
response 与 fake opener，不使用真实 token，也不发起网络请求。
