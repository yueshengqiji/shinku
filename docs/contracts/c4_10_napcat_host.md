# C4-10：NapCat 宿主组合层

## 目的

把 C4-1～C4-9 的边界组合成一个可被 HTTP/WebSocket/QQ SDK 宿主调用的内核：
事件解码、路由与注意力、短窗口批处理、图片准备和出站投递由同一对象协调。

## 契约

- `NapCatHost.handle_event()` 接受 JSON body 或 mapping，返回入站结果和视觉准备结果；
- `NapCatHost.from_http()` 只配置 HTTP action caller，不在构造时发请求；
- `send()` 没有配置 transport 时返回明确的 `transport_unconfigured`，不生成假成功；
- `flush()` 复用既有注意力批处理，不重复调用模型；
- 网络监听服务器不属于本批，外层只需把收到的 body 转交给宿主；
- 所有 token、图片读取和远程下载仍由显式配置的 transport/materializer 控制。

## 灰度口径

测试覆盖宿主组合、入站图片准备、HTTP 延迟到 send 才调用、未配置 transport、短窗口
flush 和出站 action。测试使用 fake opener 和内存事件，不启动端口、不调用真实 QQ/NapCat。
