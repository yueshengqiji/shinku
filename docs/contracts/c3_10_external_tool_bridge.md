# C3-10：外部工具宿主桥接边界

## 目的

QQ、浏览器、MCP 或其他独立进程不应把进程管理、网络客户端和密钥读取塞进 Agent
核心。本批定义一个轻量桥接层：外部宿主提供能力描述与注入式 `call()`，Shinku 将
能力描述包装成既有 `ToolHandler`，继续复用 `ToolRegistry`、`ToolHost` 和 AgentLoop。

## 契约

`ExternalToolDescriptor` 负责工具名称、说明和 object 输入 schema 的边界校验；
`ExternalToolTransport` 只规定：

```text
call(tool_name, arguments, context) -> result
```

`ExternalToolBridge` 把描述包装成 registry，不负责：

- 启动或停止 QQ、浏览器、MCP 进程；
- 读取 API key、token 或用户 cookie；
- 选择网络地址、端口或代理；
- 把外部异常伪装成成功结果。

参数校验和执行结果仍经 C3-3 的统一执行边界；外部 transport 抛出的异常由既有
handler error envelope 接住。

## 灰度口径

本批使用记录调用的内存 transport 和假 runtime，验证“描述→registry→ToolHost→
AgentLoop→外部调用通道”的两轮链路。没有启动真实进程、连接 QQ/浏览器/MCP 或发出
网络请求；后续宿主只需实现 transport，不应改动 Agent 核心。
