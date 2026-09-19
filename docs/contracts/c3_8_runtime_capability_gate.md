# C3-8：Agent runtime 能力闸门

## 目的

C3-7 已经能把工具宿主和 AgentLoop 接起来，但“宿主注册了工具”不等于“当前模型
供应方支持 native tool calls”。如果这两件事混在一起，模型可能收到工具 schema，
实际请求却被 runtime 静默丢弃，随后出现假装调用、重复调用或错误汇报。

## 契约

`ToolHost.runtime_health(runtime)` 读取 runtime 的可选能力探针：

- 返回 `True`：宿主可以暴露 native tools；
- 返回 `False`：本轮不暴露工具，宿主状态为 `runtime_native_tools_unsupported`；
- 没有探针或探针异常：保留 `None`，兼容离线 runtime 与未来宿主，状态标为
  `runtime_capability_unknown`，不把未知能力误报为确定支持。

`ToolHost.build_loop()` 与同一探针使用相同决定：能力明确为 `False` 时，同时清空
发送给模型的 native schema 和 AgentLoop 的 handler 映射。这样“模型不可见”和“实际
不可执行”保持一致；能力未知时沿用注入式 runtime 的兼容路径。

本批不自动把 native tools 降级成高密度 prompt 指令。需要 legacy JSON 工具调用的
供应方，应在后续批次单独实现并测试对应 planner 协议，不能让能力闸门偷偷改变输入面。

## 灰度口径

测试使用真实独立仓库的 `ToolHost`、`AgentLoop`、handler 协议和一个明确返回“不支持”
的 runtime 桩，没有发起远程请求，也没有连接 QQ、浏览器或后台进程。它证明的是
能力协商和执行边界一致，不代表任何实际供应商已经支持工具调用。
