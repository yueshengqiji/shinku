# C3-6：工具注册边界与 Agent 联合回归

## 本批交付

`ToolRegistry` 给工具宿主一个最小注册面：handler 需提供名称、参数归一化和执行
方法；Agent 只能拿到只读 handler 视图，并可按白名单生成 native schema。

联合回归把下列链路串起来：

```text
注册 handler → 生成 native schema → LLMPlanner 假 runtime
     → AgentLoop → execute_invocation → envelope 回灌 → 下一轮完成
```

## 边界

本批仍是离线联合回归，不启动真实模型、QQ、浏览器或后台进程。真实宿主接入时，
只需要把宿主实现注册进 `ToolRegistry`，不应把宿主线程和供应商配置写入 AgentLoop。
