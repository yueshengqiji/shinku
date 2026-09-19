# C3-7：工具宿主接线与 Agent 灰度回归

## 本批交付

`shinku.hosts.tool_host.ToolHost` 是本地工具宿主适配层：

- 读取 `ToolRegistry` 的只读 handler 视图；
- 生成当前允许工具的 native schema；
- 组装 `LLMPlanner` 与 `AgentLoop`；
- 提供宿主健康状态；
- 允许通过策略阻断高风险或暂不可用工具。

## 灰度口径

本批的“灰度”是离线宿主灰度：使用假 runtime、真实独立仓库 Agent 内核和注入式
handler，验证宿主边界、工具白名单、策略阻断和完整回灌链路。尚未启动 QQ、浏览器
或远程模型进程，因此不能把本批结果写成线上端口已接通。

真实宿主接入只需实现 `ToolHandler` 并注册到 `ToolRegistry`；不应把宿主进程管理、
供应商 API 密钥或 QQ 会话状态写入 `ToolHost`。
