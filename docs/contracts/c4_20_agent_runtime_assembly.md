# C4-20：独立 persona / runtime / QQ Agent 装配

## 目标

把独立主人设文件、`LLMRuntime`、`ToolHost`、`QQAgentBridge` 和回合调度器接成一条
可诊断的装配边界，同时保留安全默认值：Agent 默认关闭，真实 QQ 出站默认关闭。

## 契约

- 主人设只能由显式 `SHINKU_PERSONA_FILE` 指定；不存在时失败，不从旧项目路径回退；
- `.env` 与 shell 只读取 `SHINKU_*` 键，密钥可以判断是否存在但不能被 doctor 回显；
- `SHINKU_QQ_AGENT_ENABLED=true` 才装配 Agent；缺少主人设、端点、模型或鉴权配置时拒绝启动；
- `SHINKU_QQ_AGENT_SEND_ENABLED=false` 时 sender 强制为空，Agent 只能 dry-run；
- 回合调度器只在 Agent 显式开启时接入宿主；关闭时旧的 webhook 入站行为保持不变；
- 本批不重启 NapCat，不改动真实 QQ 出站开关，不读取旧项目配置或历史 persona。

## 验收

专项测试覆盖主人设 UTF-8 加载、无回退、配置边界和 dry-run 发送保护。
