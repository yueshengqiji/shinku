# C3-1：Agent 任务协议层

## 范围

本批只建立 Agent 任务循环会用到的无状态协议：

- 工作流、步骤、事件状态词汇与别名归一化；
- 工作流迁移校验，以及显式 reopen / cleanup 开关；
- 步骤、产物、文本列表的输入归一化；
- 步骤与产物的有序合并规则。

不包含数据库、任务工作区、模型调用、工具执行、QQ 路由和后台线程。后续 C3 批次只能通过这些稳定边界接入执行器，不能把持久化逻辑塞回协议层。

## 独立实现决定

- 任务域落在 `src/shinku/tasks/`，不沿用来源侧的扁平模块名。
- `contracts/tasks.py` 已承载 Worker 请求/汇报以及 Agent 工具白名单；本批只补充状态和 payload 规则。
- `status.py` 保留少量迁移期函数别名，方便后续接线，但新代码优先使用 `normalize_workflow_status` 与 `validate_workflow_transition`。
- 所有函数均为纯函数；输入异常只在明确的非法状态迁移处抛出 `ValueError`。

## 验收

`tests/test_contract_c3_1.py` 覆盖别名、默认值、迁移矩阵、显式重开/清理、步骤 ID、字段截断、产物去重和有序合并。C3-1 不宣称 Agent 执行循环已完成。
