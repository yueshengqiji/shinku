# 隔离目标环境演练

日期：2026-09-20

这是在本机临时目录完成的目标机等价演练，不是对旧 `9998` 的切换。

## 安装与启动

- 从 `dist/shinku-0.1.0-source.zip` 解包；
- 新建 Python 3.11 虚拟环境；
- 按 `requirements-build.lock` 和 `requirements-cleanenv.lock` 安装；
- 从源码包安装 `shinku`；
- `shinku doctor` 正常，数据、配置、日志目录均指向临时根；
- 使用临时端口 `19997` 启动 backend；
- `/health` 返回 `status=ok`，`/health/ready` 返回 `status=ready`；
- 进程随后正常停止，当前 19998 不受影响。

## 记忆迁移

旧库：`%LOCALAPPDATA%\Shinku\users_data\shinku_memory_v01\shinku_memory_v01.db`

- dry-run 统计：聊天 4616、普通摘要 176、语义摘要 8、经验 8、状态 5；
- 实际写入仅落到临时目标库；
- 目标库检查：活动事件 4802，普通聊天不检索，明确回忆可检索且保留来源；
- 源库迁移前后 SHA-256 一致，确认导入过程只读源库。

## 结论

安装、启动、健康检查、迁移和回滚停止均通过。此演练没有启用 QQ Agent 外发，也没有
修改旧 `9998` 或 NapCat 配置。正式切换前仍需由使用者决定是否在真实运行窗口安排停机、
备份和回滚。
