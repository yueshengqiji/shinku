# Shinku 正式切换与回滚手册

这份手册只描述下一步人工操作。自动化前置检查不会停止服务、修改 NapCat、改端口或
打开 QQ 外发。

## 切换前

1. 确认 [TARGET_REHEARSAL.md](TARGET_REHEARSAL.md) 已通过。
2. 执行：

   ```powershell
   python scripts/check_shinku_cutover.py
   ```

   必须同时看到 `OLD_READY=YES`、`NEW_READY=YES`、`RELEASE_READY=YES`。
3. 备份旧项目配置、旧记忆库、NapCat forward 配置和当前 `.env`，记录备份目录与时间。
4. 确认可以接受短暂停机；切换窗口内不做 persona、模型供应商或记忆结构改动。

## 切换顺序

1. 保持新服务的 QQ 外发开关为 `false`。
2. 停止旧 `9998`，确认端口释放。
3. 将独立版 `SHINKU_BACKEND_PORT` 临时设为 `9998`，启动独立 backend。
4. 通过 `/health` 和 `/health/ready`，再发送一条不外发的本地合成事件验证入站、记忆和
   Agent 链路。
5. 确认日志、记忆库和模型供应方均正常后，才由使用者单独确认是否打开真实 QQ 外发。

## 回滚条件与顺序

出现健康检查失败、记忆写入异常、模型配置异常、重复发送或任何无法解释的出站时：

1. 立即保持/恢复 `SHINKU_QQ_AGENT_SEND_ENABLED=false`；
2. 停止独立版 `9998`；
3. 恢复旧项目配置与 NapCat forward 配置；
4. 启动旧服务并检查旧 `9998/health/ready`；
5. 保留独立版日志和失败时间点，之后再单独分析，不在回滚窗口临时改代码。

当前状态：此手册已准备好，但本次没有执行正式切换，旧 `9998` 和当前独立 `19998`
仍然保持原状态。
