# Linux 桌面后台资源策略

开发批处理通过 `scripts/run_desktop_batch.py` 启动：

```sh
python scripts/run_desktop_batch.py -- python -m pytest tests/test_desktop_batch.py
```

跨会话文件锁将整轮测试、审计或迁移命令串行化，默认最多等待 120 秒；等待超时返回 75，原任务继续运行。命令的工作目录、参数和退出码保留。Ctrl+C 或 SIGTERM 会回收所属 scope，避免留下后台子进程。执行入口没有自动发送消息、切换模型或修改业务数据的行为。

每轮命令的临时目录位于 `~/.cache/shinku-runtime/tmp/batch-*`，数值库线程设为 2。失败产物保留。显式指定 pytest `--basetemp` 会覆盖 Python 的临时目录选择，因此应使用磁盘路径。

本机宿主配置将开发 scope 和 `shinku-shell-*` transient services 放入 `shinku-background.slice`：合计 CPU 上限为 200%（两个逻辑核心），CPU/I/O 权重较低，内存软阈值为 2 GiB。`MemoryHigh` 用于回收和节流，不设置杀死任务的 `MemoryMax`。这限制实际资源竞争，并不把所有独立 shell 会话强制改成两个会话，也不修改已有 shell 会话的超时或权限规则。

正常后端不在这个资源组内。正在运行的后端无需为开发批处理策略重启；其服务中的磁盘临时目录配置在下一次正常启动生效。

用户环境中的 `TMPDIR/TMP/TEMP` 也指向磁盘缓存。已有进程可能缓存原环境，新的登录 shell 和 systemd 用户任务会获得新目录。旧 `/tmp` 产物仅在确认没有活动引用后复制校验并归档，不能直接清空 `/tmp`。

宿主级崩溃转储使用 `Storage=none`、`ProcessSizeMax=0`，保留崩溃元数据日志，停止自动复制整进程内存。该设置不会修复程序原本的崩溃；需要完整 core 调试时应临时恢复，结束后再启用桌面策略。

宿主配置、验证日志、归档清单及撤销工具由本次桌面维护任务单独保存，不将真实环境文件、运行数据库或私用源码纳入公共提交。
