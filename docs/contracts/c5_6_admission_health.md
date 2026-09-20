# C5-6：准入自动检查与健康接口收口

## 完成内容

- 新增 `scripts/check_shinku_admission.py`，自动检查运行时代码导入图、活动环境变量、
  来源台账覆盖和直接依赖许可证清单。
- 新增 `docs/THIRD_PARTY_LICENSES.md`，登记 `pyproject.toml` 的直接依赖及当前 clean
  environment 观察到的许可证。
- `/health` 改为消费 `shinku.health.build_basic_health_payload()`，配置快照和进程事实
  只有一个装配入口，不再维护两套 `pid/status` 来源。
- 本机 `.env` 的 QQ 出站默认值保持 `false`。

## 验收

准入审计结果：

```text
PASS import_graph
PASS active_env
PASS source_record
PASS license_inventory
```

全量回归 **1172 passed / 0 failed**；compileall、`git diff --check`、
`127.0.0.1:19998/health` 和 `/health/ready` 均通过。旧 `9998` 未修改。

许可证版本是审计时本机环境的观察值，发布前仍应针对最终锁定环境重新生成清单。
