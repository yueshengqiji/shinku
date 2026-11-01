# 参与开发

先读根目录 AGENTS.md。保持改动范围明确，使用合成数据，提交完整的源码、测试和文档。
不提交密钥、真实会话、数据库、角色媒体或本机私用扩展；不要把真实故障原文直接复制到测试。

## 本地验证

在独立 venv 中安装 `.[dev]`。定向运行本次涉及的测试，先修原失败链路；
Linux 桌面批处理通过共享入口运行，避免并发回归抢占桌面资源：

```sh
python scripts/run_desktop_batch.py -- python -m pytest tests/test_cli.py tests/test_release_scope.py
python scripts/run_desktop_batch.py -- python scripts/audit_shinku_release.py
```

Windows 可直接运行对应 Python 命令。公共测试显式指定 `tests`，本机 `private/tests`
是可选私用测试，不属于公共检出。需要 Playwright 的页面测试必须自行安装相应开发依赖
与 Chromium；未安装导致跳过不代表已通过。构建测试还需要 `build`、setuptools、wheel。
离线测试不能读取用户已有模型配置，不能提交真实任务或发送 QQ；真实评估应单独明确范围与费用。

构建资源：

```sh
python -m pip install build
python -m build
```

前端编辑器修改后，在 `frontend/angelos/editor` 运行 `npm ci --ignore-scripts`、
`npm run build`；提交必要生成资源及许可证。发布检查须核对 wheel 中 HTML 引用的全部本地资源，
不能仅证明源码目录下页面可用。

## 评审重点

- 状态归属、来源、撤销、恢复和实际副作用应有明确边界；不要以模型自述替代回执。
- 修复要从用户入口复现；替身应遵循真实签名，测试不应只重复实现或迁就错误输出。
- 注释说明设计原因和中文维护入口；文档区分实现、测试、加载及真实使用效果。
- 不因为增加经验或补充材料而扩大权限，也不要把可选资料缺失变成无条件整轮失败。
- 当前使用说明放在 README/GETTING_STARTED/KNOWN_LIMITATIONS，内部计划不作为公开文档。

## 发布

公开文档只保留使用指南、架构/接口说明、许可证和必要注意事项。`docs/plans/`、任务计划、阶段进度、交接、演练及部署记录仅保留本地，不随公开源码推送；发布前检查最终 Git 文件树，不能只依赖 `.gitignore`。

源码开发快照与公开快照可分开保留。若未推送的本地历史包含私人信息，不能直接推整个分支；
在保留本地原件的前提下，从已公开基线建立审查过的最新源码快照，普通快进推送。
记录两者的对应关系，不强推、不冒充所有旧提交均已公开。公开快照也必须检查内容和构建依赖。
