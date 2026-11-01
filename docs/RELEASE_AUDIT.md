# 2026-11-01 开发者预览发布验收

本次发布当前公共源码快照；不发布本机私用执行器、模型配置、数据库与历史回执。
本地未发布历史曾包含已删除的个人信息，因此公开快照以现有公开 main 为父提交，
普通快进更新；不携带私人中间提交，不改写已有公开历史。

包含当前 src/frontend、公共测试、脚本与使用指南。部分仅供维护者使用的历史计划
不新增到公开树；已有公开历史文档仅作历史参考。私用日志执行器测试移出公共套件。

必要打包修复：将 log-view.js 纳入 wheel，安装测试逐项验证 HTML 引用资源。
最终验收结果在本文件更新。历史全量数量不等于本版已全量通过。

复验命令：

```sh
python scripts/audit_shinku_release.py
python -m pytest tests/test_frontend_packaging.py tests/test_release_scope.py tests/test_cli.py
```

Linux 桌面批处理加 `python scripts/run_desktop_batch.py --` 前缀。
源码包与 wheel 通过 `scripts/audit_shinku_artifact.py` 检查；密钥和私人内容仍需独立人工审查。
未授权网络/模型评估不属于此离线发布检查。当前缺口见 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。

## 本次实际验收

- Linux / Python 3.13 环境，独立公共候选的 7926 项公共测试可收集；这是收集检查，未宣称全量运行通过。
- 公共候选定向 106 项通过：源码包到 wheel、离开源码目录后的 HTTP 前端资源、CLI、发布范围、
  工作台草稿/任务隔离、Agent 记忆边界和 Goozma 插件。
- 匿名化多人会话夹具 5 项通过；仅替换私人化文案和消息标识，不改变产品源码或验收断言。
- 首轮打包发现遗漏 log-view.js / sessions.js，修复后实际 HTML 引用全部返回成功。
- 发布前内容审查覆盖当前公开树、已知历史个人信息、私用导入和实际归档；
  自动匹配的测试密钥均为合成样例，未将真实数据或私用执行器纳入候选。
- 未调用真实模型、未发送 QQ、未加载正式服务；未做 Windows/macOS 实机与长期性能验收。
- 基础依赖由现有隔离验证环境提供；本次不是从空机器联网解析所有依赖的兼容性证明。

源码 ZIP 与 runtime wheel 已构建并分别通过产物边界检查；未发布到 PyPI，也未创建 GitHub Release。
