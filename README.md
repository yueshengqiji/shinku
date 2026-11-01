# Shinku · 记忆与 Agent 工作台

本地优先的记忆、上下文管理和 Agent 运行时。包含真红 Bot 通用后端、独立的
Angelos 工作台，以及可选的 Goozma 经验学习实验框架（源码包名为 `goozam`）。

**当前定位：开发者预览版 / Alpha。** 适合源码研究、集成和隔离项目试用；尚不承诺
无人值守可靠性、跨平台全部功能或学习效果。请先读[已知限制](docs/KNOWN_LIMITATIONS.md)。
本次更新是源码发布，不代表运行中的服务已升级，也不提供开箱即用的私人模型账号或工具。

## 能力与边界

| 部分 | 公共源码提供 | 使用前提 |
| --- | --- | --- |
| 真红记忆 | SQLite 存储、来源、生命周期、检索、上下文投影 | 自行配置模型及可选语义依赖；真实记忆数据不随仓库分发 |
| Agent 运行时 | 任务账本、工具协议、恢复、权限、材料与结果记录 | 必须装配具体模型运行时和工具宿主 |
| Angelos | 工作区、独立会话/记忆、模型选择、代码视图、任务/用量展示 | 不带工厂启动时显示 Agent 未连接；仅可查看和管理界面 |
| Goozma | 独立经验库、候选、评估、启用与撤销接口 | 实验功能，需显式装配；不能据此宣称会自主进化 |
| Bot / QQ | 通用接入、来源处理和表达链路 | 另行配置 NapCat、模型、账号和消息发送范围 |

真红记忆与 Angelos 记忆分开存储、检索和授权。Goozma 经验只供参考，不授予工具权限，
不替代任务执行证据。模型说“完成”、命令退出码为零、文件存在，均不单独代表整项任务成功。

## 快速开始：先启动未连接模型的界面

需要 Python 3.11+。以下使用 Linux shell，在一个新检出目录运行；不要套用到正在服务的目录。

```sh
git clone https://github.com/yueshengqiji/shinku.git
cd shinku
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m shinku.angelos --port 8770 --data-root "$PWD/private/preview/angelos"
```

打开 <http://127.0.0.1:8770/angelos>。此时 **Agent 未连接是预期状态**，不会自动读取
其他实例的账号或提交模型请求。端口占用时换一个空闲端口，不要停止已有服务。
关闭终端中的服务可使用 Ctrl+C；保留数据目录即可保留工作台数据。

Windows PowerShell 的环境准备：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m shinku.angelos --port 8770 --data-root "$PWD/private/preview/angelos"
```

Windows 命令是入口说明，不代表本版全部工具已在 Windows 验证；系统路径选择器当前依赖 Linux/Zenity。
详细安装、Bot 后端与可信工厂接线见[使用指南](docs/GETTING_STARTED.md)。

## 真正执行任务前

Angelos 通过 `--factory module:callable` 接收可信本机 `ServiceHost`。
工厂负责模型、工具、权限控制器、专用数据目录及资源关闭；它是 Python 代码执行入口，
不要使用来源不明的工厂。仅在网页选中模型，并不会凭空安装执行器。

建议首次使用 HE 只读；需要写入时切换 WAW，核对任务后批准。
ZAYIN 仅对话，ALEPH 允许已装配并授权的工具免逐项确认。
**权限档位不是操作系统沙箱**；无 root 也不能阻止当前用户权限下的文件损坏。
对重要项目使用版本控制、独立工作副本与备份。

公开版本不包含私用 Shell/桌面控制实现、账号凭据、真实聊天和记忆数据库、私人角色资料。
公共工具注册接口、工作台人工文件编辑与私用执行器是不同层次，见[能力边界](docs/MCP_SCOPE.md)。

## 使用注意

- 默认仅绑定本机；不要把控制台或工作台直接暴露公网。这不是多租户服务。
- 发送的原话、附件、检索内容及工具输出可能提交给配置的模型供应商。使用前确认数据处理范围。
- 新工作区登记不等于扩大工具权限；新会话独立上下文不等于操作系统隔离。
- 编辑器未保存草稿不能依赖刷新恢复；查看文件差异不代表差异一定由 Agent 产生。
- 上下文用量可能是估算，容量取决于实际模型路由。缓存命中不是答案正确率。
- 恢复前核对已发生的副作用；取消不等于回滚。不要反复点击继续来解决余额、协议或权限错误。
- 模型、embedding、MCP 与 QQ 都是可选外部连接；配好之后才可能产生调用费用或实际发送。
- 具体缺口、日志完整性与版本限制见[已知限制](docs/KNOWN_LIMITATIONS.md)。

## 文档

- [安装与首次使用](docs/GETTING_STARTED.md)
- [已知限制与排错](docs/KNOWN_LIMITATIONS.md)
- [安全与隐私](SECURITY.md)
- [参与开发与测试](CONTRIBUTING.md)
- [Angelos 工作台](docs/operations/angelos.md) · [独立记忆](docs/operations/angelos_memory.md)
- [记忆设计](docs/MEMORY_DESIGN.md) · [完整文档导航](docs/INDEX.md)
- [发布边界与验收](docs/RELEASE_AUDIT.md) · [第三方许可证](docs/THIRD_PARTY_LICENSES.md)

日期计划、旧测试数量和部署记录只证明当时版本。它们不是当前安装指南，也不保证本机路径在别人的环境存在。
源码更新后，新页面需要重新加载资源；后端 Python 修改必须由使用者在合适时机重启自己的实例。

## 许可证与内容

原创代码、测试和文档采用 Apache-2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
第三方依赖遵守各自许可证，CodeMirror 依赖清单随前端保留。
本项目不分发或授予任何第三方角色台词、声线、图片、游戏素材、字体或账号数据的权利。
“Shinku”是项目名称；使用者自行接入的资料需要自行确认来源和许可。
