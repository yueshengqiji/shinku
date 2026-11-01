# 安装、配置与首次使用

本指南针对公共开发者预览版。先使用独立目录和空数据，不复制维护者的运行副本。

## 1. 安装范围

推荐从 Git 源码安装：按根 README 创建 venv 后执行 `python -m pip install -e ".[dev]"`。
运行依赖以 `pyproject.toml` 为准；`requirements-*.lock` 是特定历史环境快照，不能当作所有平台通用锁。
需要 MCP 或文档功能时，按实际使用安装 `.[mcp]`、`.[documents]`；基础安装不包含
embedding 模型权重、浏览器、Zenity、NapCat、私用执行器或第三方模型服务。
不要把旧项目 `.env` 或整份运行数据库直接放进新实例。

## 2. Angelos 工作台

```sh
python -m shinku.angelos --port 8770 --data-root "$PWD/private/preview/angelos"
```

访问 `/angelos`。无 `--factory` 时不连接 Agent：这是界面/数据入口，不是可执行任务的完整助手。
可以创建工作区、查看界面；发送任务需要先接入宿主。地址始终为启动服务的那台电脑的本机地址。
工作台数据库位于 `data-root/angelos/workbench.sqlite3`，不要与 Bot 数据目录混用。

接入真实任务：

```sh
# local_host 是使用者自己安装并审查的模块，不随公共仓库提供。
python -m shinku.angelos --port 8770 --data-root "$PWD/private/preview/angelos" --factory local_host:factory
```

工厂必须返回 `shinku.angelos.host.ServiceHost`，也可用上下文管理器 yield 它。
先构造 `AgentService`（任务存储、执行器、协调器、模型及已注册工具），再包装为
`ServiceHost`；启用四档权限须显式传入 `PermissionControls`，独立记忆须传入专用
`memory_root`。退出工厂时关闭协调器、模型客户端及扩展资源。
接口入口见 `src/shinku/angelos/host.py`、`src/shinku/agent_runtime/service.py`；
`src/shinku/agent_runtime/demo.py` 和 `tests/angelos_fixture.py` 是离线装配参考，
其中确定性模型不是联网模型，不应冒充真实回答。当前不提供通用“一键真实模型工厂”。

正式模型配置属于宿主工厂。不要假设 Angelos 启动器会自动读取 Bot 的 `.env`、模型或密钥。
模型目录只证明能列出名称；原生工具、流式返回、思考参数和上下文容量需对具体路由验证。
工厂应拒绝 root 或自行建立更强隔离，不能仅凭界面 ALEPH 标识断言操作系统权限安全。

## 3. 真红通用后端

从 `.env.example` 选择自己需要的配置，保存为不提交的 `.env`。CLI 只接受 `SHINKU_*` 键；
非空 shell 环境优先于文件。下面先使用独立路径，不配置模型、QQ 或扩展：

```sh
export SHINKU_ENV_FILE=/dev/null
export SHINKU_DATA_ROOT="$PWD/private/preview/bot/data"
export SHINKU_CONFIG_ROOT="$PWD/private/preview/bot/config"
export SHINKU_LOG_ROOT="$PWD/private/preview/bot/logs"
shinku doctor
shinku serve --service backend
```

访问 <http://127.0.0.1:9998/health> 和 <http://127.0.0.1:9998/debug/hub>。
Windows 用 `$env:SHINKU_DATA_ROOT=...` 设置对应变量，`SHINKU_ENV_FILE` 指向自己的空配置文件。
需使用项目 `.env` 时移除上面的 `/dev/null` 覆盖，并核对继承环境。
`doctor` 会创建所选配置/数据/日志目录；健康响应只证明服务启动，不证明模型或 QQ 已接通。
QQ 接入与实际发送另行配置，先核对账号和允许范围；不要把他人的登录态复制进仓库。

## 4. 记忆与 Goozma

- 真红：`src/shinku/memory/`；按作用域保存/检索，有可选语义索引与来源投影。
- Angelos：`src/shinku/agent_memory/`；独立数据库、检索与任务来源，不调用 Bot Router。
- Goozma：`src/goozam/`；经验只作建议，评估/启用/撤销与业务任务分开；默认启动不会自动接入。

语义能力需要额外模型和依赖，具体配置参考 `.env.example` 与记忆文档。不能把关闭语义时
词法检索通过当作语义效果验收。Goozma 实验准备脚本 `scripts/prepare_goozam_lab.py --help`
只建立隔离副本；目录隔离不是进程沙箱，也不会自动复用生产配置或回灌经验。

## 5. 更新、备份与回退

1. 记录 `git rev-parse HEAD`、工厂版本、依赖版本及数据目录；密钥单独保管。
2. 先在另一份检出和空数据中验证更新；生产操作安排在任务空闲时。
3. 备份配置、工作台库、Agent 账本、独立记忆与 Goozma 库。SQLite 使用备份 API，
   或停止自己的实例后一致性复制；不要只复制运行中的 `.sqlite3` 而遗漏 WAL。
4. 静态资源更新要刷新页面；后端变更需要重启。重启和取消都不承诺撤销已发生的工具操作。
5. 回退时使用匹配的源码和备份，在副本验证数据库兼容；不得盲目用旧版打开已迁移的新库。

排错时先看[已知限制](KNOWN_LIMITATIONS.md)。开发版本建议固定提交，不对正在运行的目录直接执行拉取更新。
