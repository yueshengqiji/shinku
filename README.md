# Shinku

Shinku 的**独立实现**仓库。AI 角色陪伴运行时。

本仓库是**洁净室重建**产物：它不包含 `AkaneCompanionLab` 的任何源码、测试、配置或素材。
业务模块在 C 阶段按行为契约逐个重写后加入，每加入一个模块就更新 `NOTICE` 与
`docs/SOURCE_RECORD.md`。

## 当前状态（C4-20：Agent 装配边界已完成）

本仓库已经完成 C1～C4-20 的独立洁净室重写边界：LLM runtime、AgentLoop、ToolHost、
QQ/NapCat 入站、图片视觉桥、短窗口回合合并、独立 persona 加载和 QQ Agent dry-run
装配均已落地。当前仍处于灰度接线阶段，真实 NapCat 出站默认关闭，尚未替换旧项目的
9998 服务。

严格分类下，旧项目 183 个运行时文件里只有 2 个具备 `INDEPENDENT_KEEP` 依据，
其余是 106 个 `REWRITE_REQUIRED` + 36 个 `UNCONFIRMED`。按计划 B 的验收口径
（"不存在 `UNCONFIRMED` 或 `REWRITE_REQUIRED` 文件进入独立发布包"），
**B1 阶段一个业务文件都不能搬进来**。

最初 B1 交付的是：

| 内容 | 位置 |
| --- | --- |
| 命名契约（包名 / 环境变量 / 端口 / 路径 / 日志名 / 服务名） | `src/shinku/names.py` |
| 配置加载（不读旧项目的 `config.py` / `.env`） | `src/shinku/config.py` |
| 后端应用工厂 + 健康探针 | `src/shinku/api/app.py` |
| 命令行入口（`doctor` / `serve`） | `src/shinku/cli.py` |
| 五个宿主的落点（C 阶段填充） | `src/shinku/hosts/` |
| 准入规则：哪些分类允许进入本仓库 | `docs/ADMISSION.md` |
| 旧项目作为只读来源档案的说明 | `docs/SOURCE_RECORD.md` |

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**不需要** Akane 目录、`code_shared` 或任何旧项目的环境变量。这是 B 阶段验收条款。

## 使用

```powershell
# 看它到底解析到了哪套配置（数据根 / 端口 / 日志名）
.venv\Scripts\shinku.exe doctor

# 起后端
.venv\Scripts\shinku.exe serve --service backend
# 然后： curl http://127.0.0.1:9998/health
```

```powershell
# 测试
.venv\Scripts\python.exe -m pytest
```

## 命名约定

完整契约在 `src/shinku/names.py`，这里只列要点：

| 项 | 值 |
| --- | --- |
| 包名 / 发行名 | `shinku` |
| 环境变量前缀 | `SHINKU_` |
| 服务名 | `backend` · `response-host` · `time-manager` · `browser-host` · `agent` |
| 默认端口 | 9998 / 9995 / 9996 / 9997 / 9100 |
| 默认绑定 | `127.0.0.1`（要暴露到局域网必须显式开 `SHINKU_ALLOW_LAN`） |
| 关联 ID 头 | `X-Correlation-ID`（线路协议，不随项目改名而改） |
| 服务令牌头 | `X-Shinku-Service-Token` |

### 明确不继承的两个旧命名

1. **`COMPANION_HOST` / `COMPANION_PORT`** —— Akane 时代的命名。旧项目
   `companion_v01/service_supervisor.py` 至今仍在设置它们。新项目不使用，后端走
   `SHINKU_BACKEND_HOST` / `SHINKU_BACKEND_PORT`。设了旧键会被**忽略并告警**
   （`shinku doctor` 会打出来，测试里有覆盖）。
2. **`SHINKU_SERVER_HOST` / `SHINKU_SERVER_PORT`** —— 旧项目里它指的是
   **Agent(Java) 宿主**的 9100 端口，"SERVER" 与实际语义不符。新项目不继承这个歧义，
   Agent 宿主走 `SHINKU_AGENT_HOST` / `SHINKU_AGENT_PORT`。

代价是旧项目的 `.env` 不能直接搬过来。这是有意的——旧配置里混着 Akane 时代的键名，
搬过来等于把污染也搬过来。

### 数据根换了位置

旧项目默认把可变数据放在仓库内的 `users_data/`；新项目按平台约定放在用户目录：

- Windows：`%LOCALAPPDATA%\Shinku\{data,config,logs}`
- macOS：`~/Library/Application Support/Shinku/{data,config,logs}`
- Linux：`$XDG_STATE_HOME/shinku/{data,config,logs}`

要保留旧行为就设 `SHINKU_DATA_ROOT`。

## 许可

见 `NOTICE`。本仓库尚未确定发布许可——这属于 D 阶段"发布边界审计"的范围。
