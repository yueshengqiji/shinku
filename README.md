# Shinku

面向 AI 助手的本地优先记忆与上下文管理运行时。

Shinku 的重点不是某个具体角色，而是把长期记忆、短期事件、上下文裁剪、工具授权和模型调用组织成可替换的运行时框架。它可以作为本地助手、聊天机器人或其他 Agent 应用的后端基础。

## 设计重点

### 记忆不是每轮全量注入

每一轮请求都经过受范围约束的记忆路由，而不是把整个数据库塞进提示词：

```text
incoming turn
  -> memory router
  -> ordinary chat: no unnecessary historical retrieval
  -> explicit recall: deterministic retrieval
  -> ambiguous request: lightweight routing
  -> scoped retrieval
  -> MemoryContext with sources
  -> model / agent prompt
```

记忆层分为：

- 原始对话与旁观记录：保留可追溯来源；
- 短期事件与阶段摘要：帮助承接近期话题；
- 已确认的长期事实：只保存相对稳定、可复用的信息；
- 遗忘记录：记录降权、过期和删除，避免旧信息永久占据上下文。

### 上下文管理

- 将近期对话、长期记忆、工具结果和系统状态分开管理；
- 按当前任务检索，而不是按关键词堆叠记忆；
- 工具返回结果默认不直接升级为长期记忆；
- 接近上下文上限时进行摘要、裁剪和来源保留；
- 对模型调用策略、超时、重试和输出长度集中管理。

### Agent 与工具边界

AgentLoop、ToolHost、能力清单、风险等级和人工授权是分开的。工具并不因为被声明就自动获得本机权限：

- 只读、低风险能力可以直接执行；
- 中风险能力按策略进入待授权队列；
- 高风险能力默认拒绝自动执行；
- 工具调用结果会带着来源和状态回到上下文管理层。

本次公开版本只保留通用能力边界和图片/视觉输入桥接，不附带本地 CLI、Shell、文件删除、浏览器 Cookie 或内网控制等高风险 MCP。详见 [`docs/MCP_SCOPE.md`](docs/MCP_SCOPE.md)。

## 当前版本包含

- Python 运行时、FastAPI 后端和 CLI；
- 多供应方 LLM 适配与统一调用策略；
- SQLite 记忆存储、长期事实、摘要和遗忘流程；
- AgentLoop、工具 schema、能力清单与授权边界；
- QQ/NapCat 图片输入与视觉引用桥接；
- 测试、发布审计、依赖锁定和独立运行文档。

## 安装

需要 Python 3.11 或更高版本：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

如果只运行后端，可以省略 `[dev]`。

## 使用

运行自检：

```powershell
shinku doctor
```

启动后端：

```powershell
shinku serve --service backend
```

健康检查：

```text
http://127.0.0.1:9998/health
```

供应方、模型和环境变量示例见 [`.env.example`](.env.example)。不要把真实 API key、Cookie、聊天记录或运行期 `data/` 提交到 Git。

## 文档入口

- [记忆与上下文设计](docs/MEMORY_DESIGN.md)
- [MCP 与外部能力发布范围](docs/MCP_SCOPE.md)
- [发布审计](docs/RELEASE_AUDIT.md)
- [第三方依赖与许可证](docs/THIRD_PARTY_LICENSES.md)

## 项目边界与版权

本仓库发布的是通用的记忆、上下文、Agent 和工具授权框架，不包含任何特定角色的人设原文、游戏台词、小说或剧本内容、配音、音乐、美术、Live2D、网络素材或第三方账号数据。

“Shinku”在这里是项目名称。本项目不主张任何相关角色、作品、名称、形象或其他原作材料的著作权，也不授予使用这些原作材料的许可。使用者如自行接入角色资料或外部素材，应自行确认来源和授权。

本仓库原创代码、测试和文档采用 Apache-2.0，见 [`LICENSE`](LICENSE) 和 [`NOTICE`](NOTICE)。第三方依赖仍受其各自许可证约束，清单见 [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md)。

## 开发与验证

```powershell
python -m pytest
python scripts/audit_shinku_release.py
python scripts/build_shinku_release_manifest.py
```

发布前还应检查没有把本地配置、运行期数据、凭据、角色素材或高危工具实现带入提交。
