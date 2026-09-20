# C1-3 能力提供方清单

## 当前边界

Shinku 独立版已经移除桌宠角色包、立绘、背景、BGM 和资源索引层。
因此本批只保留通用的能力提供方清单：它描述 MCP、插件或其他外部能力，
不描述任何桌宠前端素材。

保留入口：

- `src/shinku/capabilities/manifest.py`：读取并校验 YAML 能力清单；
- `src/shinku/capabilities/safety.py`：外部提供方的安全边界；
- `src/shinku/contracts/capability.py`：能力清单的数据契约。

已移除入口：

- `src/shinku/resources/`：角色包、情绪图片、背景、音频和桌宠提示词索引；
- 对应资源清单测试与桌宠资源契约。

## 能力清单最小形状

```yaml
schema: capability_adapter/v1
provider:
  id: local_tools
  type: mcp_stdio
  endpoint:
    url: http://127.0.0.1:9101
    loopback_only: true
capabilities:
  - id: read_file
    risk: low
    confirm: never
    effects: [file_read]
```

`load_manifest` 对外返回 `CapabilityManifest` 或 `InvalidManifest`，不会让单个坏清单
把整个宿主进程打崩。外部提供方的风险、确认方式、效果和端点仍由这层统一校验；
QQ、Agent 和记忆系统不依赖桌宠资源目录。

## 清理验收

运行时代码和测试不应再引用 `shinku.resources`、`ResourceManifest` 或桌宠资源入口。
`speech_segments` 等 LLM 回复字段不属于桌宠资源层，仍保留用于回复断句和宿主投递。
