# 插件运行接口 v1

## 目的与行为原则

插件管理器管理接线和资源，不审查用户请求。不新增关键词、步骤数、任务类别或
复杂度门槛，不额外调用模型，不重复审批。普通公共插件不自动变成私用扩展；
原私用扩展的主人边界保持不变。工具仍经过原 ToolHost/执行策略/任务与核验链。

`PluginManifest.version` 是展示标签，可以是自用版本、开发版本；未知 JSON 元数据
保留。`api_version` 是实际接口主版本，无法接入时只报告该插件不可用。必须依赖它的
插件也不可用，其他插件和聊天继续工作。不自动启用、安装、重试或替换插件。
可选清单文件损坏或不存在时，诊断中形成 plugin-configuration 故障项；原宿主照常
启动，不把可选插件配置问题变成全局拒绝。需要该插件的操作仍如实报告能力不可用。

## 最小插件

插件工厂为无参 callable，返回提供 `activate(context)` 和可选 `close()` 的对象。
activate 返回 `PluginContributions` 或 None。贡献复用原 handler 和 ChatReadCapability；
可提供 tools、chat_read_capabilities、prompt_notes。插件没有回合审查钩子。
工厂应只创建对象，连接、线程等资源放在 activate 内创建，使部分启动失败也能 close。

- `PluginContext`：宿主/实例身份、宿主配置、插件自身配置、可选状态根和显式 services。
  目录不自动创建，记忆和其他依赖由宿主选择注入。配置中的权限描述不构成授权。
- `PluginDefinition`：manifest、factory、enabled、插件配置。未启用/不属于当前宿主的
  定义不导入。factory 接受 callable、`module:callable` 或 `entrypoint:名称`。
- Entry point 组固定为 `shinku.plugins`，仅查找显式选中的名称；重名报告不兼容。
  独立包可在 pyproject 的 `[project.entry-points."shinku.plugins"]` 声明工厂。
- `PluginManager`：一个宿主实例一个管理器，持有运行记录和注册项，不全局缓存。

本机时间示例位于 `src/shinku/plugins/examples/local_clock.py`，选择清单位于
`examples/plugins/local-clock.json`。它读取本机时钟，不发网络请求。

## 接入现有启动器

可信启动器可将原 bindings 工厂包装一次，以后增减插件只改它显式选择的清单：

```python
from shinku.cli import main
from shinku.plugins.host import with_plugins

# original_factory 是本机原有 QQAgentHostBindings 上下文工厂；不在插件中重建模型。
factory = with_plugins(
    original_factory,
    plugin_config_path,  # 启动器指定的 JSON 绝对路径
    host_id="shinku",
    instance_id="my-instance",
)
raise SystemExit(main(["serve", "--service", "backend"], agent_bindings=factory))
```

此示例说明接线方式，不授权启动服务或发送消息。插件不配置 sender、模型、权限模式
或委托目标。原工厂资源先创建、后关闭，插件后创建、先关闭。旧 RuntimeExtension 的
工具、人设、命令和运行时绑定通过适配原样调用；旧扩展本身配置错误仍沿用原处理。

也可以用 PluginManager 上下文直接创建 runtime_extension，传入 QQAgentHostBindings
的同名字段。公共 assemble_qq_agent 支持该显式对象，不从用户消息加载插件。
未传该对象时原 SHINKU_RUNTIME_EXTENSION_MODULE 的启动语义保持不变。

## 生命周期、诊断与停用

依赖先激活，插件准备成功后在宿主启动装配时发布工具。同名工具不覆盖原对象；
发布失败只撤回本插件已注册的项，并停用真正依赖它的插件。旧 handler 的后续调用
受生命周期包装器控制，参数、context、工具元数据和原执行结果保持原样。

`disable(id)` 先停依赖者，再阻止自身新调用并撤回目录；已有调用允许结束，结束后
才关闭资源。依赖者尚未排空时，被依赖者的资源也不能提前关闭。`close()` 幂等。
同进程代码不能由管理器强杀；长任务取消与副作用仍由原任务执行链负责。

diagnostics 中的 active 表示插件资源准备成功，tools 是已发布的本插件工具列表；
disabled/inactive/failed/draining 分别表示关闭、不适用、故障和仍待排空。closed 表示
管理器已接到关闭请求；cleanup_pending 才能区分还有资源等待关闭。cleanup_error
明确记录清理失败，不能把它报告为清理成功。异常正文不进入诊断。

本版本的生命周期变更由宿主串行发起，不支持在线并发装载/热替换；执行线程可以
并发调用已发布的工具，并在停用期间正常结束。公共插件代码不是安全沙箱。使用
第三方不可信代码时，继续使用独立进程/既有外部工具桥，并按需求施加 OS 隔离。

## 当前验收范围

合成用例通过真实 CLI、QQ 装配、HTTP 桌面入口和真实只读工具执行链；模型为离线
合成响应，不连接模型服务。验证插件故障隔离、依赖、冲突回撤、在途排空、原请求
透传、公共/私用主人边界、原执行策略及宿主退出清理。生产加载和真实送达另行验收。
