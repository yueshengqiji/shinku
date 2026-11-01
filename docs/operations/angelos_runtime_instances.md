# Angelos 只读运行实例查询

该能力由可信 ServiceHost 工厂显式安装，不调用 LLM、不探测模型生成、不修改配置或重启服务。默认不安装；网页、模型及普通工作区文件不能改变实例登记。它不读取 Bot 或 Agent 记忆。

## 装配

```python
from shinku.angelos.instance_registry import InstanceRegistry

instances = InstanceRegistry.from_file('/absolute/private/runtime-instances.json')
host = ServiceHost(service, runtime_instances=instances, permissions=permissions,
                   reasoning_path=data / 'reasoning.sqlite3')
```

登记文件由当前运行用户持有，不能组写或全局可写。实际登记放私用目录，不提交身份/运行路径；下面仅为合成示例：

```json
{
  "schema_version": 1,
  "instances": [{
    "id": "fixture-chat",
    "display_name": "合成聊天实例",
    "service": "fixture-chat.service",
    "endpoint": "http://127.0.0.1:19001",
    "adapter": "console",
    "model_store": "/absolute/fixture/model_services.json",
    "store_format": "model_services",
    "data_root": "/absolute/fixture",
    "log_path": "/absolute/fixture/logs"
  }]
}
```

`adapter` 支持 `console`（`/debug/hub/state` 的 `agent.live_models`）与 `angelos`（`/angelos/api/models` 的 `models.selected`）。后者报告实例默认选择，不代表某个会话的选择。`store_format` 支持 `model_services`、`angelos_preferences` 和平面环境键形式的 `settings`。不支持的响应/结构返回未知，不扫描别的文件猜值。

登记身份不等于现场证明。当前版本仅查询 Linux **user** systemd 服务；必须能将该服务主PID或同unit cgroup子进程的监听socket关联到登记的IPv4本机端口，才读取模型接口。其他用户、容器、system服务、子进程不在同unit cgroup或无法读取 `/proc` 时保守报告 `endpoint_owner_unconfirmed`，不借用邻居状态。后续可新增显式适配器，不能静默放宽归属核验。

## 工具与语义

- `list_runtime_instances()`：小型登记目录，`is_live_state=false`；不发任何网络请求，不声明在线。
- `inspect_runtime_instance(instance_id, sections=["models"])`：服务/端口归属与运行、保存模型；可加 `locations` 返回登记的配置/数据/日志位置。模型不能提交URL、SQL或文件路径。
- `running` 是当前服务报告值；`saved` 是磁盘保存值；缺失为 `null`，不互相补齐。`consistency` 比较两边有效聊天模型标识及双方均可用的供应商/辅助模型字段，`comparison_fields`列出实际比较范围，返回 `matched/different/unknown`，不把不同直接归因为“未重启”。
- 查询前后PID/InvocationID不同，丢弃拼接的运行值并报告换代缺口。结果带观测时间；后续核验只证明当时读取，不能保证未来不变。
- `locations.source` 明确为登记路径，不声称从运行进程证明了所有加载路径。反代最终上游未知；模型目录/白名单不等于当前调用选择。
- 配置/HTTP仅投影已知短模型和供应商字段；不返回认证信息、环境变量、进程参数或异常正文。HTTP禁用代理与重定向，限制本机固定路径、响应体和等待时间。
- 现有 `read_service_logs` 附上登记服务归属、历史PID/调用代次；未知服务不猜实例，旧快照不修改。

HE可查询，ZAYIN禁用；WAW的只读查询不产生执行审批，显式宿主deny仍优先。读取前后复查任务授权，观察凭据绑定请求/权限；不新增“未调用即失败”的验收条件。缺口允许普通回答如实交付。

## 验收边界

`tests/test_runtime_instances.py` 使用隔离本机HTTP、真实文件/socket归属与固定模型SDK夹具，验证多实例、配置冲突、撤权、快照篡改和纠正历史进入同一原生循环；固定回答不能证明真实模型的辨别能力。相关回归还覆盖日志原文与现有权限/工作台流程。部署与真实模型测试需分别明确范围，不能因公共源码提交就宣称运行服务已加载。
