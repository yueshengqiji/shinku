# Shinku 直接依赖许可证清单

这份清单只登记 `pyproject.toml` 中的直接依赖；传递依赖仍由发布构建和锁定环境
另行导出。Shinku 自身原创代码采用 Apache-2.0，但不会覆盖这些第三方依赖各自的
许可证。版本号是本机 C5-6 审计时的观察值，发布前应重新核对实际安装版本的发行包
许可证文件。

| 包名 | 当前观察版本 | 许可证 | 用途/来源 |
| --- | --- | --- | --- |
| fastapi | 0.141.1 | MIT | HTTP 应用和路由 |
| uvicorn | 0.53.0 | BSD-3-Clause | ASGI 服务启动 |
| pyyaml | 6.0.3 | MIT | 能力清单 YAML 解析 |
| tzdata | 2026.4 | Apache-2.0 | Windows 时区数据库 |
| openai | 3.16.0 | Apache-2.0 | OpenAI 兼容端点客户端 |
| requests | 2.34.2 | Apache-2.0 | Anthropic HTTP/SSE 传输 |
| pytest | 9.1.1 | MIT | 开发测试工具 |
| httpx | 0.28.1 | BSD-3-Clause | 开发测试 HTTP 客户端 |

审计依据为本机 clean environment 中各发行包的 `*.dist-info/licenses/` 文件和
元数据；本项目不复制这些依赖的源码。发布时应把相应许可证文本随依赖分发方式
一并保留，并重新生成版本清单。
