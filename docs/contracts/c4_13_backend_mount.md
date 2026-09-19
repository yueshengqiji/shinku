# C4-13：后端应用挂载 NapCat webhook

## 目的

让独立后端应用在需要时挂载 NapCat 入站路由，同时保持默认最小暴露面。

## 契约

- `create_app()` 默认不暴露 `/events/napcat`，不会创建 NapCat 宿主或读取 token；
- 只有显式传入 `napcat_host` 才会 include webhook router；
- webhook 路径和入站 Bearer token 都由调用方显式传入；
- `app.state.napcat_host` 只在挂载时存在，便于启动器和测试确认实际装配；
- 本批只完成 FastAPI 应用挂载，不改变 uvicorn 端口和启动器，不自动连接 QQ。

## 灰度口径

测试验证默认 404、显式挂载、路径覆盖和 Bearer 校验；使用 FastAPI `TestClient`，不启动
真实后端端口，不连接 NapCat。
