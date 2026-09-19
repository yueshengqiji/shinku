# C4-11：NapCat 配置与安全预检

## 目的

为真实 NapCat 接线提供独立配置入口和不联网诊断，避免把 QQ 端口、token 或旧项目
环境变量硬编码进消息宿主。

## 配置键

- `SHINKU_NAPCAT_BASE_URL`：OneBot HTTP action 根地址，例如 `http://127.0.0.1:3000`；
- `SHINKU_NAPCAT_ACCESS_TOKEN`：可选，直接 token；
- `SHINKU_NAPCAT_ACCESS_TOKEN_FILE`：可选，token 文件路径，直接 token 优先；
- `SHINKU_NAPCAT_TIMEOUT`：HTTP 超时秒数，默认 10。

## 契约

- `NapCatConnectionConfig.diagnostics()` 只返回端点、配置状态、错误和 token 是否存在，绝不返回 token 内容；
- 只允许 `http` / `https`，拒绝 URL 用户信息、缺失 host 和非正超时；
- `NapCatHost.from_config()` 在配置无效时立即失败，配置有效也不会自动发请求；
- 真实请求仍要等到 `NapCatHost.send()`，没有 transport 时返回明确的未配置状态；
- `.env.example` 只记录键名和示例地址，不包含真实账号信息。

## 灰度口径

测试覆盖环境变量解析、token 脱敏、无效配置拒绝、token 文件读取和“构造不联网、send 才联网”。
测试使用假 opener；没有连接真实 NapCat、QQ 账号或端口。
