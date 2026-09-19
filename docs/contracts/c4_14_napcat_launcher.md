# C4-14：NapCat 启动器接线

## 目的

把 NapCat 配置接入 `shinku serve --service backend`，让真实联调可以通过环境变量
开启，同时维持默认关闭和最小暴露面。

## 契约

- `SHINKU_NAPCAT_WEBHOOK_ENABLED` 非真值时，后端不创建 NapCatHost、不挂载 webhook；
- 开关开启但地址无效时，启动器返回退出码 2，不启动 uvicorn；
- 开关开启且配置有效时，启动器构造 `NapCatHost`、`NapCatImageMaterializer` 并挂载配置的 webhook 路径；
- 图片目录来自 `SHINKU_NAPCAT_IMAGE_ROOTS`，逗号分隔；为空时 materializer 不限制根目录，建议生产环境显式填写；
- action token 和 event token 彼此独立；doctor 只显示 set/unset，不打印内容；
- 启动器测试只 mock `uvicorn.run`，不启动真实端口、不连接 NapCat。

## 灰度口径

测试覆盖 doctor 脱敏、启用但配置无效时拒绝启动、启用且配置有效时挂载 webhook，
并复用原有 CLI/健康探针回归。
