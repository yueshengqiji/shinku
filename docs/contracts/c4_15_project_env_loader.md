# C4-15：独立 `.env` 加载

## 目的

修复 `.env.example` 与 CLI 行为不一致的问题：用户复制成 `.env` 后，启动器能够读取
Shinku 配置，同时不会把 legacy 时代的变量带进新项目。

## 契约

- `shinku doctor` / `shinku serve` 启动前读取 `SHINKU_ENV_FILE` 指定文件，否则读取当前目录 `.env`；
- shell 环境变量优先于 `.env`；
- 只接受合法的 `SHINKU_*` 键，忽略 `COMPANION_*`、未知键和注释；
- 支持最小的单引号/双引号包裹值，不做变量展开；
- 不打印任何配置 value，token 只显示 set/unset；
- 库函数 `load_settings()` 不隐式读取文件，只有 CLI 明确进入启动流程时加载。

## 灰度口径

测试覆盖白名单、shell 优先、缺失文件 no-op、CLI doctor 读取和 token 脱敏；不启动真实
服务、不连接 NapCat。
