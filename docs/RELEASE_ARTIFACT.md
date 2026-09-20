# D4：发布产物记录

## 本轮产物

在 2026-09-20 使用当前独立仓库构建了两类产物：

| 文件 | 用途 | 结果 |
| --- | --- | --- |
| `dist/shinku-0.1.0-source.zip` | 含源码、测试、文档和锁文件的干净源码包 | 201 个文件，审计 PASS |
| `dist/shinku-0.1.0-py3-none-any.whl` | 只含 `shinku` runtime 包的安装包 | 85 个条目，审计 PASS |

哈希不写回源码包自身，避免产生自引用；需要核对时直接执行
`Get-FileHash dist\shinku-0.1.0-source.zip,dist\shinku-0.1.0-py3-none-any.whl -Algorithm SHA256`。

两者都排除了 `.env`、运行期 `data/`、缓存、旧项目目录和媒体素材。`dist/` 已被
`.gitignore` 排除；它是本机复验产物，不会被误当作源码提交内容。

## 依赖锁定

- `requirements-cleanenv.lock`：当前 cleanenv 的完整非 editable 版本快照，包含运行时和
  回归测试所需的包；不包含本地 editable 的 `shinku`，也不把 `pip` 当作项目依赖。
- `requirements-build.lock`：本轮实际验证构建所用的 `setuptools==84.0.0`、
  `wheel==0.48.0` 和 `packaging==26.3`。它们是构建工具，不是运行时依赖。

cleanenv 本身没有安装构建工具，因此本轮没有污染它；构建验证使用临时隔离目录加载
上述构建工具，然后销毁临时目录。

## 复验命令

```powershell
python scripts/build_shinku_release_manifest.py
python scripts/build_shinku_release_bundle.py
python scripts/audit_shinku_artifact.py dist/shinku-0.1.0-source.zip
python scripts/audit_shinku_artifact.py dist/shinku-0.1.0-py3-none-any.whl
```

D4 只完成发布边界和可安装产物验证，不切换旧 `9998`，也不改变当前 QQ 安全发送开关。
