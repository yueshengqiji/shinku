# D1：发布边界审计

## 当前结论

自动化边界检查已通过，项目许可证已确定为 Apache-2.0。D4 已完成最终依赖锁定、源码
发布包和 runtime wheel 的复验。许可证决策记录见 `docs/LICENSE_DECISION.md`，构建产物
记录见 `docs/RELEASE_ARTIFACT.md`。

## 已核对项目

- 独立代码、测试、脚本和文档均经过当前发布边界审计；
- `scripts/check_shinku_admission.py` 的四项准入检查全部通过；
- 直接依赖已登记在 `docs/THIRD_PARTY_LICENSES.md`；
- 独立仓库没有发现图片、音频、视频、字体或其他二进制素材；
- 仓库根的 `.env`、运行期 `data/` 和缓存目录不属于发布内容；
- NOTICE 已更新为当前 C 阶段实际内容，并明确旧项目只用于来源审计。
- `scripts/build_shinku_release_manifest.py` 已生成 201 个清洁发布文件，排除了密钥、
  运行期数据、缓存、egg-info 和媒体素材。
- 根目录 `LICENSE`、`pyproject.toml` 和 NOTICE 已统一为 Apache-2.0 发布口径。
- `requirements-cleanenv.lock` 固定了 cleanenv 的完整非 editable 依赖快照；
  `requirements-build.lock` 固定了本次验证使用的构建工具版本。
- `dist/shinku-0.1.0-source.zip` 和 `dist/shinku-0.1.0-py3-none-any.whl` 均通过
  `scripts/audit_shinku_artifact.py`；dist 被 `.gitignore` 排除，不进入源码仓库。
- [TARGET_REHEARSAL.md](TARGET_REHEARSAL.md) 记录了新虚拟环境安装、临时端口启动和只读
  记忆迁移演练，全部通过。

## 发布前仍需人工确认

1. 发布前仍需人工确认 persona、voice、游戏台词、美术和网络素材不被外部补入发布包；
2. 按目标机器重新安装锁文件并做一次启动、回滚和记忆迁移演练；
3. 在上述演练完成前不切换旧 `9998` 主入口。

执行命令：

```powershell
python scripts/audit_shinku_release.py
python scripts/build_shinku_release_manifest.py
python scripts/build_shinku_release_bundle.py
python scripts/audit_shinku_artifact.py dist/shinku-0.1.0-source.zip
python scripts/audit_shinku_artifact.py dist/shinku-0.1.0-py3-none-any.whl
```
