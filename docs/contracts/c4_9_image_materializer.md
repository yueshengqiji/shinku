# C4-9：图片 materializer

## 目的

把 C4-8 的图片引用真正准备成视觉模型可消费的 data URL，同时控制远程请求、本地
文件读取和 media id 获取的边界。

## 契约

- 远程 URL 只允许 `http` / `https`，并通过可注入 opener 请求；
- 本地路径可通过 `allowed_roots` 限制，`file://` 和 Windows 路径都先归一化；
- `media_id` 只能通过宿主注入 loader 获取，不隐式调用 NapCat action；
- 读取有超时和最大字节数限制，超限、非图片或读取失败返回 unresolved；
- 返回值是 data URL，交给 C4-8 的 `NapCatVisualInputBridge`；未解析成功的图片仍保持 pending；
- 默认构造不联网、不读文件、不调用 media loader，只有显式调用 materializer 才执行。

## 灰度口径

测试覆盖远程 URL、本地 `file://` 路径、目录白名单、media id 注入、超限拒绝和视觉桥接。
使用 fake opener、临时文件和内存 loader，不使用真实 QQ、NapCat、token 或外部图片地址。
