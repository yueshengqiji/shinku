# C4-8：NapCat 事件解码与图片视觉引用桥

## 目的

把 NapCat webhook 的 JSON body 变成统一 `IncomingMessage`，再将图片附件明确分成
“已准备为模型 data URL”和“仍待解析”两种状态，避免图片丢失后错误走纯文本兜底。

## 契约

- `NapCatEventDecoder` 接受 JSON bytes、文本、mapping，拒绝非法 JSON、非对象 JSON 和非 message 事件；
- `NapCatVisualInputBridge` 保留每张图片的顺序和原始引用，不去重、不折叠为 `[图片]`；
- 没有注入 `materialize` 时，不自动联网、不自动读本地文件，图片保持 pending；
- `materialize` 是唯一的图片下载/读文件边界，可返回 data URL、bytes 或带 `data_url` 的 mapping；
- 只有合法 `data:image/...` 才进入 `model_images()`，未准备好的图片必须由上层决定重试、等待或提示用户；
- 该批不负责视觉模型调用，也不负责监听端口。

## 灰度口径

测试使用内存 JSON 和注入式 materializer，覆盖事件体解码、图片引用保留、顺序保持、
多图不去重、pending 状态和 bytes→data URL 转换；不访问真实 QQ、NapCat、网络或本地图片。
