# QQ 群聊轮次注意力

默认 `SHINKU_QQ_ATTENTION_MODE=turn_state`。明确 @、回复真红或直呼真红继续由原路由处理；群自主回复关闭、@其他人、纯图片等边界不会被分类器放行。

未点名消息先检查现有群聊短期快照、回复链、他人定向对话、真红最近**成功发送**的对象与时间。明确接真红上一句、同一说话人延续真红对话可进入候选；热闹群里的普通陈述不会随机抽中。安静群里的公开问题是候选，不保证发言：回复模型仍可返回 `<SILENCE>`。冷却只在自主回复实际发出至少一段后开始。

旧概率实现仅用于显式回退：`SHINKU_QQ_ATTENTION_MODE=legacy_probability`。`SHINKU_QQ_AUTONOMOUS_REPLY_PROBABILITY` 在默认轮次模式中不参与决策。`/at reply on/off` 及按群开关保持原有优先级。

## 本地分类器

分类器是可选层，默认不存在，也不会在没有标注数据时声称已训练。`AttentionDecision.as_dict()` 与注意力 trace 含有数字特征和模型分数，不包含原文。需要人工给历史消息标“真红此时自然该接”=1 或“不该接”=0。不能用旧机器人的实际回复与否作标签，因为那是旧随机采样的结果。

每行 JSONL：`{"label":1,"features":{"owner":1.0,...}}`。`features` 必须包含 `shinku.qq.attention_classifier.FEATURE_NAMES` 中全部字段。按时间排序后训练，末尾 20% 留作验证：

```powershell
python -m pip install scikit-learn
$env:PYTHONPATH='src'
python scripts/train_attention_classifier.py data/attention/human_labels.jsonl data/attention/model.json
```

至少需要 150 条人工标签、每类至少 30 条。模型只保存数值权重的 JSON，不加载 pickle。验证时重点看高置信候选的准确率、漏接率，以及各群的误插话；训练脚本会打印留出集指标。配置模型路径后默认仅 `shadow`：计算分数、保留原判断。只有人工确认灰测质量后，才设置 `SHINKU_QQ_ATTENTION_CLASSIFIER_MODE=active`。即使启用，明确对别人的对话等硬边界仍优先。模型不确定时回退轮次状态判断。
