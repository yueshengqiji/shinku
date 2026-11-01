"""记忆写入的窄准入检查，不触发动作，也不代替模型理解。

只接受当前用户整句中的明确请求，排除条件句、转述、引号和否定前缀。
句内被保存的内容仍完整保留，包括否定与适用条件；不能靠命中一个词执行。
不在此范围的说法明确拒绝，留给用户澄清，不另加模型分类调用。
"""
import re


def explicit_management(text, operation):
    if not isinstance(text, str):
        return False
    text = text.strip()
    # “记住不了/忘记了吗”是能力陈述或追问，不是本轮保存/遗忘指令。
    # 不检查正文中的“不允许安装”等内容，避免删掉本来要保存的否定条件。
    if re.match(r'^(?:请\s*)?(?:帮我\s*)?(?:记住|记下|忘记|忘掉)(?:不了|不得|了吗|了没|过|吗|么)',text):
        return False
    if operation == 'remember':
        return bool(re.match(r'^(?:请\s*)?(?:帮我\s*)?(?:记住|记下|记录下来|保存到\s*(?:Agent|agent)\s*记忆库)'
                             r'(?=[：:，,\s]|[^\x00-\x7f]|$)', text)
                    or re.match(r'^(?:please\s+)?remember\b',text,re.I))
    return bool(re.match(r'^(?:请\s*)?(?:帮我\s*)?(?:忘记|忘掉|从\s*(?:Agent|agent)\s*记忆库(?:中)?删除)',text)
                or re.match(r'^(?:please\s+)?forget\b',text,re.I))
