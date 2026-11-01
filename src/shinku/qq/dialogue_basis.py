"""主生成的来源边界提示；删除事后裁判、标签分类和重写协议。"""

PERSONAL_HISTORY_BOUNDARY = '【个人历史依据】当前想法、愿望或情绪不证明长期习惯。对某人的偏好、频率、过去表现、共同经历或曾经所见的判断，必须来自实际可用原话/材料，保留原作者、时间和条件。没有依据就只承接当前意愿；改成感叹、反问、比较或委婉语气，也不能补入历史前提。'


def instruction(context):
    """依据边界直接给主生成，不要求另一份分类，也不派发模型审核。"""
    context["qq_dialogue_basis_required"] = False
    context.pop("qq_dialogue_basis", None)
    return (PERSONAL_HISTORY_BOUNDARY +
        "历史助手仅证明说过，不证明执行、共同经历或用户动机。保留来源的作者、时间、否定与条件；"
        "没有本轮图片/声音时不要按对象类别补造所见。无法确定时自然说明不确定，不宣称已独立核验。")
