"""主生成期间的本地来源/图片校验；不含模型审核、改写或再次审核。

快照一致只证明输入未变，不证明生成观点全部正确；权限与执行验收归原宿主。
"""
import hashlib
import json

from shinku.memory.context_sources import CONTEXT_DIAGNOSTIC_KEY, ContextSourceCheck


def capture_sources(model, context, *, phase="check", trace_id=""):
    """只回查提示中已经选中的定位符，不重新检索或触发语义提取模型。"""
    getter = getattr(model, "source_projection", None)
    sources = getter(context) if callable(getter) else []
    if isinstance(sources, ContextSourceCheck):
        diagnostic = sources.diagnostics()
        previous = context.get(CONTEXT_DIAGNOSTIC_KEY, {})
        # 只保留两个生成时点的固定原因和计数；不记录来源正文、ID 或模型输出。
        checks = dict(previous.get("checks", {})) if isinstance(previous, dict) else {}
        checks[phase] = diagnostic
        context[CONTEXT_DIAGNOSTIC_KEY] = {**(previous if isinstance(previous, dict) else {}),
                                          "checks": dict(list(checks.items())[-2:])}
        trace = getattr(getattr(model, "effects", None), "trace", None)
        if callable(trace):
            trace(trace_id, "context_sources", phase=phase, **diagnostic)
        if not sources.valid:
            return None
        sources = sources.snapshot
    if sources is None:
        return None
    images = model.user_images(context) if callable(getattr(model, "user_images", None)) else []
    value = [sources, images]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_dialogue_sources(turn, decision, *, context):
    """来源失效时只撤销候选，不追加生成，不把模型自述当作独立核验。"""
    if context.get("qq_dialogue_sources_changed"):
        from shinku.agent.loop import AgentDecision
        return AgentDecision.final("这轮上下文未通过来源校验，暂时无法据此回答。", outcome="failed")
    return decision
