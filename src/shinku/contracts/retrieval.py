"""The single envelope exchanged by Shinku's retrieval pipeline.

A retrieval decision and the verification pass that follows it are reported
together, as one immutable record, so callers never see a half-finished result.

Every field stays JSON-shaped on purpose.  A retrieval backend is free to attach
diagnostics of its own, and this contract still will not take on a database, an
embedding library or a model SDK as a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["RetrievalPipelineResult"]


@dataclass(frozen=True)
class RetrievalPipelineResult:
    """Full outcome of one retrieval decision plus its verification.

    All seven fields are required.  ``router_*``, ``retrieval_result`` and
    ``verifier_*`` hold backend-shaped dictionaries rather than typed objects so
    that a new backend can add diagnostics without changing this contract.
    """

    used_retrieval: bool
    confirmed_snippets: list[str]
    router_output: dict[str, Any]
    router_timing: dict[str, Any]
    retrieval_result: dict[str, Any]
    verifier_output: dict[str, Any]
    verifier_timing: dict[str, Any]
