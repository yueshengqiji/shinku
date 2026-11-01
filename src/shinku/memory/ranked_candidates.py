"""Fuse independent ranked locators before consuming a shared source budget.

Ranks combine discovery evidence only. They confer no truth, identity, scope
or source validity; callers must reread through their canonical validator.
"""
from collections import defaultdict


def fuse_ranked_handles(*rankings, limit, rank_constant=60):
    if limit < 1 or rank_constant < 1:
        raise ValueError("invalid_candidate_fusion_budget")
    scores = defaultdict(float)
    first_seen = {}
    handles = {}
    for ranking in rankings:
        seen = set()
        for handle in ranking[:limit]:
            key = handle[:2]  # source identity includes its authorized scope
            if key in seen:
                continue
            seen.add(key)
            first_seen.setdefault(key, len(first_seen))
            handles.setdefault(key, handle)
            scores[key] += 1 / (rank_constant + len(seen))
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return tuple(handles[key] for key in ordered[:limit]), len(ordered)
