"""Bind a semantic review of every answer segment to actual read evidence."""
from collections.abc import Mapping


def compact(text):
    return "".join(text.split())


def covered(text, fragments, *, answer=None, offset=0, boundary_punctuation=False):
    text = compact(text)
    mask = [False] * len(text)
    search_text = compact(answer) if answer is not None else text
    for fragment in fragments:
        fragment = compact(fragment)
        if not fragment:
            continue
        start = search_text.find(fragment)
        while start >= 0:
            left = max(start, offset) - offset
            right = min(start + len(fragment), offset + len(text)) - offset
            if right > left:
                mask[left:right] = [True] * (right - left)
            start = search_text.find(fragment, start + 1)
    if boundary_punctuation and any(mask):
        first, last = mask.index(True), len(mask) - 1 - mask[::-1].index(True)
        # Only clause delimiters outside the bound span are formatting. Keep
        # question marks, signs, decimal points and all words evidence-bound.
        for i, char in enumerate(text):
            if (i < first or i > last) and char in "，。；,;":
                mask[i] = True
    return bool(text) and all(mask)


def full_partition(answer, value, claims, lookups):
    parts = value.get("answer_parts")
    if parts is None:
        # Older critics can still prove an answer made entirely of bound
        # claims. A limitation field alone cannot bless an unreviewed tail.
        return covered(answer,[c["answer_fragment"] for c in claims])
    if not isinstance(parts,list) or not 1<=len(parts)<=32:
        return False
    if any(not isinstance(p,Mapping) or not isinstance(p.get("text"),str)
           or not p["text"].strip() for p in parts):
        return False
    if compact("".join(p["text"] for p in parts))!=compact(answer):
        return False
    offset = 0
    for part in parts:
        start = offset
        offset += len(compact(part["text"]))
        kind = part.get("kind")
        if not isinstance(kind, str):
            return False
        if kind=="supported":
            indices = part.get("claim_indices")
            if (not isinstance(indices,list) or not indices
                    or any(type(i) is not int or not 0<=i<len(claims) for i in indices)):
                return False
            if not covered(part["text"], [claims[i]["answer_fragment"] for i in indices],
                           answer=answer, offset=start, boundary_punctuation=True):
                return False
        elif kind=="bounded_unknown":
            scope = part.get("scope_fragment")
            # The critic must identify an explicit epistemic boundary in the
            # candidate itself; a completed lookup is not proof of absence.
            if (not isinstance(scope,str) or not compact(scope)
                    or compact(scope) not in compact(part["text"])):
                return False
            index = part.get("lookup_index")
            if (type(index) is not int or not 0<=index<len(lookups)
                    or lookups[index].get("query_completed") is not True
                    or lookups[index].get("requested_scope_supported") is not True):
                return False
        elif kind not in {"non_factual", "uncertainty"}:
            return False
    return True
