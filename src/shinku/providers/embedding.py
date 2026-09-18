"""检索侧把文本变成向量的三种角色。

一层只做一件事：

``BaseEmbeddingProvider``
    定公开面——名字、维度、派生出的集合键，以及必须由子类填的 ``embed_text``。
    它把「集合键」这条规则收在一处：名字与版本与维度拼起来、折成小写、
    非法字符压成一个下划线，于是「同一个 provider 换维度」必然落到不同集合上。
``HashedEmbeddingProvider``
    不依赖任何模型与网络的近似向量：把词哈希进固定宽度的槽位再定长归一化。
    它只用来做粗排——同义词不共享槽位是刻意接受的代价，换来的是零依赖与可复现。
``CachedEmbeddingProvider``
    给任意 provider 套一层有界 LRU。同一段文本在一次会话里会被反复取向量，
    缓存把第二次以后的成本压成一次字典查找。

集合键与旧集合名是两回事：前者由本层算出来，后者是存量数据的位置，
只在「维度恰好 128 且版本恰好 v1」时才交出去——换过维度或版本就不能再指向旧集合。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterable

from ..text.tokenizer import tokenize

#: 集合键里不允许出现的字符。连续一段压成一个下划线，不逐个字符替换。
_NON_KEY_CHARS = re.compile(r"[^a-z0-9_-]+")

#: 一个空集合键的兜底值，保证键永远非空。
_FALLBACK_COLLECTION_KEY = "embedding"


class BaseEmbeddingProvider(ABC):
    """向量化器的共同脸面。

    子类只需要填 ``embed_text``；批量入口默认就是逐个调它，不去重也不过滤空串——
    「同一段文本算几次」是调用方的选择，不该由基类暗地里合并。
    """

    provider_name = "base"
    version = "v1"

    def __init__(self, *, dimension: int) -> None:
        self._width = max(1, int(dimension))

    @property
    def name(self) -> str:
        return str(self.provider_name or "base").strip() or "base"

    @property
    def dimension(self) -> int:
        return self._width

    @property
    def legacy_collection_name(self) -> str | None:
        """存量集合的位置。基类没有历史包袱，一律 ``None``。"""
        return None

    def collection_key(self) -> str:
        raw = f"{self.name}_{self.version}_{self.dimension}".lower()
        squeezed = _NON_KEY_CHARS.sub("_", raw).strip("_")
        return squeezed or _FALLBACK_COLLECTION_KEY

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class HashedEmbeddingProvider(BaseEmbeddingProvider):
    """哈希词袋向量：无模型、无网络、可复现。

    每个词取 sha256，前 4 字节决定落在哪个槽位，第 5 字节的奇偶决定加还是减，
    权重随词长轻微上浮（长词信息量更大）。最后按 L2 范数归一化，
    于是余弦相似度退化成点积；一个词都没切出来时范数为 0，直接给全零向量。
    """

    provider_name = "hashed"
    version = "v1"

    def __init__(
        self,
        *,
        dimension: int = 128,
        legacy_collection_name: str | None = "shinku_memory_v01",
    ) -> None:
        super().__init__(dimension=dimension)
        self._retired_name = str(legacy_collection_name or "").strip() or None

    @property
    def legacy_collection_name(self) -> str | None:
        if self.dimension == 128 and self.version == "v1":
            return self._retired_name
        return None

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[slot] += sign * (1.0 + len(token) / 10.0)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector


class CachedEmbeddingProvider(BaseEmbeddingProvider):
    """给任意 provider 套一层有界 LRU。

    缓存存的是 ``tuple`` 而不是 ``list``：拿出去的一定是副本，
    调用方 append 自己的向量不会改到缓存里的那一份。
    """

    def __init__(self, inner: BaseEmbeddingProvider, *, max_entries: int = 2048) -> None:
        super().__init__(dimension=inner.dimension)
        self.inner = inner
        self.max_entries = max(0, int(max_entries))
        self._guard = threading.RLock()
        self._stored: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def version(self) -> str:
        return self.inner.version

    @property
    def legacy_collection_name(self) -> str | None:
        return self.inner.legacy_collection_name

    def collection_key(self) -> str:
        return self.inner.collection_key()

    def embed_text(self, text: str) -> list[float]:
        raw = str(text or "")
        if self.max_entries <= 0:
            return self.inner.embed_text(raw)

        remembered = self._recall(raw)
        if remembered is not None:
            return list(remembered)

        frozen = tuple(float(value) for value in self.inner.embed_text(raw))
        self._store(raw, frozen)
        return list(frozen)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = [str(text or "") for text in texts]
        if self.max_entries <= 0:
            return self.inner.embed_texts(values)

        filled: list[list[float] | None] = [None] * len(values)
        pending: dict[str, list[int]] = {}
        for index, value in enumerate(values):
            remembered = self._recall(value)
            if remembered is None:
                pending.setdefault(value, []).append(index)
            else:
                filled[index] = list(remembered)

        if pending:
            computed = self.inner.embed_texts(list(pending))
            for value, vector in zip(pending, computed):
                frozen = tuple(float(item) for item in vector)
                self._store(value, frozen)
                for index in pending[value]:
                    filled[index] = list(frozen)

        return [vector or [] for vector in filled]

    def _recall(self, key: str) -> tuple[float, ...] | None:
        """命中就把这一项挪到队尾（最近用过），没命中返回 ``None``。"""
        with self._guard:
            vector = self._stored.get(key)
            if vector is not None:
                self._stored.move_to_end(key)
            return vector

    def _store(self, key: str, vector: tuple[float, ...]) -> None:
        """写入并压回上限；超出就从队头（最久没用过）开始丢。"""
        with self._guard:
            self._stored[key] = vector
            self._stored.move_to_end(key)
            while len(self._stored) > self.max_entries:
                self._stored.popitem(last=False)


__all__ = ["BaseEmbeddingProvider", "CachedEmbeddingProvider", "HashedEmbeddingProvider"]
