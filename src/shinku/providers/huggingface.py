"""可选的本地句向量后端：把本地句向量模型接成一个 provider。

这一层只干三件事——把参数归一化、把模型载进来、把 encoder 的返回值转成 Python 浮点列表。
真正的计算全在 ``sentence_transformers`` 里，本模块不做任何缓存或批处理策略。

载模型时顺手处理两个环境开关：离线模式与自建镜像端点。它们**只在载入这一瞬间**生效，
退出时无条件把 ``os.environ`` 还原成进门前那样——原值不存在的键要删掉而不是留个字符串，
否则调用方进程里会永久多出一个假地址，后续别的库照着它去联网就全走错了。
两个开关都没给的时候，本模块连读都不读环境，做到零副作用。

``sentence_transformers`` 是可选依赖（它会把 torch 一起拖进来），没装时给出可操作的报错
而不是让 ``ImportError`` 裸露到调用方。
"""

from __future__ import annotations

import inspect
import os
import warnings
from contextlib import contextmanager
from typing import Iterable

from .embedding import BaseEmbeddingProvider

#: 默认模型。多语，中英混排的检索场景够用，体积也比更大的选项温和。
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-m3"

#: 未装可选依赖时的报错文本（要指向具体的安装文件，不然调用方无从下手）。
_INSTALL_HINT = (
    "sentence-transformers is not installed; install requirements-ml.txt "
    "to enable HuggingFace embeddings."
)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """包一层 ``SentenceTransformer``，让它跟哈希词袋共用同一张脸。

    维度不靠声明，而是问模型要；只有模型自己说不出来时才拿一条探针文本实跑一次量长度。
    """

    provider_name = "huggingface"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        device: str | None = None,
        local_files_only: bool = False,
        cache_folder: str | None = None,
        hf_endpoint: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = (
            str(model_name or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL).strip()
            or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
        )
        self.device = str(device or "").strip() or None
        self.local_files_only = bool(local_files_only)
        self.cache_folder = str(cache_folder or "").strip() or None
        self.hf_endpoint = str(hf_endpoint or "").strip().rstrip("/") or None
        self.normalize_embeddings = bool(normalize_embeddings)
        self._encoder = self._open_encoder()

        width = self._encoder.get_sentence_embedding_dimension()
        if not width:
            probe = self._encoder.encode(
                ["探针"],
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            width = int(len(probe[0]))
        super().__init__(dimension=int(width))

    @property
    def version(self) -> str:
        """版本就是模型名——换模型等于换集合，不需要另立字段。"""
        return self.model_name

    def _open_encoder(self):
        """载模型。窗口过滤 → 临时环境 → 导入 → 拼参数 → 构造，顺序不能换。"""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The pynvml package is deprecated\..*",
                category=FutureWarning,
            )
            with _hf_env_override(
                local_files_only=self.local_files_only,
                hf_endpoint=self.hf_endpoint,
            ):
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(_INSTALL_HINT) from exc

                options: dict[str, object] = {"device": self.device}
                if self.cache_folder:
                    options["cache_folder"] = self.cache_folder

                # 旧版 SentenceTransformer 不收 local_files_only。拿不到签名时
                # 按「收」处理：多给一个关键字总比该离线却联网好。
                try:
                    parameters = inspect.signature(SentenceTransformer)
                except (TypeError, ValueError):
                    parameters = None
                if parameters is None or "local_files_only" in parameters.parameters:
                    options["local_files_only"] = self.local_files_only

                return SentenceTransformer(self.model_name, **options)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = [str(text or "") for text in texts]
        if not values:
            return []
        vectors = self._encoder.encode(
            values,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


@contextmanager
def _hf_env_override(*, local_files_only: bool, hf_endpoint: str | None):
    """载入期临时改写 HF 的两个环境开关，退出时无条件还原。

    没有要改的键就直接 yield——一条环境变量都不碰，是最省事也最安全的路径。
    """
    pending: dict[str, str] = {}
    if local_files_only:
        pending["HF_HUB_OFFLINE"] = "1"
    endpoint = str(hf_endpoint or "").strip().rstrip("/")
    if endpoint:
        pending["HF_ENDPOINT"] = endpoint

    if not pending:
        yield
        return

    saved = {key: os.environ.get(key) for key in pending}
    os.environ.update(pending)
    try:
        yield
    finally:
        for key, original in saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


__all__ = ["DEFAULT_HUGGINGFACE_EMBEDDING_MODEL", "HuggingFaceEmbeddingProvider"]
