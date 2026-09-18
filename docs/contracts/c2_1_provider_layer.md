# C2-1 行为契约：LLM 熔断、文本分词、嵌入供应商

本文件是 **C2-1 洁净室重写**的验收依据。实现可以换写法，**本文件写下的外部可观察行为不能换**。

---

## 0. 判据与口径

### 0.1 契约事实 vs 表达层（沿用来源记录 §5.9「路径 A」）

| 层 | 内容 | 归属 |
| --- | --- | --- |
| **契约事实** | 类名、公开方法名与签名、参数名与默认值、返回结构里的键名、协议常量、环境变量名、错误类型 | 必须逐字保留，属接口信息 |
| **表达层** | docstring、注释、内部组织与文件切分、私有命名风格、实现步骤与算法写法 | 必须 Shinku 独立撰写 |

**推论（本批的一条特殊规定）：** 对照物里的**私有名**属表达层，新实现**不得复用**。
本批登记了 34 个旧私有名（见 `_kit/batches/c2_1.py` 的 `LEGACY_NAMES`），
新实现在每一处都换名——例如旧实现的 `_dimension` / `_lock` / `_cache` / `_get` / `_put` /
`_model` / `_load_model` / `_open_until` / `_consecutive_failures` 全部改掉。

### 0.2 字符串字面量不在「散文筛子」口径内

`STOPWORDS` 的词表、`TOKEN_RE` 的正则、`HF_HUB_OFFLINE` 这类环境变量名，
都是**契约事实**（外部可观察 + 线路协议），必须逐字保留。
散文筛子只覆盖 docstring 与注释（SKILL 规矩十二）。

### 0.3 本批的四个模块与来源

| 新落点 | 来源对照物 | 对照物字节 | 行为基准 |
| --- | --- | ---: | --- |
| `llm/circuit_breaker.py` | Akane `companion_v01/llm_circuit_breaker.py` | 3,573 | 同一份（与 Shinku 旧侧逐字节相同） |
| `text/tokenizer.py` | `code_shared/code_shared/text_tokenizer.py` | 1,144 | Shinku 旧侧 `companion_v01/text_utils.py` 的 `tokenize` |
| `providers/embedding.py` | `code_shared/code_shared/embedding_provider.py` | 5,384 | Shinku 旧侧 `companion_v01/embedding_provider.py` |
| `providers/huggingface.py` | `code_shared/code_shared/huggingface_provider.py` | 4,305 | Shinku 旧侧 `companion_v01/huggingface_provider.py` |

**对照物判定（SKILL 规矩六）：** `embedding_provider.py` 在 Akane 侧是 628 B / 20 行的
转发壳、`huggingface_provider.py` 是 279 B / 10 行的转发壳，都不能当上游实现；
`text_tokenizer.py` 在 Akane 侧不存在。三者都改用 `code_shared` 的真实现。

**为什么 `text/tokenizer.py` 在本批：** 它是 `embedding.py` 的**传递依赖**——
`embedding.py` 用 `tokenize()` 做 hashed embedding 的分词。
传递依赖必须随行（C1-3 的 `capability_safety.py` 同理）。
只迁 `TOKEN_RE`／`STOPWORDS`／`normalize_text`／`tokenize` 四样；
`text_utils.py` 其余 500 行（时间解析、话题抽取、聊天渲染）属 C5，**不在本批**。

---

## 1. `llm/circuit_breaker.py`

### 1.1 公开面

```python
class LlmCircuitBreaker:
    def __init__(self, *, failure_threshold: int = 3,
                 base_cooldown_seconds: float = 60.0,
                 max_cooldown_seconds: float = 600.0) -> None
    def record_success(self) -> None
    def record_failure(self) -> None
    def is_open(self) -> bool
    def remaining_seconds(self) -> float
    def snapshot(self) -> dict

def get_llm_circuit_breaker() -> LlmCircuitBreaker
```

实例上另有三个**公开属性**（构造时归一化后存放，外部可读）：
`failure_threshold`、`base_cooldown_seconds`、`max_cooldown_seconds`。

### 1.2 行为规则

| 规则 | 内容 |
| --- | --- |
| 归一化 | `failure_threshold = max(1, int(v))`；`base_cooldown_seconds = max(1.0, float(v))`；`max_cooldown_seconds = max(base_cooldown_seconds, float(v))`。**边界是闭区间**：`failure_threshold=0` → 1；`base=0` → 1.0；`max < base` → `max` 被抬到 `base` |
| 初始态 | 连续失败 0、未熔断、熔断次数 0、两个时间戳均 0.0 |
| `record_failure` | 时间戳记为当前时刻；连续失败 +1；**未达阈值直接返回**（不改熔断状态）；达到阈值则熔断次数 +1，冷却 `min(max_cooldown, base * 2 ** (开闸次数 - 1))`，静默期到 `now + 冷却` |
| `record_success` | 连续失败归 0、静默期归 0、熔断次数归 0、成功时间戳记为当前时刻 |
| `is_open` | `now < 静默期截止时刻` |
| `remaining_seconds` | `max(0.0, 静默期截止时刻 - now)` |
| 退避序列 | 阈值 3、base 60、max 600 ⇒ 开闸后冷却依次 60 / 120 / 240 / 480 / 600 / 600 …（封顶） |
| 线程安全 | 全部读写都在同一把可重入锁内 |

### 1.3 `snapshot()` 的返回结构（键名逐字保留）

```python
{
    "open": bool,                 # now < 静默期截止
    "consecutiveFailures": int,
    "openCount": int,
    "remainingSeconds": float,    # round(max(0.0, 截止 - now), 1)
    "lastFailureAt": float,       # round(时间戳, 3)
    "lastSuccessAt": float,       # round(时间戳, 3)
}
```

### 1.4 失败模式

无。本模块不抛异常、不做 IO、不读环境变量。

---

## 2. `text/tokenizer.py`

### 2.1 公开面

```python
TOKEN_RE: re.Pattern            # re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
STOPWORDS: set[str]             # 23 个元素，逐字保留
def normalize_text(text: str) -> str
def tokenize(text: str) -> list[str]
__all__ = ["STOPWORDS", "TOKEN_RE", "normalize_text", "tokenize"]
```

`STOPWORDS` 的 23 个词（逐字保留，顺序不作要求）：
的 / 了 / 呢 / 啊 / 呀 / 吗 / 吧 / 哦 / 喵 / 我 / 你 / 他 / 她 / 它 /
我们 / 你们 / 他们 / 然后 / 就是 / 这个 / 那个 / 现在 / 一下

### 2.2 行为规则

| 步骤 | 规则 |
| --- | --- |
| 归一化 | `re.sub(r"\s+", " ", str(text or "").strip())`。`None` → `""`；纯空白 → `""` |
| 空输入 | 归一化后为空 ⇒ 返回 `[]`（不抛错） |
| 匹配 | 在**小写化**后的归一化文本上跑 `TOKEN_RE.findall`——即 `A-Z` 也被折成小写 |
| 汉字串 | 整串**原样保留**；长度 > 3 时**再追加所有长度 2 与 3 的连续子串**，`size` 从 2 到 3、起点从左到右 |
| 其他 | 数字/字母/下划线串原样保留（已小写） |
| 过滤 | 丢掉空串与命中 `STOPWORDS` 的项；**不过滤重复**（同词出现几次就留几个） |

**子串的边界（易错）：** 长度 4 的汉字串产生 `2-gram` 3 个 + `3-gram` 2 个；
长度恰好 3 的汉字串**只保留整串**，不产生子串。
`range(0, max(0, len - size + 1))` 的右端是开区间——长度 5 的串，`2-gram` 产生 4 个。

### 2.3 分层

`normalize_text` 与 `TOKEN_RE` 对 `tokenize` 是内部依赖，但对下游是公开面
（`__all__` 里四个都导出）。

---

## 3. `providers/embedding.py`

### 3.1 公开面

```python
class BaseEmbeddingProvider(ABC):
    provider_name = "base"
    version = "v1"
    def __init__(self, *, dimension: int) -> None
    @property name -> str
    @property dimension -> int
    @property legacy_collection_name -> str | None
    def collection_key(self) -> str
    @abstractmethod def embed_text(self, text: str) -> list[float]
    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]

class HashedEmbeddingProvider(BaseEmbeddingProvider):
    provider_name = "hashed"
    version = "v1"
    def __init__(self, *, dimension: int = 128,
                 legacy_collection_name: str | None = "shinku_memory_v01") -> None

class CachedEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, inner: BaseEmbeddingProvider, *, max_entries: int = 2048) -> None
    # 公开属性 inner / max_entries

__all__ = ["BaseEmbeddingProvider", "CachedEmbeddingProvider", "HashedEmbeddingProvider"]
```

### 3.2 行为规则

**`BaseEmbeddingProvider`**

| 成员 | 规则 |
| --- | --- |
| `__init__` | `dimension = max(1, int(v))` |
| `name` | `str(provider_name or "base").strip() or "base"` |
| `dimension` | 归一化后的值 |
| `legacy_collection_name` | 基类返回 `None` |
| `collection_key` | `f"{name}_{version}_{dimension}".lower()`，把 `[^a-z0-9_-]+` 连续段替换成单个 `_`，去掉首尾 `_`；结果为 `""` 时返回 `"embedding"` |
| `embed_text` | 抽象方法，直接调用抛 `NotImplementedError` |
| `embed_texts` | 对每个元素逐个调 `embed_text`（**不**去重、**不**过滤空串） |

**`HashedEmbeddingProvider`**

| 成员 | 规则 |
| --- | --- |
| `__init__` | `legacy_collection_name = str(v or "").strip() or None` |
| `legacy_collection_name` | **仅当** `dimension == 128 且 version == "v1"` 时返回构造时存的值，否则 `None` |
| `embed_text` | 见下 |

`embed_text` 的算法（逐字保留顺序与常数）：

1. `vector = [0.0] * dimension`
2. 对 `tokenize(text)` 的每个 token：
   - `digest = sha256(token.encode("utf-8")).digest()`
   - `index = int.from_bytes(digest[:4], "big") % dimension`
   - `sign = +1.0 if digest[4] % 2 == 0 else -1.0`
   - `vector[index] += sign * (1.0 + len(token) / 10.0)`
3. `norm = sqrt(sum(v * v for v in vector))`
4. `norm` 非 0 ⇒ 逐项除以 `norm`；`norm == 0` ⇒ **原样返回全 0 向量**

**注意：`version` 在 `CachedEmbeddingProvider` 上是 property**（转发 `inner`），
基类上是类属性——这是一个真实的接口不对称，必须保留。

**`CachedEmbeddingProvider`**

| 成员 | 规则 |
| --- | --- |
| `name` / `version` / `legacy_collection_name` | 转发 `inner` 的同名属性 |
| `collection_key` | 转发 `inner.collection_key()` |
| `max_entries` | `max(0, int(v))` |
| `embed_text` | `str(text or "")` 后查缓存；`max_entries <= 0` 时**直通** `inner`（不缓存）。命中返回**副本** `list(cached)`；未命中调 `inner.embed_text`，把结果转成 `tuple(float(...))` 存缓存，再返回副本 |
| `embed_texts` | 先全部 `str(x or "")`；`max_entries <= 0` 时直通 `inner.embed_texts`。否则：**按首次出现顺序**收集未命中的**去重**值，一次性调 `inner.embed_texts(list(missing))`，把结果按原顺序 zip 回填所有同值下标。返回值形如 `[vector or [] for vector in result]` |
| 缓存淘汰 | LRU：命中与写入都 `move_to_end`；写入后若长度超 `max_entries`，从头部 `popitem(last=False)` 弹出直到不超 |
| 线程安全 | 缓存读写与淘汰在同一把可重入锁内 |

**`embed_texts` 的一个边界（易错）：** 返回值里的 `[]` 只可能来自
`inner.embed_texts` 给出的空向量（`vector or []`），
**不是**"未命中"的哨兵——所有位置都会被回填。

### 3.3 失败模式

- `BaseEmbeddingProvider` 是 `abc.ABC`：直接实例化抛 `TypeError`；
  子类未实现 `embed_text` 也在实例化时抛 `TypeError`。
- 负/零 `dimension` 被抬到 1，不抛错。

---

## 4. `providers/huggingface.py`

### 4.1 公开面

```python
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-m3"

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    provider_name = "huggingface"
    def __init__(self, *, model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
                 device: str | None = None, local_files_only: bool = False,
                 cache_folder: str | None = None, hf_endpoint: str | None = None,
                 normalize_embeddings: bool = True) -> None
    @property version -> str          # 返回 model_name
    def embed_text(self, text: str) -> list[float]
    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]

__all__ = ["DEFAULT_HUGGINGFACE_EMBEDDING_MODEL", "HuggingFaceEmbeddingProvider"]
```

实例上的公开属性：`model_name`、`device`、`local_files_only`、`cache_folder`、
`hf_endpoint`、`normalize_embeddings`。

### 4.2 归一化规则

| 参数 | 规则 |
| --- | --- |
| `model_name` | `str(v or DEFAULT).strip() or DEFAULT` |
| `device` | `str(v or "").strip() or None` |
| `local_files_only` | `bool(v)` |
| `cache_folder` | `str(v or "").strip() or None` |
| `hf_endpoint` | `str(v or "").strip().rstrip("/") or None`（**先 strip 再去尾斜杠**） |
| `normalize_embeddings` | `bool(v)` |

### 4.3 构造期行为

1. 载入模型（见 §4.5），得到 `encoder`
2. `dimension = encoder.get_sentence_embedding_dimension()`
3. **若该值为假**（`0`／`None`），退化为 `encoder.encode(["探针"], normalize_embeddings=…,
   convert_to_numpy=True, show_progress_bar=False)`，取 `len(probe[0])` 作为维度——
   **探针文本就是 `"探针"` 两个字，逐字保留**
4. 用该维度调 `super().__init__(dimension=int(dimension))`

### 4.4 `embed_text` / `embed_texts`

- `embed_text(text)` 等价于 `embed_texts([text])[0]`（**空列表会 `IndexError`**，不额外兜底）
- `embed_texts`：先 `[str(x or "") for x in texts]`；空列表 ⇒ 返回 `[]`（**不调 encoder**）；
  否则调 `encoder.encode(values, normalize_embeddings=…, convert_to_numpy=True,
  show_progress_bar=False)`，返回 `[list(map(float, v)) for v in vectors]`

### 4.5 载入模型与「环境变量临时改写」

载入时的行为，顺序与副作用都要保留：

1. 在 `warnings.catch_warnings()` 里**忽略**匹配
   `r"The pynvml package is deprecated\..*"` 的 `FutureWarning`
2. 进入临时环境上下文（见下）
3. `from sentence_transformers import SentenceTransformer` ——
   失败（`ImportError`）则抛 `RuntimeError`，消息为：
   `"sentence-transformers is not installed; install requirements-ml.txt to enable HuggingFace embeddings."`
   并把原异常挂成 `__cause__`
4. 构造 kwargs：总是带 `device`；`cache_folder` 非空才带
5. 用 `inspect.signature(SentenceTransformer)` 探签名：
   取不到签名（`TypeError`／`ValueError`）**或**签名里有 `local_files_only` 参数 ⇒ 带上 `local_files_only`
6. 返回 `SentenceTransformer(model_name, **kwargs)`

**临时环境上下文（`_temporary_hf_load_env` 的同义行为）：**

| 条件 | 写入的环境变量 |
| --- | --- |
| `local_files_only` 为真 | `HF_HUB_OFFLINE="1"` |
| `hf_endpoint` 去空去尾斜杠后非空 | `HF_ENDPOINT=<该值>` |

- 两个都不满足 ⇒ **完全不碰 `os.environ`**（连读取都不做）
- 否则：先记下这两个键的原值（不存在的记 `None`），写入新值，`yield`；
  **`finally` 里无条件恢复**——原值为 `None` 的键要 `pop` 掉（不是设成字符串 `"None"`）

### 4.6 失败模式

| 触发 | 结果 |
| --- | --- |
| `sentence-transformers` 未安装 | `RuntimeError`（消息见上），`__cause__` 是 `ImportError` |
| `embed_text` 在空字符串上 | 正常返回向量（`""` 会被 encoder 处理） |
| `get_sentence_embedding_dimension()` 返回 0/None | 走探针路径 |

---

## 5. 本批的验收口径

1. **散文重叠 = 0**：四个新文件的 docstring+注释与对照物、与 Shinku 旧实现，均无 ≥12 字符逐字片段；
2. **旧私有名命中 = 0**：34 个登记名（`_kit` 的 `namegap` 已核覆盖完整）；
3. **契约字段表达式逐字对照实现**：§1.3 的 6 个键、§2.1 的 23 个停用词、
   §3.2 的哈希常数、§4.3 的探针文本、§4.5 的 `RuntimeError` 消息与两个环境变量名；
4. **行为等价性对拍 = 0 差异**：与 Shinku 旧实现逐项比返回值；
5. **变异测试漏 0 个**；
6. **干净环境全量回归通过**。

**对拍的三处特殊处理（本批新增，写进批次记录）：**

| 模块 | 为什么必须特殊处理 |
| --- | --- |
| `circuit_breaker` | 全读 `time.time()`。60s 冷却没法真的等 ⇒ 注入可控时钟，两侧共用同一时钟 |
| `huggingface` | `__init__` 会真的 import 并 encode ⇒ 往 `sys.modules` 塞假 `sentence_transformers`，两侧共用同一个假实现 |
| `tokenizer` / `embedding` | 纯函数，直接喂语料即可 |
