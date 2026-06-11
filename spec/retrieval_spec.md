# Retrieval Module Spec

> **同步声明**：本文档严格反向推导自 `src/retrieval.py`、`src/reranking.py` 与 `src/post_retrieval_correction.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`retrieval.py` 是文档检索系统的核心模块，提供三种检索策略：**BM25 稀疏检索**、**向量密集检索** 和 **混合检索（向量 + LLM 重排）**。模块面向企业报告场景，按 `company_name` 定位目标文档，在文档分块（chunks）或页面（pages）层级上进行相关性检索。

`post_retrieval_correction.py` 提供检索后文档相关性评分（RetrievalGrader），用于后处理验证。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 多策略检索 | 支持稀疏（BM25）、密集（向量）、混合三种检索模式，适应不同场景 |
| 按公司隔离 | 通过 `company_name` 精确锁定单份报告，避免跨文档污染 |
| 多 Embedding 源 | 支持 OpenAI (`text-embedding-3-large`) 和 DashScope (`text-embedding-v1`) |
| LLM 重排融合 | 混合检索结合向量相似度与 LLM 相关性评分，加权融合后重排序 |
| 灵活返回粒度 | 支持返回 `chunks`（分块）或 `parent_pages`（完整页面） |
| 检索后验证 | 支持对检索结果进行 LLM 二元相关性评分 |

---

## 3. 架构设计

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  BM25Retriever  │     │ VectorRetriever │     │ HybridRetriever │
│  (稀疏检索)      │     │  (向量检索)      │     │ (混合 + 重排)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    BM25Okapi 索引          FAISS 向量库            VectorRetriever
    + 分块 JSON             + 分块 JSON             + LLMReranker
                                                          │
                                                          ▼
                                              ┌─────────────────────┐
                                              │   RetrievalGrader   │
                                              │  (检索后相关性评分)   │
                                              └─────────────────────┘
```

- **BM25Retriever**：基于 `rank_bm25` 构建，适合关键词匹配场景。
- **VectorRetriever**：基于 `faiss` 向量库，支持语义相似度检索。
- **HybridRetriever**：组合 `VectorRetriever` 与 `LLMReranker`，先向量召回、再 LLM 重排，返回加权融合后的 Top-N 结果。
- **RetrievalGrader**：基于 LLM 对单篇文档与问题的相关性进行二元评分（yes/no）。

---

## 4. 依赖清单

```
python >= 3.10
rank_bm25
faiss-cpu 或 faiss-gpu
openai
dashscope (可选，用于国内 Embedding)
numpy
pandas
python-dotenv
```

**内部依赖**：
- `src.reranking.LLMReranker`：用于 HybridRetriever 的重排阶段
- `src.prompts`：LLMReranker 使用的 system prompt 和结构化 schema
- `src.api_requests.APIProcessor`：RetrievalGrader 使用的 LLM 调用封装

---

## 5. 环境变量

| 变量名 | 用途 | 使用方 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI Embedding / LLM 鉴权 | VectorRetriever, LLMReranker, RetrievalGrader |
| `OPENAI_API_BASE` | OpenAI 自定义 Base URL（可选） | LLMReranker |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope 鉴权 | VectorRetriever, LLMReranker |
| `JINA_API_KEY` | Jina Reranker API 鉴权 | JinaReranker（本模块未直接使用） |

---

## 6. 输入数据格式

模块依赖两种持久化数据：

### 6.1 文档 JSON (`documents_dir/*.json`)

```json
{
  "metainfo": {
    "company_name": "示例科技",
    "sha1": "a1b2c3d4...",
    "file_name": "示例科技_2024年报.pdf"
  },
  "content": {
    "chunks": [
      {"page": 1, "text": "..."},
      {"page": 2, "text": "..."}
    ],
    "pages": [
      {"page": 1, "text": "...完整页面文本..."},
      {"page": 2, "text": "...完整页面文本..."}
    ]
  }
}
```

### 6.2 BM25 索引 (`bm25_db_dir/{index_name}.pkl`)

由 `rank_bm25.BM25Okapi` 序列化生成的 `pickle` 文件，文件名由 `BM25Retriever` 构造时传入的 `index_name` 决定（默认为 `default`）。

### 6.3 向量索引 (`vector_db_dir/{sha1}.faiss`)

由 `faiss` 生成的二进制索引文件，文件名与文档 `metainfo.sha1` 对应。

---

## 7. 类接口定义

### 7.1 BM25Retriever

```python
class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path, index_name: str = "default")
    
    def retrieve(
        self,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False
    ) -> List[Dict]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bm25_db_dir` | `Path` | - | BM25 索引存放目录 |
| `documents_dir` | `Path` | - | 文档 JSON 存放目录（`__init__` 时预加载所有 pages 信息） |
| `index_name` | `str` | `"default"` | 索引标识，决定加载的 `.pkl` 文件名 |
| `query` | `str` | - | 查询文本 |
| `top_n` | `int` | `3` | 返回结果数量上限 |
| `return_parent_pages` | `bool` | `False` | `True` 时按 `(sha1, page)` 去重，返回整页内容 |

**返回结构**：
```python
[
  {
    "distance": 12.3456,   # BM25 分数
    "page": 1,
    "text": "..."
  }
]
```

**实现要点**：
- `__init__` 时遍历 `documents_dir/*.json`，建立 `sha1 -> {page_num: page_text}` 映射表 `_pages_by_sha1`
- 内部通过 `BM25Ingestor.search(...)` 加载全局索引并计算 scores
- 不再遍历 `documents_dir` 按 `company_name` 匹配单份报告
- `return_parent_pages=True` 时，根据 `metadata["sha1"]` + `page` 从 `_pages_by_sha1` 查表获取整页内容；查不到时回退到 chunk text
- `return_parent_pages=False` 时直接返回 chunk text |

---

### 7.2 VectorRetriever

```python
class VectorRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        index_name: str = "default"
    )

    def retrieve(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = None,   # 占位，兼容 HybridRetriever
        top_n: int = 3,
        return_parent_pages: bool = False
    ) -> List[Dict]

    @staticmethod
    def get_strings_cosine_similarity(str1, str2) -> float
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector_db_dir` | `Path` | - | ChromaDB 持久化目录 |
| `documents_dir` | `Path` | - | 文档 JSON 存放目录（预加载 pages 映射） |
| `index_name` | `str` | `"default"` | ChromaDB collection 名称 |
| `top_n` | `int` | `3` | 返回最近邻数量 |

**实现要点**：
- `__init__` 时通过 `Chroma(...)` 加载全局向量库，使用 `OpenAIEmbeddings`（与 `VectorDBIngestor` 一致）
- `__init__` 时遍历 `documents_dir/*.json` 建立 `sha1 -> {page_num: page_text}` 映射表 `_pages_by_sha1`
- `retrieve` 调用 `Chroma.similarity_search_with_score(query, k=top_n, ...)`；仅当 `company_name` 非空时传入 `filter={"company_name": company_name}`
- `return_parent_pages=True` 时，根据 `metadata["sha1"]` + `page` 从 `_pages_by_sha1` 查表获取整页内容；查不到时回退到 chunk text
- 已移除对 DashScope/FAISS 的支持

**`retrieve` 返回结构**：
```python
[
  {
    "distance": 0.2345,   # ChromaDB 距离分数（越小越近）
    "page": 1,
    "text": "..."
  }
]
```



---

### 7.3 HybridRetriever

```python
class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path)
    
    def retrieve(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_reranking_sample_size` | `int` | `28` | 首轮向量召回候选数 |
| `documents_batch_size` | `int` | `10` | 每批送入 LLM 重排的文档数 |
| `top_n` | `int` | `6` | 最终返回结果数 |
| `llm_weight` | `float` | `0.7` | LLM 评分权重，`vector_weight = 1 - llm_weight` |

**融合评分公式**：
```
combined_score = llm_weight * relevance_score + vector_weight * distance
```

> 注意：当前 `distance` 为 FAISS L2 距离，数值越小表示越相似。若需与 `relevance_score`（越大越好）同向，后续可考虑对 `distance` 做归一化或取倒数。

**计时输出**：方法内部使用 `time.time()` 打印各阶段耗时：
- `[计时] [HybridRetriever] 开始向量检索 ...`
- `[计时] [HybridRetriever] 向量检索耗时: X.XX 秒`
- `[计时] [HybridRetriever] 开始LLM重排 ...`
- `[计时] [HybridRetriever] LLM重排耗时: X.XX 秒`
- `[计时] [HybridRetriever] 总耗时: X.XX 秒`

**返回结构**：
```python
[
  {
    "distance": 0.2345,        # 原始向量距离
    "page": 1,
    "text": "...",
    "relevance_score": 0.95,   # LLM 给出的相关性评分
    "combined_score": 0.7215   # 加权融合分数
  }
]
```

---

## 8. LLM 重排器（src.reranking）

### 8.1 LLMReranker

```python
class LLMReranker:
    def __init__(self, provider: str = "openai")
    
    def get_rank_for_single_block(self, query: str, retrieved_document: str) -> dict
    
    def get_rank_for_multiple_blocks(self, query: str, retrieved_documents: list) -> dict
    
    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,
        llm_weight: float = 0.7
    ) -> list
```

- 支持单条/批量两种重排模式（由 `documents_batch_size` 决定）。
- 默认 `max_workers=1` 串行调用，避免 DashScope QPS 超限。
- 单条模式使用 `gpt-4o-mini-2024-07-18` / `qwen-turbo` + 结构化输出 (`response_format`)。

**OpenAI 单条模式**：
- 调用 `self.llm.beta.chat.completions.parse`，返回 `{"relevance_score": float, "reasoning": str}`

**DashScope 单条模式**：
- 调用 `dashscope.Generation.call`，返回 `{"relevance_score": 0.0, "reasoning": content}`（未结构化解析）

**OpenAI 批量模式**：
- 调用 `self.llm.beta.chat.completions.parse`，返回 `{"block_rankings": [{"relevance_score": float, "reasoning": str}, ...]}`

**DashScope 批量模式**：
- 调用 `dashscope.Generation.call`，返回 `{"block_rankings": [{"relevance_score": 0.0, "reasoning": content}, ...]}`（未结构化解析）

---

## 9. 检索后评分器（src.post_retrieval_correction）

### 9.1 RetrievalGrader

```python
class RetrievalGrader:
    def __init__(self, provider: str = "openai")
    
    def grade(self, question: str, document: str) -> GradeDocuments
    
    def invoke(self, inputs: dict) -> GradeDocuments
```

**GradeDocuments 结构**：
```python
class GradeDocuments(BaseModel):
    binary_score: str  # "yes" 表示相关，"no" 表示不相关
```

**System Prompt**：
```
你是一个评估检索到的文档与用户问题相关性的评分员。
如果文档包含与问题相关的关键词或语义，则将其评为相关。
给出一个二元评分'yes'或'no'来表示文档是否与问题相关。
```

**边界行为**：
- 调用 `APIProcessor.send_message` 进行结构化输出，返回 `GradeDocuments` 实例
- `invoke` 方法兼容 LangChain / LangGraph Runnable 接口，要求 `inputs` 包含 `"question"` 和 `"document"` 键

### 9.2 默认实例

```python
retrieval_grader = RetrievalGrader()
```

---

## 10. 使用示例

### 10.1 BM25 检索

```python
from pathlib import Path
from src.retrieval import BM25Retriever

retriever = BM25Retriever(
    bm25_db_dir=Path("data/bm25"),
    documents_dir=Path("data/documents"),
    index_name="default"
)

results = retriever.retrieve(
    company_name="示例科技",
    query="营业收入增长原因",
    top_n=5
)
```

### 10.2 向量检索

```python
from src.retrieval import VectorRetriever

retriever = VectorRetriever(
    vector_db_dir=Path("data/vector"),
    documents_dir=Path("data/documents"),
    embedding_provider="dashscope"   # 或 "openai"
)

results = retriever.retrieve(
    company_name="示例科技",
    query="核心竞争力分析",
    top_n=5
)
```

### 10.3 混合检索

```python
from src.retrieval import HybridRetriever

retriever = HybridRetriever(
    vector_db_dir=Path("data/vector"),
    documents_dir=Path("data/documents")
)

results = retriever.retrieve(
    company_name="示例科技",
    query="未来三年战略规划",
    llm_reranking_sample_size=28,
    documents_batch_size=10,
    top_n=6,
    llm_weight=0.7
)
```

### 10.4 检索后评分

```python
from src.post_retrieval_correction import RetrievalGrader

grader = RetrievalGrader(provider="openai")
result = grader.grade(
    question="公司营收增长原因",
    document="2024年公司营业收入同比增长15%，主要得益于..."
)
print(result.binary_score)  # "yes" 或 "no"
```

---

## 11. 性能与限制

| 项目 | 说明 |
|------|------|
| 索引加载 | `VectorRetriever` 在 `__init__` 时全量加载所有 FAISS 索引到内存，文档量大时需关注内存占用 |
| 并发控制 | `LLMReranker` 固定 `max_workers=1`，避免第三方 API QPS 限制 |
| 向量距离 | 当前使用 FAISS L2 距离，未做归一化；与 LLM 分数融合时方向需注意 |
| 异常处理 | 文档缺失、索引缺失、Embedding 为空时均会抛出异常或记录警告日志 |
| DashScope 重排 | DashScope 模式下返回的 relevance_score 固定为 0.0，实际未做结构化解析 |

---

## 12. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始版本，支持 BM25 / Vector / Hybrid 三种检索模式 |
| v1.1 | 2026-06-11 | 新增 RetrievalGrader 描述；补充 HybridRetriever 计时输出、VectorRetriever 返回类型修正、DashScope 重排限制说明 |
