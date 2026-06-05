# Retrieval Module Spec

## 1. 概述

`retrieval.py` 是文档检索系统的核心模块，提供三种检索策略：**BM25 稀疏检索**、**向量密集检索** 和 **混合检索（向量 + LLM 重排）**。模块面向企业报告场景，按 `company_name` 定位目标文档，在文档分块（chunks）或页面（pages）层级上进行相关性检索。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 多策略检索 | 支持稀疏（BM25）、密集（向量）、混合三种检索模式，适应不同场景 |
| 按公司隔离 | 通过 `company_name` 精确锁定单份报告，避免跨文档污染 |
| 多 Embedding 源 | 支持 OpenAI (`text-embedding-3-large`) 和 DashScope (`text-embedding-v1`) |
| LLM 重排融合 | 混合检索结合向量相似度与 LLM 相关性评分，加权融合后重排序 |
| 灵活返回粒度 | 支持返回 `chunks`（分块）或 `parent_pages`（完整页面） |

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
```

- **BM25Retriever**：基于 `rank_bm25` 构建，适合关键词匹配场景。
- **VectorRetriever**：基于 `faiss` 向量库，支持语义相似度检索。
- **HybridRetriever**：组合 `VectorRetriever` 与 `LLMReranker`，先向量召回、再 LLM 重排，返回加权融合后的 Top-N 结果。

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

---

## 5. 环境变量

| 变量名 | 用途 | 使用方 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI Embedding / LLM 鉴权 | VectorRetriever, LLMReranker |
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

### 6.2 BM25 索引 (`bm25_db_dir/{sha1}.pkl`)

由 `rank_bm25.BM25Okapi` 序列化生成的 `pickle` 文件，文件名与文档 `metainfo.sha1` 对应。

### 6.3 向量索引 (`vector_db_dir/{sha1}.faiss`)

由 `faiss` 生成的二进制索引文件，文件名与文档 `metainfo.sha1` 对应。

---

## 7. 类接口定义

### 7.1 BM25Retriever

```python
class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path)
    
    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False
    ) -> List[Dict]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `company_name` | `str` | - | 目标公司名称，用于匹配 `metainfo.company_name` |
| `query` | `str` | - | 查询文本 |
| `top_n` | `int` | `3` | 返回结果数量上限 |
| `return_parent_pages` | `bool` | `False` | `True` 时返回完整页面（去重），`False` 返回分块 |

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

---

### 7.2 VectorRetriever

```python
class VectorRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        embedding_provider: str = "dashscope"
    )
    
    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = None,   # 占位，当前未使用
        top_n: int = 3,
        return_parent_pages: bool = False
    ) -> List[Dict]
    
    def retrieve_all(self, company_name: str) -> List[Dict]
    
    @staticmethod
    def get_strings_cosine_similarity(str1, str2) -> float
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_provider` | `str` | `"dashscope"` | `"openai"` 或 `"dashscope"` |
| `top_n` | `int` | `3` | 返回最近邻数量 |

**`retrieve_by_company_name` 返回结构**：
```python
[
  {
    "distance": 0.2345,   # FAISS L2 距离（越小越近）
    "page": 1,
    "text": "..."
  }
]
```

**`retrieve_all` 返回结构**：返回该公司全部页面，`distance` 固定为 `0.5`。

---

### 7.3 HybridRetriever

```python
class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path)
    
    def retrieve_by_company_name(
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

## 8. LLM 重排器（依赖模块）

`HybridRetriever` 内部委托 `src.reranking.LLMReranker` 执行重排。

```python
class LLMReranker:
    def __init__(self, provider: str = "openai")
    
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

---

## 9. 使用示例

### 9.1 BM25 检索

```python
from pathlib import Path
from src.retrieval import BM25Retriever

retriever = BM25Retriever(
    bm25_db_dir=Path("data/bm25"),
    documents_dir=Path("data/documents")
)

results = retriever.retrieve_by_company_name(
    company_name="示例科技",
    query="营业收入增长原因",
    top_n=5
)
```

### 9.2 向量检索

```python
from src.retrieval import VectorRetriever

retriever = VectorRetriever(
    vector_db_dir=Path("data/vector"),
    documents_dir=Path("data/documents"),
    embedding_provider="dashscope"   # 或 "openai"
)

results = retriever.retrieve_by_company_name(
    company_name="示例科技",
    query="核心竞争力分析",
    top_n=5
)
```

### 9.3 混合检索

```python
from src.retrieval import HybridRetriever

retriever = HybridRetriever(
    vector_db_dir=Path("data/vector"),
    documents_dir=Path("data/documents")
)

results = retriever.retrieve_by_company_name(
    company_name="示例科技",
    query="未来三年战略规划",
    llm_reranking_sample_size=28,
    documents_batch_size=10,
    top_n=6,
    llm_weight=0.7
)
```

---

## 10. 性能与限制

| 项目 | 说明 |
|------|------|
| 索引加载 | `VectorRetriever` 在 `__init__` 时全量加载所有 FAISS 索引到内存，文档量大时需关注内存占用 |
| 并发控制 | `LLMReranker` 固定 `max_workers=1`，避免第三方 API QPS 限制 |
| 向量距离 | 当前使用 FAISS L2 距离，未做归一化；与 LLM 分数融合时方向需注意 |
| 异常处理 | 文档缺失、索引缺失、Embedding 为空时均会抛出异常或记录警告日志 |

---

## 11. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始版本，支持 BM25 / Vector / Hybrid 三种检索模式 |
