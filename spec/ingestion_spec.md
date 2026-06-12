# Ingestion Module Spec

> **同步声明**：本文档严格反向推导自 `src/ingestion.py`、`src/openai_embedding.py` 与 `src/text_splitter.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`ingestion.py` 负责将解析后的报告 JSON 构建为两种检索索引：
- **BM25 稀疏索引**（`BM25Ingestor`）：基于 `rank_bm25`，按空白符分词
- **ChromaDB 向量索引**（`VectorDBIngestor`）：基于 OpenAI Embedding API + `langchain_chroma.Chroma`

两个 Ingestor 均按**单份报告**独立处理，输出以报告 `metainfo.sha1` 为准。

---

## 2. 依赖

```
python >= 3.10
rank_bm25
langchain-chroma
langchain-core
langchain-openai
openai
numpy
tenacity
tqdm
pydantic-settings
```

**环境变量**：

| 变量名 | 用途 | 说明 |
|--------|------|------|
| `OPENAI_API_KEY` | VectorDBIngestor 调用 OpenAI Embedding | 通过 `src.config.settings` 读取 |
| `OPENAI_API_BASE` | OpenAI 自定义 Base URL（可选） | 通过 `src.config.settings` 读取 |
| `EMBEDDING_MODEL` | 默认 Embedding 模型 | 通过 `src.config.settings` 读取，默认 `text-embedding-3-large` |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化目录 | 通过 `src.config.settings` 读取，默认 `data/stock_data/databases/vector_dbs` |
| `BM25_OUTPUT_DIR` | BM25 索引输出目录 | 通过 `src.config.settings` 读取，默认 `data/stock_data/databases/bm25_index` |
| `REPORTS_INPUT_DIR` | 报告 JSON 输入目录 | 通过 `src.config.settings` 读取，默认 `data/stock_data/databases/chunked_reports` |

> 注意：`VectorDBIngestor` 的 embedding 实现已改为 OpenAI，不再使用 DashScope。

---

## 3. 输入数据格式

两个 Ingestor 消费的输入为同一套**分块后的报告 JSON**，路径：`all_reports_dir/*.json`

### 3.1 报告 JSON 结构（Ingestion 消费侧）

```json
{
  "metainfo": {
    "sha1": "abc123...",
    "sha1_name": "abc123...",
    "company_code": "001",
    "file_name": "示例科技_2024年报.pdf",
    "pages_amount": 100
  },
  "content": {
    "chunks": [
      {"id": 0, "type": "content", "page": 1, "text": "...", "length_tokens": 50},
      {"id": 1, "type": "serialized_table", "page": 1, "text": "...", "length_tokens": 80}
    ]
  }
}
```

**关键字段约束**：
- `metainfo.sha1`：必须存在且非空，用于生成输出文件名
- `content.chunks`：必须存在，元素为 dict，至少包含 `text` 字段
- `VectorDBIngestor._build_docs` 额外消费：`chunk.id`, `chunk.type`, `chunk.page`, `chunk.length_tokens`, `metainfo.sha1_name`, `metainfo.file_name`, `metainfo.pages_amount`

---

## 4. BM25Ingestor

### 4.1 接口

```python
class BM25Ingestor:
    def __init__(self)
    
    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi

    @staticmethod
    def _build_chunk_metadata(chunk: dict, metainfo: dict) -> dict
    
    def process_reports(self, all_reports_dir: Path | None = None, output_dir: Path | None = None, index_name: str = "default")

    @staticmethod
    def load_bm25_index(index_path: Path) -> tuple[BM25Okapi, list[dict], list[str]]

    def search(self, query: str, index_name: str = "default", output_dir: Path | None = None) -> tuple[np.ndarray, list[dict], list[str]]
```

### 4.2 create_bm25_index

| 项 | 说明 |
|----|------|
| 输入 | `chunks: List[str]` — 纯文本字符串列表 |
| 分词方式 | `jieba.cut(chunk)` — 支持中文分词 |
| 输出 | `BM25Okapi` 索引对象 |

### 4.3 process_reports

批量处理目录下所有 `*.json` 报告：

1. `all_reports_dir` 为空时，自动取 `settings.reports_input_dir`（`.env` 中的 `REPORTS_INPUT_DIR`，默认 `data/stock_data/databases/chunked_reports`）
2. `output_dir` 为空时，自动取 `settings.bm25_output_dir`（`.env` 中的 `BM25_OUTPUT_DIR`，默认 `data/stock_data/databases/bm25_index`）
3. `output_dir.mkdir(parents=True, exist_ok=True)` — 自动创建输出目录
4. 遍历 `all_reports_dir.glob("*.json")`，带 tqdm 进度条
5. 逐报告读取 JSON，提取 `metainfo` 与 `content.chunks`
6. 对每个 chunk：
   - 收集 `text` 字段
   - 调用 `_build_chunk_metadata(chunk, metainfo)` 生成元数据（含 `chunk_id`, `chunk_type`, `page`, `length_tokens`, `sha1`, `sha1_name`, `company_code`, `file_name`, `pages_amount`）
7. 调用 `create_bm25_index` 构建合并索引
8. 将索引、元数据列表与原始文本列表一起保存为 `pickle`，文件名为 `{index_name}.pkl`，格式为 `{"index": BM25Okapi, "metadatas": List[dict], "texts": List[str]}`
9. 打印处理数量

**输出文件**：
```
output_dir/
└── {index_name}.pkl
```

### 4.4 _build_chunk_metadata

| 项 | 说明 |
|----|------|
| 输入 | `chunk: dict` — 单个 chunk 对象；`metainfo: dict` — 报告元信息 |
| 输出 | `dict` — 包含以下字段的元数据字典 |

**输出字段**：

| 字段 | 来源 | 默认值 |
|------|------|--------|
| `chunk_id` | `chunk["id"]` | `0` |
| `chunk_type` | `chunk["type"]` | `""` |
| `page` | `chunk["page"]` | `0` |
| `length_tokens` | `chunk["length_tokens"]` | `0` |
| `sha1` | `metainfo["sha1"]` | `""` |
| `sha1_name` | `metainfo["sha1_name"]` | `""` |
| `company_code` | `metainfo["company_code"]` | `""` |
| `file_name` | `metainfo["file_name"]` | `""` |
| `pages_amount` | `metainfo["pages_amount"]` | `0` |

### 4.5 load_bm25_index

| 项 | 说明 |
|----|------|
| 输入 | `index_path: Path` — `.pkl` 文件路径 |
| 输出 | `tuple[BM25Okapi, list[dict], list[str]]` — (索引对象, 元数据列表, 原始文本列表) |
| 兼容性 | 兼容旧格式：纯 `BM25Okapi` 对象时，元数据和文本均返回 `[]` |

### 4.6 search

| 项 | 说明 |
|----|------|
| 输入 | `query: str` — 查询文本；`index_name: str` — 索引标识；`output_dir: Path \| None` — 索引目录 |
| 分词 | `jieba.cut(query)` — 与构建索引时保持一致 |
| 输出 | `tuple[np.ndarray, list[dict], list[str]]` — (scores 数组, 元数据列表, 原始文本列表) |
| 异常 | `FileNotFoundError` — 索引文件不存在 |

**流程**：
1. `output_dir` 为空时回退到 `settings.bm25_output_dir`
2. 构造路径 `{output_dir}/{index_name}.pkl`
3. 调用 `load_bm25_index` 加载索引与元数据
4. 对 query 用 `jieba.cut` 分词后调用 `bm25_index.get_scores(tokenized_query)`
5. 返回 `(scores, metadatas)`，两者长度一致

---

## 5. VectorDBIngestor

### 5.1 接口

```python
class VectorDBIngestor:
    def __init__(self, embedder)
    
    def _get_embeddings(self, text: Union[str, List[str]]) -> List[List[float]]
    
    def _build_docs(self, report: dict, index_name: str = "default") -> List[Document]
    
    def process_reports(self, all_reports_dir: Path | None = None, output_dir: Path | None = None, index_name: str | None = None)
```

### 5.2 __init__

```python
def __init__(self, embedder)
```

- `embedder` 为必填参数，由调用方注入（通常为 `OpenAIEmbedder` 实例）

### 5.3 _get_embeddings

**签名**：
```python
@retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
def _get_embeddings(self, text)
```

**重试策略**：固定间隔 20 秒，最多 2 次尝试（含首次）。

**输入处理流程**：

| 步骤 | 行为 | 异常 |
|------|------|------|
| 1 | 若输入为 `str` 且 `strip()` 后为空 | `ValueError("Input text cannot be an empty string.")` |
| 2 | 统一转为一维字符串列表 `text_chunks` | — |
| 3 | 类型检查：所有元素必须为 `str` | `ValueError("所有待嵌入文本必须为字符串类型！...")` |
| 4 | 过滤 `strip()` 后为空的字符串 | 若过滤后为空列表，抛 `ValueError("所有待嵌入文本均为空字符串！")` |
| 5 | 调用 `self.embedder.get_embeddings(text_chunks)` | — |

### 5.4 _build_docs

针对**单份报告**的处理流程：

1. 提取 `report['content']['chunks']` 中的每个 chunk
2. 过滤空 `text`
3. 截断到 `max_len = 2048` 字符：`t[:max_len]`
4. 构建 `langchain_core.documents.Document` 列表，metadata 包含：
   - `chunk_id`, `chunk_type`, `page`, `length_tokens`
   - `sha1`, `sha1_name`, `company_code`, `file_name`, `pages_amount`
   - `index_name`（由 `process_reports` 传入，默认 `"default"`）

> ⚠️ 注意：截断按**字符数**而非 token 数，与 `text_splitter.py` 中的 token 统计逻辑不一致。

### 5.5 process_reports

批量处理目录下所有 `*.json` 报告：

1. `all_reports_dir` 为空时，自动取 `settings.reports_input_dir`（`.env` 中的 `REPORTS_INPUT_DIR`，默认 `data/stock_data/databases/chunked_reports`）
2. `output_dir` 为空时，自动取 `settings.chroma_persist_dir`（`.env` 中的 `CHROMA_PERSIST_DIR`，默认 `data/stock_data/databases/vector_dbs`）
3. `index_name` 为空时，默认值为 `"default"`，会传递给 `_build_docs` 写入 metadata
4. `output_dir.mkdir(parents=True, exist_ok=True)`
5. 遍历 `all_reports_dir.glob("*.json")`，带 tqdm 进度条
6. 每份报告：
   - 读取 JSON
   - 调用 `_build_docs(report_data, index_name=index_name)` 构建 Document 列表
   - 若为空则跳过
   - 首次创建 `Chroma.from_documents(..., persist_directory=output_dir, collection_name=index_name)`
   - 后续调用 `vectorstore.add_documents(docs)`
7. 打印处理数量

**输出文件**：
ChromaDB 持久化目录结构（由 `persist_directory` 指定）。

---

## 6. OpenAIEmbedder（内部依赖）

定义于 `src/openai_embedding.py`：

```python
class OpenAIEmbedder:
    def __init__(self, model: str = None)
    def get_embeddings(self, texts: List[str]) -> List[List[float]]
```

- `model` 默认从 `settings.embedding_model` 读取，回退 `"text-embedding-3-large"`
- 底层使用 `langchain_openai.OpenAIEmbeddings`
- 内部会先过滤空字符串，若全部为空返回 `[]`

---

## 7. 异常与边界行为

| 场景 | 行为 |
|------|------|
| BM25: `all_reports_dir` 为空 | `glob` 返回空列表，输出 `"Processed 0 reports"`，不会创建 `.pkl` 文件 |
| BM25: `all_reports_dir` 为空 | 自动回退到 `settings.reports_input_dir` |
| BM25: `output_dir` 为空 | 自动回退到 `settings.bm25_output_dir` |
| BM25: 所有报告 chunks 为空 | 不会创建 `.pkl` 文件 |
| BM25: JSON 缺少 `content.chunks` | 代码未做防御，会抛 `KeyError` |
| BM25: JSON 缺少 `metainfo` | `_build_chunk_metadata` 回退为空 dict，字段取默认值 |
| Vector: `OPENAI_API_KEY` 未设置 | `settings.openai_api_key` 为空，API 调用时由 OpenAI SDK 抛错 |
| Vector: 单条文本为空字符串 | `ValueError`（过滤前拦截） |
| Vector: 所有文本过滤后为空 | `ValueError` |
| Vector: JSON 缺少 `metainfo.sha1` | 代码未做防御，`_build_docs` 中 `sha1` 为空字符串 |
| Vector: `content.chunks` 结构异常 | 会抛 `KeyError` 或 `TypeError`（未防御） |
| Vector: `all_reports_dir` 为空 | 自动回退到 `settings.reports_input_dir` |
| Vector: `output_dir` 为空 | 自动回退到 `settings.chroma_persist_dir` |
| Vector: `index_name` 为空 | 自动回退到 `"default"` |

---

## 8. 当前实现缺陷（需后续优化）

1. **硬编码参数过多**：
   - `max_len = 2048`
   - 重试间隔 20 秒、最多 2 次
   - 这些均应改为可配置参数

2. **截断逻辑不一致**：`_build_docs` 按字符数截断（2048 chars），而 `text_splitter.py` 按 token 数分块

3. ~~**BM25 分词缺陷**：`chunk.split()` 对中文效果极差（按字分词），应改用 jieba 或其他中文分词器~~（已修复：v1.3 引入 jieba 分词）

4. **VectorDBIngestor 残留 DashScope 痕迹**：`ingestion.py` 文件头仍导入了 `dotenv`, `openai`（未实际使用），`BM25Ingestor` 代码整洁但 `VectorDBIngestor` 历史包袱较多

---

## 9. 使用示例

### 9.1 BM25 索引构建

```python
from pathlib import Path
from src.ingestion import BM25Ingestor

ingestor = BM25Ingestor()
ingestor.process_reports(
    all_reports_dir=Path("data/reports_chunked"),
    output_dir=Path("data/bm25_index")
)
```

### 9.2 向量索引构建

```python
from pathlib import Path
from src.ingestion import VectorDBIngestor

ingestor = VectorDBIngestor()

# 使用 .env 中配置的 CHROMA_PERSIST_DIR 作为输出目录，index_name 默认 "default"
ingestor.process_reports(all_reports_dir=Path("data/reports_chunked"))

# 显式指定输出目录与索引名称
ingestor.process_reports(
    all_reports_dir=Path("data/reports_chunked"),
    output_dir=Path("data/vector_index"),
    index_name="annual_reports"
)
```

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始实现，支持 BM25 + DashScope/FAISS 向量索引构建 |
| v1.1 | 2026-06-11 | VectorDBIngestor 迁移至 OpenAI Embedding + ChromaDB；移除 FAISS 与 DashScope 依赖 |
| v1.3 | 2026-06-12 | chunk metadata 使用 `company_code` 替代 `company_name`，值取自报告 JSON 的 `metainfo` |
