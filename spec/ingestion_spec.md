# Ingestion Module Spec

> **同步声明**：本文档严格反向推导自 `src/ingestion.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`ingestion.py` 负责将解析后的报告 JSON 构建为两种检索索引：
- **BM25 稀疏索引**（`BM25Ingestor`）：基于 `rank_bm25`，按空白符分词
- **FAISS 向量索引**（`VectorDBIngestor`）：基于 DashScope Embedding API + FAISS `IndexFlatIP`

两个 Ingestor 均按**单份报告**独立处理，输出文件名以报告 `metainfo.sha1` 为准。

---

## 2. 依赖

```
python >= 3.10
rank_bm25
faiss-cpu / faiss-gpu
numpy
dashscope
openai        # 已导入但当前未实际使用
python-dotenv # 已导入但当前未实际使用
tenacity
tqdm
```

**环境变量**：

| 变量名 | 用途 | 说明 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | VectorDBIngestor 调用 DashScope Embedding | 在 `__init__` 中直接赋值给 `dashscope.api_key` |

> 注意：代码中虽然导入了 `openai` 和 `dotenv`，但当前 `VectorDBIngestor` 的 embedding 实现**硬编码为 DashScope**，未使用 OpenAI。

---

## 3. 输入数据格式

两个 Ingestor 消费的输入为同一套**分块后的报告 JSON**，路径：`all_reports_dir/*.json`

### 3.1 报告 JSON 结构（Ingestion 消费侧）

```json
{
  "metainfo": {
    "sha1": "abc123...",
    "company_name": "示例科技"
  },
  "content": {
    "chunks": [
      {"text": "第一块文本内容..."},
      {"text": "第二块文本内容..."}
    ]
  }
}
```

**关键字段约束**：
- `metainfo.sha1`：必须存在且非空，用于生成输出文件名
- `content.chunks`：必须存在，元素为 dict，至少包含 `text` 字段

---

## 4. BM25Ingestor

### 4.1 接口

```python
class BM25Ingestor:
    def __init__(self)
    
    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path)
```

### 4.2 create_bm25_index

| 项 | 说明 |
|----|------|
| 输入 | `chunks: List[str]` — 纯文本字符串列表 |
| 分词方式 | `chunk.split()` — 按任意空白符分割，不做中文分词 |
| 输出 | `BM25Okapi` 索引对象 |

### 4.3 process_reports

批量处理目录下所有 `*.json` 报告：

1. `output_dir.mkdir(parents=True, exist_ok=True)` — 自动创建输出目录
2. 遍历 `all_reports_dir.glob("*.json")`
3. 每份报告：
   - 读取 JSON
   - 提取 `content.chunks` 中的 `text` 字段
   - 调用 `create_bm25_index` 构建索引
   - 保存为 `pickle`，文件名为 `{sha1}.pkl`
4. 打印处理数量

**输出文件**：
```
output_dir/
├── {sha1_1}.pkl
├── {sha1_2}.pkl
└── ...
```

---

## 5. VectorDBIngestor

### 5.1 接口

```python
class VectorDBIngestor:
    def __init__(self)
    
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-v1") -> List[List[float]]
    
    def _create_vector_db(self, embeddings: List[List[float]]) -> faiss.IndexFlatIP
    
    def _process_report(self, report: dict) -> faiss.IndexFlatIP
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path)
```

### 5.2 __init__

- 直接读取环境变量 `DASHSCOPE_API_KEY`
- 赋值给全局 `dashscope.api_key`
- 未做异常处理（若环境变量缺失，后续 API 调用会失败）

### 5.3 _get_embeddings

**签名**：
```python
@retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
def _get_embeddings(self, text, model="text-embedding-v1")
```

**重试策略**：固定间隔 20 秒，最多 2 次尝试（含首次）。

**输入处理流程**：

| 步骤 | 行为 | 异常 |
|------|------|------|
| 1 | 若输入为 `str` 且 `strip()` 后为空 | `ValueError("Input text cannot be an empty string.")` |
| 2 | 统一转为一维字符串列表 `text_chunks` | — |
| 3 | 类型检查：所有元素必须为 `str` | `ValueError("所有待嵌入文本必须为字符串类型！...")` |
| 4 | 过滤 `strip()` 后为空的字符串 | 若过滤后为空列表，抛 `ValueError("所有待嵌入文本均为空字符串！")` |
| 5 | 分批调用 DashScope API | — |

**批处理逻辑**：
- `MAX_BATCH_SIZE = 25`
- 循环 `range(0, len(text_chunks), MAX_BATCH_SIZE)` 分批发送
- 每批调用：`TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v1, input=batch)`

**响应解析**（兼容两种格式）：

| 响应结构 | 处理逻辑 |
|----------|----------|
| `resp['output']['embeddings']` | 遍历每条 embedding，检查 `emb['embedding']` 是否为空/None。若为空，将异常文本追加写入 `embedding_error.log`，抛 `RuntimeError` |
| `resp['output']['embedding']` | 单条格式，检查是否为空，逻辑同上 |
| 其他 | 抛 `RuntimeError("DashScope embedding API返回格式异常: ...")` |

**空 embedding 处理细节**：
- 错误日志文件硬编码为 `embedding_error.log`，追加模式（`'a'`）
- 日志内容包含：`text_index`、对应文本、分隔线
- 单条格式下记录 `batch[0]`

**返回值**：`List[List[float]]` — 与过滤后 `text_chunks` 一一对应的 embedding 列表

> ⚠️ 注意：`model` 参数在方法签名中存在，但实际调用时**未使用**，硬编码为 `text_embedding_v1`。

### 5.4 _create_vector_db

| 项 | 说明 |
|----|------|
| 输入 | `embeddings: List[List[float]]` |
| 数组转换 | `np.array(embeddings, dtype=np.float32)` |
| 维度 | 取 `len(embeddings[0])` |
| 索引类型 | `faiss.IndexFlatIP(dimension)` — **内积（Inner Product）**，非 L2 |
| 添加向量 | `index.add(embeddings_array)` |
| 输出 | `faiss.IndexFlatIP` 对象 |

> ⚠️ 注意：索引使用 `IndexFlatIP`，对应**余弦相似度**（需前置归一化），但当前代码未对 embedding 做归一化。若 DashScope 返回的向量未归一化，则 `IP` 结果与真实余弦相似度存在偏差。

### 5.5 _process_report

针对**单份报告**的处理流程：

1. 提取 `report['content']['chunks']` 中的 `text`
2. **过滤 + 截断**：
   - 过滤 `len(t) > 0` 的文本
   - 截断到 `max_len = 2048` 字符：`t[:max_len]`
3. 调用 `_get_embeddings` 获取全部 embedding
4. 调用 `_create_vector_db` 构建 FAISS 索引
5. 返回索引对象

> ⚠️ 注意：截断按**字符数**而非 token 数，与 `text_splitter.py` 中的 token 统计逻辑不一致。

### 5.6 process_reports

批量处理目录下所有 `*.json` 报告：

1. `output_dir.mkdir(parents=True, exist_ok=True)`
2. 遍历 `all_reports_dir.glob("*.json")`，带 tqdm 进度条
3. 每份报告：
   - 读取 JSON
   - 调用 `_process_report(report_data)`
   - 取 `metainfo.sha1`，若为空抛 `ValueError`
   - 保存为 `faiss.write_index(index, str(faiss_file_path))`
   - 文件名为 `{sha1}.faiss`
4. 打印处理数量

**输出文件**：
```
output_dir/
├── {sha1_1}.faiss
├── {sha1_2}.faiss
└── ...
```

---

## 6. 异常与边界行为

| 场景 | 行为 |
|------|------|
| BM25: `all_reports_dir` 为空 | `glob` 返回空列表，输出 `"Processed 0 reports"` |
| BM25: JSON 缺少 `content.chunks` | 代码未做防御，会抛 `KeyError` |
| BM25: JSON 缺少 `metainfo.sha1` | 代码未做防御，会抛 `KeyError` |
| Vector: `DASHSCOPE_API_KEY` 未设置 | `dashscope.api_key = None`，API 调用时由 dashscope SDK 抛错 |
| Vector: 单条文本为空字符串 | `ValueError`（过滤前拦截） |
| Vector: 所有文本过滤后为空 | `ValueError` |
| Vector: 某条 embedding 返回空 | `RuntimeError`，同时追加写入 `embedding_error.log` |
| Vector: JSON 缺少 `metainfo.sha1` | `ValueError("分块报告 ... 缺少 sha1 字段...")` |
| Vector: `content.chunks` 结构异常 | 会抛 `KeyError` 或 `TypeError`（未防御） |

---

## 7. 当前实现缺陷（需后续优化）

1. **硬编码参数过多**：
   - `MAX_BATCH_SIZE = 25`
   - `max_len = 2048`
   - `LOG_FILE = 'embedding_error.log'`
   - 重试间隔 20 秒、最多 2 次
   - 这些均应改为可配置参数

2. **未使用 OpenAI**：代码导入了 `OpenAI` 和 `dotenv`，但 `VectorDBIngestor` 完全硬编码为 DashScope

3. **FAISS 索引类型与归一化**：使用 `IndexFlatIP` 但**未对 embedding 做 L2 归一化**，若 DashScope 返回非单位向量，则 IP ≠ cosine similarity

4. **截断逻辑不一致**：`_process_report` 按字符数截断（2048 chars），而 `text_splitter.py` 按 token 数分块

5. **日志污染**：`print('11111111')`、`print('22222222')`、`print('33333333')` 等调试输出残留

6. **异常文本日志记录**：空 embedding 时仅记录日志文件，未对正常流程做降级处理（如跳过该条）

7. **BM25 分词缺陷**：`chunk.split()` 对中文效果极差（按字分词），应改用jieba或其他中文分词器

---

## 8. 使用示例

### 8.1 BM25 索引构建

```python
from pathlib import Path
from src.ingestion import BM25Ingestor

ingestor = BM25Ingestor()
ingestor.process_reports(
    all_reports_dir=Path("data/reports_chunked"),
    output_dir=Path("data/bm25_index")
)
```

### 8.2 向量索引构建

```python
import os
from pathlib import Path
from src.ingestion import VectorDBIngestor

os.environ["DASHSCOPE_API_KEY"] = "your-key"
ingestor = VectorDBIngestor()
ingestor.process_reports(
    all_reports_dir=Path("data/reports_chunked"),
    output_dir=Path("data/vector_index")
)
```

---

## 9. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始实现，支持 BM25 + DashScope/FAISS 向量索引构建 |
