# Evaluation Module Spec

> **同步声明**：本文档描述 `eval/evaluation.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`eval/evaluation.py` 是 RAG 系统的评估模块，所有评估代码统一放在 `eval/` 目录下。该模块提供**离线批量评估**与**单条实时评估**两种能力。模块基于 ragas 框架计算标准 RAG 指标，同时自定义「页面召回率」以适配本项目「按页面检索」的场景。

评估不引入 trulens，全部指标由 ragas + 自定义逻辑覆盖；单条实时评估接口可直接嵌入生产链路做轻量监控。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 分层评估 | 检索、响应、整体三层评估，职责分离 |
| 页面级粒度 | 自定义 `page_recall@k`，精确到年报页码 |
| 离线 + 实时 | `RAGEvaluator` 跑批量评测集；`SingleTurnEvaluator` 做单条实时打分 |
| 统一 LLM 配置 | 评估器 LLM 走项目 `config.py`，与生成阶段保持一致 |
| 失败不中断 | 批量评估时单条失败记为 NaN，不阻断整批 |

---

## 3. 架构设计

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   EvalDataset   │────→│  RAGEvaluator   │────→│  ragas.evaluate │
│   (评测集加载)   │     │  (批量离线评估)  │     │  (标准指标计算)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ SingleTurnEvaluator │
                       │   (单条实时评估)   │
                       └─────────────────┘
```

---

## 4. 依赖清单

```
python >= 3.10
ragas >= 0.4.0
pandas
langchain-core
langchain-openai
```

**内部依赖**：
- `src.rag_app.RAGApp`：执行完整 RAG 流程获取检索结果与生成答案
- `src.config.settings`：读取 `chat_model`、`openai_api_key`、`openai_api_base`
- `src.generage_graph.build_llm`：构建评估器所用的 ChatOpenAI 实例

---

## 5. 输入数据格式

### 5.1 评估集 JSON (`eval_dataset.json`)

对象数组，每条样本包含以下字段：

```json
[
  {
    "question": "中芯国际2024年营业收入是多少？",
    "expected_answer": "2024年营业收入为577.96亿元。",
    "expected_source_doc": "中芯国际2024年年度报告",
    "expected_source_pages": [12, 13],
    "company_code": "001"
  }
]
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | `str` | 是 | 用户查询 |
| `expected_answer` | `str` | 是 | 预期回答，用于 answer_correctness 与 context_recall |
| `expected_source_doc` | `str` | 是 | 预期来源文档名，供人工核对 |
| `expected_source_pages` | `List[int]` | 是 | 预期来源页面号列表；为空列表时 page_recall@k 不参与计算 |
| `company_code` | `str` | 是 | 目标公司代码，传递给检索器做文档隔离 |

### 5.2 评估集 CSV (`eval/data/eval_dataset.csv`)

列名与 JSON 字段一致，额外约定：

| 列 | 说明 |
|----|------|
| `expected_source_pages` | JSON 数组字符串，如 `"[3,4]"` |
| `company_code` | 目标公司代码；旧列名 `company_name` 会自动映射 |

示例：

```csv
question,expected_answer,expected_source_doc,expected_source_pages,company_code
"问题文本","预期答案","来源文档名","[3,4]","001"
```

`EvalDataset.from_path(path)` 按后缀自动选择 `from_csv`（`.csv`）或 `from_json`（`.json`）。

---

## 6. 指标定义

### 6.1 页面召回率 `page_recall@k`

**公式**：

```
page_recall@k = |{page | page ∈ expected_pages, page 出现在前 k 个结果的 metadata.page 中}| / |expected_pages|
```

- 取前 `k` 个检索结果
- 对每个结果读取 `doc.metadata.get("page")`，若该页面号在 `expected_source_pages` 中则计为命中
- 同一页面多次出现只计一次命中
- 分母为预期页面集合大小 `|expected_pages|`

**边界行为**：
- `expected_source_pages` 为空列表 → 返回 `None`（无法评估）
- `documents` 为空列表 → 返回 `0.0`
- `doc.metadata` 缺少 `"page"` 键 → 该文档视为未命中

### 6.2 ragas 标准指标

| 指标 | 维度 | 需要字段 | 说明 |
|------|------|----------|------|
| `context_precision` | 检索 | `user_input`, `retrieved_contexts`, `reference` | 检索上下文中有多少与问题相关 |
| `context_recall` | 检索 | `user_input`, `retrieved_contexts`, `reference` | 预期回答中的信息有多少被上下文覆盖 |
| `faithfulness` | 响应 | `response`, `retrieved_contexts` | 答案中的陈述有多少能从上下文中找到依据 |
| `answer_relevancy` | 响应 | `response`, `user_input`, `retrieved_contexts` | 答案与问题的相关程度 |
| `answer_correctness` | 响应 | `response`, `reference`, `user_input` | 答案与预期回答的语义正确程度 |

> ragas 指标默认由 LLM（gpt-4o）判定，可通过 `evaluator_llm` 参数替换模型。

---

## 7. 类与函数接口

### 7.1 EvalDataset

```python
class EvalDataset:
    def __init__(self, samples: List[dict])
    
    @classmethod
    def from_json(cls, path: Path) -> "EvalDataset"
    
    @classmethod
    def from_csv(cls, path: Path) -> "EvalDataset"
    
    @classmethod
    def from_path(cls, path: Path) -> "EvalDataset"
    
    def __len__(self) -> int
    
    def __getitem__(self, idx: int) -> dict
    
    def to_list(self) -> List[dict]
```

- `from_json` 从 JSON 文件加载评估集，不做字段强校验，但要求顶层为数组
- `from_csv` 从 CSV 文件加载；`expected_source_pages` 支持 JSON 字符串解析；`company_name` 列自动映射为 `company_code`
- `from_path` 按后缀 `.csv` / `.json` 自动分发
- `to_list` 返回内部样本列表的浅拷贝

---

### 7.2 compute_page_recall_at_k

```python
def compute_page_recall_at_k(
    documents: List[Document],
    expected_pages: List[int],
    top_k: int = 6,
) -> Optional[float]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `documents` | `List[Document]` | - | 检索到的文档列表 |
| `expected_pages` | `List[int]` | - | 预期页面号列表 |
| `top_k` | `int` | `6` | 评估截止位置 |

**返回**：`float`（0.0 ~ 1.0）或 `None`（`expected_pages` 为空时）

---

### 7.3 RAGEvaluator（批量离线评估器）

```python
class RAGEvaluator:
    def __init__(
        self,
        rag_app: Optional[RAGApp] = None,
        evaluator_llm: Optional[BaseChatModel] = None,
    )
    
    def run_batch(
        self,
        dataset: EvalDataset,
        top_k: int = 6,
    ) -> pd.DataFrame
```

| 参数 | 说明 |
|------|------|
| `rag_app` | 注入的 RAG 应用实例；为 `None` 时使用默认 `RAGApp()` |
| `evaluator_llm` | 传给 ragas 的评估 LLM；为 `None` 时通过 `build_llm(temperature=0)` 构造 |

**`run_batch` 执行流程**：

1. **RAG 执行**：遍历 `dataset`，对每个样本调用 `rag_app.run(question, company_code)`
   - 成功：收集 `question`、`generation`、`documents`
   - 失败：打印警告，该样本的 generation 与 documents 记为空

2. **页面召回率计算**：对每个样本调用 `compute_page_recall_at_k`

3. **ragas 批量评估**：
   - 构建 `EvaluationDataset`，字段映射如下：
     - `user_input` → `question`
     - `response` → `generation`
     - `retrieved_contexts` → `[d.page_content for d in documents]`
     - `reference` → `expected_answer`
   - 调用 `ragas.evaluate(dataset, metrics=[...], llm=evaluator_llm)`

4. **结果合并**：将 ragas 结果（`Result.to_pandas()`）与 `page_recall@k` 按行合并

5. **输出 DataFrame**：列名包含
   - `question`, `expected_answer`, `generation`, `expected_source_pages`
   - `page_recall@k`
   - `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `answer_correctness`

**边界行为**：
- 若 ragas 评估阶段整体失败，抛出 `RuntimeError`，并附带原始异常信息
- 若某条样本的 `expected_answer` 为空字符串，ragas 指标可能为 `NaN`（ragas 内部行为）

---

### 7.4 SingleTurnEvaluator（单条实时评估器）

```python
class SingleTurnEvaluator:
    def __init__(
        self,
        evaluator_llm: Optional[BaseChatModel] = None,
    )
    
    def evaluate(
        self,
        question: str,
        generation: str,
        documents: List[Document],
        expected_answer: str = "",
        expected_pages: Optional[List[int]] = None,
        top_k: int = 6,
    ) -> dict
```

| 参数 | 说明 |
|------|------|
| `question` | 用户查询 |
| `generation` | RAG 生成的答案 |
| `documents` | 检索到的文档列表 |
| `expected_answer` | 预期回答；为空时不计算 `context_recall` 和 `answer_correctness` |
| `expected_pages` | 预期页面号列表；为 `None` 或空列表时不计算 `page_recall@k` |
| `top_k` | 页面召回率的评估截止位置 |

**返回结构**：

```python
{
    "page_recall@k": float | None,
    "context_precision": float | None,
    "context_recall": float | None,
    "faithfulness": float | None,
    "answer_relevancy": float | None,
    "answer_correctness": float | None,
}
```

**指标计算规则**：

| 指标 | 是否需要 `expected_answer` | 说明 |
|------|---------------------------|------|
| `page_recall@k` | 否 | 依赖 `expected_pages`；为空时返回 `None` |
| `context_precision` | 否 | 判断检索上下文与问题的相关性 |
| `faithfulness` | 否 | 判断答案是否基于检索上下文 |
| `answer_relevancy` | 否 | 判断答案与问题的相关程度 |
| `context_recall` | **是** | 需要 `reference` 拆解 claim，为空时返回 `None` |
| `answer_correctness` | **是** | 需要与 `reference` 对比语义，为空时返回 `None` |

**边界行为**：
- `expected_answer` 为空字符串时，调用 ragas 时不传入 `reference`，且不请求 `context_recall` 和 `answer_correctness`，这两个指标固定返回 `None`
- ragas 单条评估失败时，抛出 `RuntimeError`

---

### 7.5 报告生成（`eval/report.py`）

```python
def save_detail_csv(df: pd.DataFrame, path: Path) -> None

def generate_markdown_report(
    df: pd.DataFrame,
    *,
    dataset_path: Path,
    top_k: int,
    output_path: Path,
    run_timestamp: Optional[datetime] = None,
) -> str
```

| 函数 | 说明 |
|------|------|
| `save_detail_csv` | 将 `run_batch` 返回的 DataFrame 写入 CSV；`expected_source_pages` 序列化为 JSON 字符串 |
| `generate_markdown_report` | 生成 Markdown 报告并写入文件；返回报告文本 |

**Markdown 报告结构**：

| 章节 | 内容 |
|------|------|
| 运行信息 | 时间戳、数据集路径、样本数、top_k、评估模型 |
| 汇总指标 | 各数值列 mean 与 NaN 样本数 |
| 分项明细 | 每题 question、指标得分、generation 摘要（截断 200 字） |
| 低分样本 | 任一指标 < 0.5 或 NaN 的样本 |

---

## 8. 使用示例

### 8.1 批量离线评估

```python
from pathlib import Path
from eval.evaluation import EvalDataset, RAGEvaluator

dataset = EvalDataset.from_json(Path("eval/eval_dataset.json"))
evaluator = RAGEvaluator()

df = evaluator.run_batch(dataset, top_k=6)
print(df[["question", "page_recall@k", "faithfulness", "answer_correctness"]])
print(f"平均页面召回率: {df['page_recall@k'].mean()}")
```

### 8.2 单条实时评估

```python
from eval.evaluation import SingleTurnEvaluator
from langchain_core.documents import Document

evaluator = SingleTurnEvaluator()

result = evaluator.evaluate(
    question="中芯国际2024年营业收入是多少？",
    generation="2024年营业收入为577.96亿元。",
    documents=[Document(page_content="...", metadata={"page": 12})],
    expected_answer="2024年营业收入为577.96亿元。",
    expected_pages=[12, 13],
    top_k=6,
)
print(result)
```

### 8.3 CLI 离线评估

```bash
python eval/run_offline_eval.py \
  --dataset eval/data/eval_dataset.csv \
  --top-k 6 \
  --output-dir eval/reports
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `eval/data/eval_dataset.csv` | 评估集路径（`.csv` 或 `.json`） |
| `--top-k` | `6` | page_recall@k 截止位置 |
| `--output-dir` | `eval/reports/` | 报告输出目录 |
| `--prefix` | 时间戳 `YYYYMMDD_HHMMSS` | 输出文件名前缀 |

输出文件：
- `{prefix}_detail.csv`：逐样本明细
- `{prefix}_report.md`：Markdown 汇总报告

---

## 9. 性能与限制

| 项目 | 说明 |
|------|------|
| LLM 调用量 | ragas 每条样本每个指标至少 1 次 LLM 调用；批量评估建议先在小数据集（<20 条）上验证 |
| 并发 | `run_batch` 中 RAG 调用串行执行，避免压垮检索器；ragas 内部自行管理并发 |
| 耗时 | 10 条样本的完整评估约 2~5 分钟（视网络与模型而定） |
| 评估器模型 | 默认使用 `settings.chat_model`；建议使用推理能力强于生成模型的模型（如 gpt-4o） |

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.3 | 2026-06-24 | 新增 CSV 加载（`from_csv`/`from_path`）、报告生成（`eval/report.py`）与 CLI（`eval/run_offline_eval.py`） |
| v1.2 | 2026-06-14 | 将自定义页面指标从 `page_precision@k` 改为 `page_recall@k`，公式从 `hits / k` 调整为 `unique hits / \|expected_pages\|` |
| v1.1 | 2026-06-12 | 将评估模块从 `src/evaluation.py` 迁移到 `eval/evaluation.py`，评估代码统一放入 `eval/` 目录 |
| v1.0 | 2026-06-11 | 初始版本；定义评估集 schema、page_precision@k、ragas 五指标、批量/单条两种评估器 |
