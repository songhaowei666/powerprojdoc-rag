# Retrieval Graph Spec

> **同步声明**：本文档严格反向推导自 `src/retrieval_graph.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`retrieval_graph.py` 是基于 LangGraph 实现的**自适应检索工作流**。核心职责是：通过混合检索召回文档，经 LLM 相关性评分后，动态决定直接返回文档或改写查询后重新检索。本工作流**不生成自然语言答案**，最终输出为检索到的文档列表。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 混合检索 | 使用 `HybridRetriever`（向量检索 + LLM 重排）进行文档召回 |
| 相关性过滤 | 使用 `RetrievalGrader` 对召回文档进行二元相关性评分 |
| 查询改写 | 当首轮检索无相关文档时，自动改写查询并重新检索 |
| 最多两次检索 | 限制检索次数最多两次，避免无限循环 |
| 直接检索开关 | `is_direct_retrieve=True` 时跳过评估与重试，便于对比测试 |
| 直接返文档 | 不调用 LLM 生成答案，直接返回最终检索到的文档列表 |
| 无网络搜索 | 不依赖外部网络搜索工具，纯本地检索闭环 |

---

## 3. 架构设计

```
┌─────────┐    route_after_retrieve
│  START  │───→│ retrieve │───┬──────────────────────────────→ END（is_direct_retrieve=True）
└─────────┘    └──────────┘   │
                              │ grade
                              ▼
                        ┌───────────────┐
                        │ grade_documents│
                        └───────┬───────┘
                                │
             ┌──────────────────┼──────────────────┐
             │ 有相关文档        │ 无相关(第1次)     │ 无相关(第2次)
             ▼                  ▼                  ▼
           END           transform_query           END
                              │
                              ▼
                           retrieve
                              │
                              ▼
                        grade_documents
                              │
                              ▼
                             END
```

---

## 4. 状态定义 (GraphState)

```python
class GraphState(TypedDict):
    question: str
    company_code: str
    documents: List[Document]
    retrieval_attempts: int
    has_relevant_docs: bool
    is_direct_retrieve: bool
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | `str` | 当前查询问题（可能被改写） |
| `company_code` | `str` | 目标公司编码，用于 `HybridRetriever` 过滤 |
| `documents` | `List[Document]` | 检索到的文档列表 |
| `retrieval_attempts` | `int` | 已执行的检索次数 |
| `has_relevant_docs` | `bool` | 当前文档是否经过评分且包含相关文档；直接检索模式下未评估，恒为 `False` |
| `is_direct_retrieve` | `bool` | 为 `True` 时跳过相关性评估与重试，检索一次后直接结束 |

---

## 5. 节点定义

### 5.1 retrieve

- **职责**：使用 `HybridRetriever` 执行混合检索
- **输入**：`question`, `company_code`, `retrieval_attempts`
- **输出**：`documents` (List[Document]), `retrieval_attempts` (+1)
- **实现要点**：
  - 调用 `HybridRetriever.retrieve(..., return_parent_pages=True)`
  - 将返回的 `List[Dict]` 转换为 `List[Document]`，其中 `page_content` 取自 `text`，其余字段放入 `metadata`

### 5.2 grade_documents

- **职责**：对检索到的文档进行 LLM 二元相关性评分
- **输入**：`question`, `documents`, `retrieval_attempts`
- **输出**：`documents` (过滤后), `has_relevant_docs`
- **边界行为**：
  - 若 `retrieval_attempts >= 2` 且仍无相关文档，保留原始检索结果返回，不过滤为空列表

### 5.3 transform_query

- **职责**：使用 LLM 改写用户问题以改进检索效果
- **输入**：`question`
- **输出**：`question` (改写后)
- **实现**：调用 `QuestionRewriter.invoke({"question": ...})`

### 5.4 route_after_retrieve

- **职责**：检索后根据 `is_direct_retrieve` 决定是否进入相关性评估
- **规则**：
  - `is_direct_retrieve == True` → `"end"`，跳过 `grade_documents`
  - `is_direct_retrieve == False` → `"grade"`，进入 `grade_documents`

### 5.5 decide_next_step

- **职责**：基于 `has_relevant_docs` 和 `retrieval_attempts` 决定下一步
- **规则**：
  - `has_relevant_docs == True` → `"end"`
  - `has_relevant_docs == False` 且 `retrieval_attempts == 1` → `"transform_query"`
  - `has_relevant_docs == False` 且 `retrieval_attempts >= 2` → `"end"`

---

## 6. 辅助类

### 6.1 QuestionRewriter

```python
class QuestionRewriter:
    def __init__(self, provider: str = "openai")
    def invoke(self, inputs: dict) -> str
```

- 使用 `APIProcessor` 调用 LLM
- System Prompt 定位：查询重写专家，改进问题使其更适合文档检索
- 返回纯字符串形式的重写后问题

---

## 7. 依赖清单

**内部依赖**：
- `src.retrieval.HybridRetriever`
- `src.post_retrieval_correction.retrieval_grader`
- `src.api_requests.APIProcessor`
- `src.config.settings`
- `langgraph.graph.StateGraph, END, START`
- `langchain.schema.Document`

---

## 8. 使用示例

```python
from src.retrieval_graph import app

inputs = {
    "question": "工程总投资是多少？",
    "company_code": "001",
    "documents": [],
    "retrieval_attempts": 0,
    "has_relevant_docs": False,
    "is_direct_retrieve": False,
}

final_state = app.invoke(inputs)
for doc in final_state["documents"]:
    print(f"Page {doc.metadata.get('page')}: {doc.page_content[:200]}...")

# 直接检索模式（跳过评估，用于对比测试）
direct_inputs = {**inputs, "is_direct_retrieve": True}
direct_state = app.invoke(direct_inputs)
```

---

## 9. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| `is_direct_retrieve=True` | 仅执行一次 `retrieve` 后结束；不调用 `RetrievalGrader`；`has_relevant_docs` 保持 `False` |
| 首轮检索即有相关文档 | 直接返回经 `RetrievalGrader` 过滤后的文档列表 |
| 首轮无相关文档 | 改写查询后执行第二轮混合检索，再经评分后决定 |
| 第二轮仍无相关文档 | 返回第二轮原始检索结果（不过滤为空），由调用方自行处理 |
| 检索器返回空列表 | `grade_documents` 处理空列表，`has_relevant_docs=False`，进入第二轮或结束 |
| `company_code` 为空或不存在 | 由 `HybridRetriever` 底层抛出异常 |

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始版本；支持混合检索、相关性评分、查询改写、两次检索上限、直接返文档 |
| v1.1 | 2026-06-12 | `GraphState` 与检索入参由 `company_name` 改为 `company_code` |
| v1.2 | 2026-06-12 | 新增 `is_direct_retrieve` 开关，支持跳过相关性评估的直接检索模式 |
