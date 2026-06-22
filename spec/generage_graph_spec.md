# Generation Graph Spec

> **同步声明**：本文档描述 `src/generage_graph.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`generage_graph.py` 是基于 LangGraph 实现的**自适应生成工作流**（Self-RAG 的生成侧）。核心职责是：给定用户问题与检索到的文档列表，生成答案并对生成质量进行自检；若检测到幻觉或答案未回答问题，则改写问题后在同文档内再生成一次；仍失败则输出 `should_retry_retrieval` 供上游（如 `rag_app`）重新检索。

本工作流**不执行检索**，输入文档列表由上游模块（如 `src.retrieval_graph`）提供。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 基于文档生成 | 严格依据输入文档生成答案，禁止引入外部知识 |
| 幻觉自检 | 使用 LLM 评分器判断生成内容是否基于给定文档 |
| 答案相关性自检 | 使用 LLM 评分器判断生成内容是否回答用户问题 |
| 查询改写 | 评分失败时改写问题，同文档内再生成一次 |
| 有限重试 | 同文档内最多 2 次生成尝试 |
| 重检索信号 | 失败时输出 `should_retry_retrieval`，由 `rag_app` 编排重检索 |
| 直接生成开关 | `is_direct_generate=True` 时跳过评估与重试，便于对比测试 |
| 接受外部文档 | 输入为 `List[Document]`，与检索模块解耦 |

---

## 3. 架构设计

```
┌─────────┐     ┌──────────┐     route_after_generate
│  START  │────→│ generate │────┬──────────────────────→ finalize → END（is_direct_generate=True）
└─────────┘     └──────────┘    │
                                │ grade
                                ▼
                        ┌─────────────────┐     ┌──────────────────┐
                        │ grade_generation │────→│ route_after_grade │
                        └─────────────────┘     └────────┬─────────┘
                                                         │
              ┌──────────────────────────────────────────┤
              │ 基于文档且回答问题                        │ 未通过评分
              ▼                                          ▼
         finalize → END                            transform_query
                                                         │
                                                         ▼
                                                 should_continue
                                                         │
                                     ┌───────────────────┴───────────────────┐
                                     │ attempts < 2                            │ attempts >= 2
                                     ▼                                       ▼
                                  generate                            finalize → END
```

---

## 4. 状态定义 (GraphState)

```python
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    generation_attempts: int
    is_grounded_in_docs: bool
    is_question_answered: bool
    is_direct_generate: bool
    should_retry_retrieval: bool
    failure_reason: str
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | `str` | 当前查询问题（可能被改写） |
| `documents` | `List[Document]` | 检索到的文档列表，作为生成上下文 |
| `generation` | `str` | LLM 生成的答案 |
| `generation_attempts` | `int` | 已执行的生成尝试次数 |
| `is_grounded_in_docs` | `bool` | 当前/最终生成是否基于检索文档；直接生成模式下未评估，恒为 `False` |
| `is_question_answered` | `bool` | 当前/最终生成是否回答了用户问题；直接生成模式下未评估，恒为 `False` |
| `is_direct_generate` | `bool` | 为 `True` 时跳过质量评估与重试，生成一次后直接结束 |
| `should_retry_retrieval` | `bool` | 工作流结束时是否建议上游重新检索 |
| `failure_reason` | `str` | `ok` \| `hallucination` \| `not_answered` \| `skipped` |

---

## 5. 节点定义

### 5.1 generate

- **职责**：基于 `documents` 与 `question` 生成答案
- **输入**：`question`, `documents`, `generation_attempts`
- **输出**：`generation`, `generation_attempts` (+1)
- **实现要点**：
  - 调用 `RAGGenerator.invoke({"context": ..., "question": ...})`
  - `context` 由 `format_docs(documents)` 拼接而成

### 5.2 grade_generation

- **职责**：对生成内容进行幻觉检测与答案相关性评分，并将结果写入 state
- **输入**：`question`, `documents`, `generation`
- **输出**：`is_grounded_in_docs`, `is_question_answered`
- **实现要点**：
  - 先调用 `GenerationGrader.check_hallucination`
  - 仅当基于文档时，再调用 `GenerationGrader.check_answer`

### 5.3 transform_query

- **职责**：使用 LLM 改写用户问题以改进生成效果
- **输入**：`question`
- **输出**：`question` (改写后)
- **实现**：调用 `QuestionRewriter.invoke({"question": ...})`

### 5.4 finalize

- **职责**：在工作流结束前统一写入 `should_retry_retrieval` 与 `failure_reason`
- **规则**：
  - `is_direct_generate=True` → `should_retry_retrieval=False`, `failure_reason=skipped`
  - grounded 且 answered → `should_retry_retrieval=False`, `failure_reason=ok`
  - 未 grounded → `should_retry_retrieval=True`, `failure_reason=hallucination`
  - grounded 但未 answered → `should_retry_retrieval=True`, `failure_reason=not_answered`

---

## 6. 条件边定义

### 6.1 route_after_generate

- **职责**：生成后根据 `is_direct_generate` 决定是否进入质量评估
- **规则**：
  - `is_direct_generate == True` → `"end"`，进入 `finalize`
  - `is_direct_generate == False` → `"grade"`，进入 `grade_generation`

### 6.2 route_after_grade

- **职责**：根据评分结果决定下一步
- **规则**：
  1. `is_grounded_in_docs == True` 且 `is_question_answered == True` → `"useful"` → `finalize`
  2. 其他情况 → `"not useful"` → `transform_query`

### 6.3 should_continue

- **职责**：防止在 `transform_query` 后无限循环
- **规则**：
  - `generation_attempts < 2` → `"generate"`
  - `generation_attempts >= 2` → `"end"` → `finalize`

---

## 7. 辅助类

### 7.1 RAGGenerator

```python
class RAGGenerator:
    def __init__(self, llm: ChatOpenAI | None = None)
    def invoke(self, inputs: dict) -> str
```

### 7.2 QuestionRewriter

```python
class QuestionRewriter:
    def __init__(self, llm: ChatOpenAI | None = None)
    def invoke(self, inputs: dict) -> str
```

### 7.3 GenerationGrader

```python
class GenerationGrader:
    def __init__(self, llm: ChatOpenAI | None = None)
    def check_hallucination(self, documents: List[Document], generation: str) -> str
    def check_answer(self, question: str, generation: str) -> str
```

### 7.4 结构化输出模型

```python
class GradeHallucinations(BaseModel):
    binary_score: str  # "yes" 或 "no"

class GradeAnswer(BaseModel):
    binary_score: str  # "yes" 或 "no"
```

---

## 8. 依赖清单

**内部依赖**：
- `src.config.settings`
- `langgraph.graph.StateGraph, END, START`
- `langchain_core.documents.Document`
- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_core.output_parsers.StrOutputParser`
- `langchain_openai.ChatOpenAI`
- `pydantic.BaseModel, Field`

---

## 9. 使用示例

```python
from langchain_core.documents import Document
from src.generage_graph import app

inputs = {
    "question": "中芯国际2024年营业收入是多少？",
    "documents": [
        Document(
            page_content="中芯国际 2024 年营业收入为 577.96 亿元，同比增长 27.0%。",
            metadata={"page": 1},
        )
    ],
    "generation": "",
    "generation_attempts": 0,
    "is_grounded_in_docs": False,
    "is_question_answered": False,
    "is_direct_generate": False,
    "should_retry_retrieval": False,
    "failure_reason": "skipped",
}

final_state = app.invoke(inputs)
print(final_state["generation"])
print(final_state["should_retry_retrieval"])
```

---

## 10. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| `is_direct_generate=True` | 仅执行一次 `generate` 后 `finalize`；`failure_reason=skipped` |
| 首次生成即通过幻觉与答案评分 | `failure_reason=ok`，`should_retry_retrieval=False` |
| 评分失败且 `generation_attempts < 2` | `transform_query` 后再次 `generate` |
| 同文档内 2 次生成仍失败 | `finalize` 写出 `should_retry_retrieval=True` |
| `documents` 为空列表 | 通常 `hallucination`，建议上游重检索 |

单次 invoke LLM 上限（正常模式）：通过约 3 次；内层用尽约 5～6 次。

---

## 11. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始版本 |
| v1.1 | 2026-06-12 | 新增 `is_grounded_in_docs`、`is_question_answered` |
| v1.2 | 2026-06-12 | 新增 `is_direct_generate` |
| v1.3 | 2026-06-22 | 新增 `should_retry_retrieval`、`failure_reason` 与 `finalize`；内层最多 2 次生成；移除同输入盲重试 |
