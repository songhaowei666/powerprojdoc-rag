# Generation Graph Spec

> **同步声明**：本文档描述 `src/generage_graph.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`generage_graph.py` 是基于 LangGraph 实现的**自适应生成工作流**（Self-RAG 的生成侧）。核心职责是：给定用户问题与检索到的文档列表，生成答案并对生成质量进行自检；若检测到幻觉或答案未回答问题，则自动改写查询并重新生成，最多重试固定次数。

本工作流**不执行检索**，输入文档列表由上游模块（如 `src.retrieval_graph`）提供。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 基于文档生成 | 严格依据输入文档生成答案，禁止引入外部知识 |
| 幻觉自检 | 使用 LLM 评分器判断生成内容是否基于给定文档 |
| 答案相关性自检 | 使用 LLM 评分器判断生成内容是否回答用户问题 |
| 查询改写 | 当答案未回答问题或存在幻觉时，自动改写查询后重试 |
| 有限重试 | 最多 3 次生成尝试，避免无限循环 |
| 直接生成开关 | `is_direct_generate=True` 时跳过评估与重试，便于对比测试 |
| 接受外部文档 | 输入为 `List[Document]`，与检索模块解耦 |

---

## 3. 架构设计

```
┌─────────┐     ┌──────────┐     route_after_generate
│  START  │────→│ generate │────┬──────────────────────────────→ END（is_direct_generate=True）
└─────────┘     └──────────┘    │
                                │ grade
                                ▼
                        ┌─────────────────┐     ┌──────────────────┐
                        │ grade_generation │────→│ route_after_grade │
                        └─────────────────┘     └────────┬─────────┘
                                                         │
              ┌──────────────────────────────────────────┼──────────────┐
              │ 基于文档且回答问题                          │ 未回答问题    │ 不基于文档
              ▼                                          ▼              │
             END                                   transform_query ◄────┘
                                                         │
                                                         ▼
                                                 should_continue
                                                         │
                                     ┌───────────────────┴───────────────────┐
                                     │ 尝试次数 < 3                            │ 尝试次数 >= 3
                                     ▼                                       ▼
                                  generate                                   END
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

---

## 5. 节点定义

### 5.1 generate

- **职责**：基于 `documents` 与 `question` 生成答案
- **输入**：`question`, `documents`, `generation_attempts`
- **输出**：`generation`, `generation_attempts` (+1)
- **实现要点**：
  - 调用 `RAGGenerator.invoke({"context": ..., "question": ...})`
  - `context` 由 `format_docs(documents)` 拼接而成
  - 使用 `APIProcessor(provider="openai")` 发送非结构化生成请求

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

---

## 6. 条件边定义

### 6.1 route_after_generate

- **职责**：生成后根据 `is_direct_generate` 决定是否进入质量评估
- **规则**：
  - `is_direct_generate == True` → `"end"`，跳过 `grade_generation`
  - `is_direct_generate == False` → `"grade"`，进入 `grade_generation`

### 6.2 route_after_grade

- **职责**：根据 `is_grounded_in_docs`、`is_question_answered` 与尝试次数决定下一步
- **规则**：
  1. `is_grounded_in_docs == True` 且 `is_question_answered == True` → `"useful"`
  2. `is_grounded_in_docs == False` 且 `generation_attempts >= 2` → `"not useful"`
  3. `is_grounded_in_docs == False` 且 `generation_attempts < 2` → `"not supported"`
  4. `is_grounded_in_docs == True` 且 `is_question_answered == False` → `"not useful"`

### 6.3 should_continue

- **职责**：防止在 transform_query 后无限循环
- **规则**：
  - `generation_attempts < 3` → `"generate"`
  - `generation_attempts >= 3` → `"end"`

---

## 7. 辅助类

### 7.1 RAGGenerator

```python
class RAGGenerator:
    def __init__(self, llm: ChatOpenAI | None = None)
    def invoke(self, inputs: dict) -> str
```

- 使用 `ChatOpenAI` 调用 LLM
- System Prompt 定位：基于检索文档回答问题的助手
- 返回纯字符串形式的答案

### 7.2 QuestionRewriter

```python
class QuestionRewriter:
    def __init__(self, llm: ChatOpenAI | None = None)
    def invoke(self, inputs: dict) -> str
```

- 使用 `ChatOpenAI` 调用 LLM
- System Prompt 定位：查询重写专家
- 返回纯字符串形式的重写后问题

### 7.3 GenerationGrader

```python
class GenerationGrader:
    def __init__(self, llm: ChatOpenAI | None = None)
    def check_hallucination(self, documents: List[Document], generation: str) -> str
    def check_answer(self, question: str, generation: str) -> str
```

- 使用 `ChatOpenAI.with_structured_output` 进行结构化评分
- 返回 `"yes"` 或 `"no"`

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
- `langchain.schema.Document`
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
}

final_state = app.invoke(inputs)
print(final_state["generation"])

# 直接生成模式（跳过评估，用于对比测试）
direct_inputs = {**inputs, "is_direct_generate": True}
direct_state = app.invoke(direct_inputs)
print(direct_state["generation"])
```

---

## 10. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| `is_direct_generate=True` | 仅执行一次 `generate` 后结束；不调用评分器；`is_grounded_in_docs` 与 `is_question_answered` 保持 `False` |
| 首次生成即通过幻觉与答案评分 | 直接返回答案 |
| 生成内容存在幻觉且尝试次数 < 2 | 回到 `generate` 重新生成 |
| 生成内容存在幻觉且尝试次数 >= 2 | 进入 `transform_query` 改写查询 |
| 生成内容回答了问题但无关 | 进入 `transform_query` 改写查询 |
| 达到最大尝试次数（3 次） | 直接返回当前生成结果，标识字段反映最后一次评分结果 |
| 正常结束 | `is_grounded_in_docs=True` 且 `is_question_answered=True` |
| `documents` 为空列表 | 生成器将基于空上下文生成，通常 `is_grounded_in_docs=False` |

---

## 11. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始版本；基于 Self-RAG 生成侧实现，支持生成、幻觉检测、答案评分、查询改写与有限重试 |
| v1.1 | 2026-06-12 | 新增 `is_grounded_in_docs`、`is_question_answered` 状态标识；评分逻辑拆分为独立节点 |
| v1.2 | 2026-06-12 | 新增 `is_direct_generate` 开关，支持跳过质量评估的直接生成模式 |
