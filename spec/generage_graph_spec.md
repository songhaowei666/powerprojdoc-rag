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
| 接受外部文档 | 输入为 `List[Document]`，与检索模块解耦 |

---

## 3. 架构设计

```
┌─────────┐     ┌──────────┐     ┌─────────────────────────────────┐
│  START  │────→│ generate │────→│ grade_generation_v_documents_   │
└─────────┘     └──────────┘     │ and_question                    │
                                 └─────────────┬───────────────────┘
                                               │
              ┌────────────────────────────────┼────────────────────────────────┐
              │ 基于文档且回答                 │ 未回答问题                     │ 不基于文档
              │ 问题                           │                                │
              ▼                                ▼                                │
             END                         transform_query ◄─────────────────────┘
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
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | `str` | 当前查询问题（可能被改写） |
| `documents` | `List[Document]` | 检索到的文档列表，作为生成上下文 |
| `generation` | `str` | LLM 生成的答案 |
| `generation_attempts` | `int` | 已执行的生成尝试次数 |

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

### 5.2 transform_query

- **职责**：使用 LLM 改写用户问题以改进生成效果
- **输入**：`question`
- **输出**：`question` (改写后)
- **实现**：调用 `QuestionRewriter.invoke({"question": ...})`

---

## 6. 条件边定义

### 6.1 grade_generation_v_documents_and_question

- **职责**：评估生成内容是否基于文档并回答问题
- **规则**：
  1. 先调用幻觉评分器 `GradeHallucinations`
     - 输出 `binary_score == "yes"` → 继续评估答案相关性
     - 输出 `binary_score == "no"` → 若 `generation_attempts >= 2` 返回 `"not useful"`，否则返回 `"not supported"`
  2. 再调用答案评分器 `GradeAnswer`
     - 输出 `binary_score == "yes"` → `"useful"`
     - 输出 `binary_score == "no"` → `"not useful"`

### 6.2 should_continue

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
}

final_state = app.invoke(inputs)
print(final_state["generation"])
```

---

## 10. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| 首次生成即通过幻觉与答案评分 | 直接返回答案 |
| 生成内容存在幻觉且尝试次数 < 2 | 回到 `generate` 重新生成 |
| 生成内容存在幻觉且尝试次数 >= 2 | 进入 `transform_query` 改写查询 |
| 生成内容回答了问题但无关 | 进入 `transform_query` 改写查询 |
| 达到最大尝试次数（3 次） | 直接返回当前生成结果 |
| `documents` 为空列表 | 生成器将基于空上下文生成，通常返回"无法回答" |

---

## 11. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始版本；基于 Self-RAG 生成侧实现，支持生成、幻觉检测、答案评分、查询改写与有限重试 |
