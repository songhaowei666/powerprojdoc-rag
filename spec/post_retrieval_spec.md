# Post-Retrieval Correction Module Spec

> **同步声明**：本文档严格反向推导自 `src/post_retrieval_correction.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`post_retrieval_correction.py` 是 RAG 检索流程的后处理验证模块，位于 `src/` 目录下。其核心职责是：对检索器召回的文档与用户问题的相关性进行 LLM 二元评分，滤除伪相关文档，提升下游生成阶段的信噪比。

该模块通常嵌入在检索节点之后、生成节点之前，作为 LangGraph 工作流的一个校验节点运行。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 二元判定 | 输出严格的 `yes` / `no`，便于调用方做布尔分支 |
| 单文档评分 | 一次只评一篇文档，降低 LLM 上下文复杂度与调用风险 |
| 结构化输出 | 使用 Pydantic 约束 LLM 返回，避免解析失败 |
| LangGraph 兼容 | 提供 `invoke(inputs: dict)` 统一接口，可直接作为节点函数 |
| 多模型适配 | 通过 `provider` 参数切换底层 LLM，复用 `APIProcessor` 能力 |

---

## 3. 架构设计

```
┌─────────────────────────────────────────────┐
│              RetrievalGrader                │
│         (检索后文档相关性评分器)               │
└──────────────────┬──────────────────────────┘
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
┌──────────────┐      ┌──────────────────┐
│ GradeDocuments│      │  APIProcessor    │
│ (Pydantic模型)│      │  (LLM调用封装)    │
└──────────────┘      └──────────────────┘
```

- **RetrievalGrader**：对外暴露评分接口，内部拼接 System Prompt + User Prompt，调用 LLM 做相关性判断。
- **GradeDocuments**：定义结构化输出 Schema，仅包含 `binary_score` 字段。
- **APIProcessor**：复用项目内统一的 LLM 请求封装，支持结构化输出（`response_format`）。

---

## 4. 依赖清单

```
python >= 3.10
pydantic
```

**内部依赖**：
- `src.api_requests.APIProcessor`：LLM 调用封装，需支持 `is_structured=True` + `response_format`

---

## 5. 数据模型

### 5.1 GradeDocuments

```python
class GradeDocuments(BaseModel):
    binary_score: str
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `binary_score` | `str` | 必须为 `"yes"`（相关）或 `"no"`（不相关） |

> 当前实现未对 `binary_score` 做额外的枚举校验，仅通过 Prompt 约束 LLM 输出。若后续需要强校验，可改用 `Literal["yes", "no"]`。

---

## 6. 类接口定义

### 6.1 RetrievalGrader

```python
class RetrievalGrader:
    def __init__(self, provider: str = "openai")
    
    def grade(self, question: str, document: str) -> GradeDocuments
    
    def invoke(self, inputs: dict) -> GradeDocuments
```

#### `__init__`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider` | `str` | `"openai"` | 传递给 `APIProcessor` 的提供商标识 |

初始化行为：
- 实例化 `APIProcessor(provider=provider)`
- 固化 System Prompt（见第 7 节）

#### `grade`

| 参数 | 类型 | 说明 |
|------|------|------|
| `question` | `str` | 用户原始问题 |
| `document` | `str` | 待评分的检索文档全文 |

**返回**：`GradeDocuments` 实例。

**内部调用链**：
1. 拼接 Human Prompt：`"检索到的文档：\n\n{document}\n\n用户问题：{question}"`
2. 调用 `APIProcessor.send_message(..., is_structured=True, response_format=GradeDocuments)`
3. 将返回字典解包为 `GradeDocuments(**result)`

#### `invoke`

兼容 LangChain / LangGraph Runnable 接口的包装方法。

| 参数 | 类型 | 说明 |
|------|------|------|
| `inputs` | `dict` | 必须包含 `"question"` 和 `"document"` 两个键 |

**返回**：`GradeDocuments` 实例。

**边界行为**：
- 若 `inputs` 缺少 `"question"` 或 `"document"` 键，直接抛出 `KeyError`
- 实际逻辑为透传调用 `self.grade(question, document)`

### 6.2 默认实例

模块末尾预置默认实例，供其他模块直接导入使用：

```python
retrieval_grader = RetrievalGrader()
```

---

## 7. Prompt 策略

### 7.1 System Prompt

```
你是一个评估检索到的文档与用户问题相关性的评分员。
如果文档包含与问题相关的关键词或语义，则将其评为相关。
给出一个二元评分'yes'或'no'来表示文档是否与问题相关。
```

**设计要点**：
- 明确角色定位（评分员），减少模型角色漂移。
- 判定标准宽松：只要包含相关的关键词或语义即视为相关，不过度要求精确答案。
- 输出严格限定为 `yes` / `no`。

### 7.2 Human Prompt 模板

```
检索到的文档：

{document}

用户问题：{question}
```

---

## 8. 使用示例

### 8.1 基础评分

```python
from src.post_retrieval_correction import RetrievalGrader

grader = RetrievalGrader(provider="openai")
result = grader.grade(
    question="公司营收增长原因",
    document="2024年公司营业收入同比增长15%，主要得益于新业务板块扩张..."
)
print(result.binary_score)  # "yes" 或 "no"
```

### 8.2 在 LangGraph 节点中使用

```python
from src.post_retrieval_correction import retrieval_grader
from langchain.schema import Document

def grade_documents(state):
    """过滤掉不相关文档。"""
    question = state["question"]
    documents = state["documents"]  # List[Document]
    
    filtered = []
    for d in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        if score.binary_score == "yes":
            filtered.append(d)
    
    return {"documents": filtered}
```

---

## 9. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| LLM 返回非预期格式 | 由 `APIProcessor` 层抛出异常或返回空字典；`GradeDocuments(**result)` 可能因字段缺失报错 |
| 文档为空字符串 | 正常传入 LLM，评分结果取决于模型对空文本的理解 |
| 问题为空字符串 | 同上，由 LLM 自行判断 |
| 并发调用 | 当前单实例无状态，但 `APIProcessor` 内部可能受限于上游 QPS，调用方自行控制并发 |

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始 Spec，反向推导自 `src/post_retrieval_correction.py` 当前实现 |
