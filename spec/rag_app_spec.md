# RAG App Spec

> **同步声明**：本文档描述 `src/rag_app.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`rag_app.py` 是项目的**最终 RAG 应用入口**，负责将检索工作流与生成工作流串联：

1. 调用 `src.retrieval_graph.app` 根据问题与公司编码检索相关文档；
2. 调用 `src.generage_graph.app` 基于检索文档生成带自检的答案；
3. 若生成图返回 `should_retry_retrieval=True`，使用改写后的问题重新检索并生成（受 `max_rag_rounds` 限制）。

本模块本身不实现具体的检索或生成算法，而是作为两个子图的编排层。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 职责清晰 | 仅做流程编排，不重复实现检索/生成逻辑 |
| 可替换 | 通过构造函数注入 retrieval_app 与 generation_app，便于测试与替换 |
| 错误明确 | 输入校验与阶段异常均抛出明确异常，不静默吞掉 |
| 外层重检索 | 根据生成图 `should_retry_retrieval` 编排多轮检索+生成 |
| 易用 | 提供 `RAGApp` 类与便捷函数 `run_rag` 两种使用方式 |

---

## 3. 架构设计

```
                    ┌──────────────────────────────────┐
                    │           RAGApp.run             │
                    │  loop rag_round < max_rag_rounds │
                    └───────────────┬──────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           │
     retrieval_graph (working_question)                    │
              │                                           │
              ▼                                           │
     generage_graph                                       │
              │                                           │
              ▼                                           │
     should_retry_retrieval? ──True──► 更新 working_question
              │                                           │
              False                                       │
              ▼                                           │
            return ◄──────────────────────────────────────┘
```

---

## 4. 类与函数

### 4.1 RAGApp

```python
class RAGApp:
    def __init__(
        self,
        retrieval_app=retrieval_app,
        generation_app=generation_app,
    )
    def run(
        self,
        question: str,
        company_code: str = "001",
        is_direct_retrieve: bool = False,
        is_direct_generate: bool = False,
        max_rag_rounds: int = 2,
    ) -> dict
```

| 方法 | 说明 |
|------|------|
| `__init__` | 注入检索图与生成图，默认使用项目实现的两个子图 |
| `run` | 执行完整 RAG 流程（可多轮），返回最终状态字典 |

### 4.2 run_rag

```python
def run_rag(
    question: str,
    company_code: str = "001",
    is_direct_retrieve: bool = False,
    is_direct_generate: bool = False,
    max_rag_rounds: int = 2,
) -> dict
```

- 使用默认 `RAGApp` 实例执行查询
- 适合简单脚本或命令行调用

---

## 5. 执行流程

### 5.1 输入校验

- `question` 为空或仅包含空白字符时抛出 `ValueError`
- `max_rag_rounds < 1` 时抛出 `ValueError`

### 5.2 问题字段约定

| 变量 | 说明 |
|------|------|
| `original_question` | 用户入参，全程不变，写入返回 `question` |
| `working_question` | 当前轮检索/生成使用的问题；首轮等于 `original_question`；生成图改写后更新 |

### 5.3 外层循环

每轮执行：

1. **检索阶段**（`retrieval_attempts` 每轮从 0 开始）

```python
{
    "question": working_question,
    "company_code": company_code,
    "documents": [],
    "retrieval_attempts": 0,
    "has_relevant_docs": False,
    "is_direct_retrieve": is_direct_retrieve,
}
```

2. **生成阶段**（`generation_attempts` 每轮从 0 开始）

```python
{
    "question": working_question,
    "documents": documents,
    "generation": "",
    "generation_attempts": 0,
    "is_grounded_in_docs": False,
    "is_question_answered": False,
    "is_direct_generate": is_direct_generate,
    "should_retry_retrieval": False,
    "failure_reason": "skipped",
}
```

3. 若 `should_retry_retrieval=False`，结束循环；否则 `working_question = generation_state["question"]`，进入下一轮（直至 `max_rag_rounds`）。

### 5.4 返回结构

```python
{
    "question": str,
    "working_question": str,
    "documents": List[Document],
    "generation": str,
    "has_relevant_docs": bool,
    "is_direct_retrieve": bool,
    "is_grounded_in_docs": bool,
    "is_question_answered": bool,
    "is_direct_generate": bool,
    "should_retry_retrieval": bool,
    "failure_reason": str,
    "rag_rounds": int,
}
```

---

## 6. 异常处理

| 场景 | 行为 |
|------|------|
| `question` 为空 | 抛出 `ValueError` |
| `max_rag_rounds < 1` | 抛出 `ValueError` |
| 检索阶段异常 | 抛出 `RuntimeError("检索阶段执行失败: ...")` |
| 生成阶段异常 | 抛出 `RuntimeError("生成阶段执行失败: ...")` |
| 未检索到文档 | 打印警告，继续基于空上下文生成 |
| 已达 `max_rag_rounds` 且仍 `should_retry_retrieval=True` | 打印警告，返回当前轮结果 |

---

## 7. 依赖清单

**内部依赖**：
- `src.retrieval_graph.app`
- `src.generage_graph.app`
- `langchain_core.documents.Document`

---

## 8. 使用示例

```python
from src.rag_app import run_rag

result = run_rag(
    question="工程总投资是多少？",
    company_code="001",
    max_rag_rounds=2,
)
print(result["generation"])
print(result["rag_rounds"])
```

---

## 9. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| `company_code` 为空 | 检索器不执行公司过滤，按全局检索 |
| `is_direct_generate=True` | 生成图不触发重检索，外层只跑 1 轮 |
| `is_direct_retrieve=True` | 允许外层多轮；检索图跳过文档评分 |
| 子图被替换为 mock | `RAGApp` 仍按相同状态约定调用 |

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始版本 |
| v1.1 | 2026-06-12 | `company_code` 入参 |
| v1.2 | 2026-06-12 | 返回 `is_grounded_in_docs`、`is_question_answered` |
| v1.3 | 2026-06-12 | `is_direct_generate` |
| v1.4 | 2026-06-12 | `is_direct_retrieve` |
| v1.5 | 2026-06-22 | 外层重检索循环、`max_rag_rounds`、新增返回字段 |
