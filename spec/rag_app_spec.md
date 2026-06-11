# RAG App Spec

> **同步声明**：本文档描述 `src/rag_app.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`rag_app.py` 是项目的**最终 RAG 应用入口**，负责将检索工作流与生成工作流串联：

1. 调用 `src.retrieval_graph.app` 根据问题与公司名检索相关文档；
2. 调用 `src.generage_graph.app` 基于检索文档生成带自检的答案。

本模块本身不实现具体的检索或生成算法，而是作为两个子图的编排层。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 职责清晰 | 仅做流程编排，不重复实现检索/生成逻辑 |
| 可替换 | 通过构造函数注入 retrieval_app 与 generation_app，便于测试与替换 |
| 错误明确 | 输入校验与阶段异常均抛出明确异常，不静默吞掉 |
| 易用 | 提供 `RAGApp` 类与便捷函数 `run_rag` 两种使用方式 |

---

## 3. 架构设计

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  用户调用    │────→│  src.retrieval_graph │────→│  src.generage_graph  │
│ run_rag()   │     │      (检索)          │     │      (生成)          │
└─────────────┘     └─────────────────────┘     └─────────────────────┘
                                                          │
                                                          ▼
                                                  返回答案 + 文档
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
    def run(self, question: str, company_name: str = "") -> dict
```

| 方法 | 说明 |
|------|------|
| `__init__` | 注入检索图与生成图，默认使用项目实现的两个子图 |
| `run` | 执行完整 RAG 流程，返回最终状态字典 |

### 4.2 run_rag

```python
def run_rag(question: str, company_name: str = "") -> dict
```

- 使用默认 `RAGApp` 实例执行查询
- 适合简单脚本或命令行调用

---

## 5. 执行流程

### 5.1 输入校验

- `question` 为空或仅包含空白字符时抛出 `ValueError`

### 5.2 检索阶段

构建 `retrieval_graph` 初始状态：

```python
{
    "question": question,
    "company_name": company_name,
    "documents": [],
    "retrieval_attempts": 0,
    "has_relevant_docs": False,
}
```

调用 `retrieval_app.invoke(...)` 获取检索结果。

### 5.3 生成阶段

将检索结果中的 `documents` 传入 `generage_graph`：

```python
{
    "question": question,
    "documents": documents,
    "generation": "",
    "generation_attempts": 0,
}
```

调用 `generation_app.invoke(...)` 获取最终答案。

### 5.4 返回结构

```python
{
    "question": str,
    "documents": List[Document],
    "generation": str,
}
```

---

## 6. 异常处理

| 场景 | 行为 |
|------|------|
| `question` 为空 | 抛出 `ValueError` |
| 检索阶段异常 | 抛出 `RuntimeError("检索阶段执行失败: ...")` |
| 生成阶段异常 | 抛出 `RuntimeError("生成阶段执行失败: ...")` |
| 未检索到文档 | 打印警告，继续基于空上下文生成 |

---

## 7. 依赖清单

**内部依赖**：
- `src.retrieval_graph.app`
- `src.generage_graph.app`
- `langchain_core.documents.Document`

---

## 8. 使用示例

### 8.1 使用便捷函数

```python
from src.rag_app import run_rag

result = run_rag(
    question="中芯国际2024年营业收入是多少？",
    company_name="中芯国际集成电路制造有限公司",
)
print(result["generation"])
```

### 8.2 使用类实例（便于测试替换子图）

```python
from src.rag_app import RAGApp

app = RAGApp()
result = app.run(
    question="中芯国际2024年营业收入是多少？",
    company_name="中芯国际集成电路制造有限公司",
)
print(result["generation"])
```

---

## 9. 边界行为与注意事项

| 场景 | 行为 |
|------|------|
| `company_name` 为空 | 检索器不执行公司过滤，按全局检索 |
| 检索结果为空 | 继续调用生成图，生成图基于空上下文生成 |
| 生成图为 None 或缺少 generation | 返回空字符串 |
| 子图被替换为 mock | `RAGApp` 仍按相同状态约定调用 |

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-06-11 | 初始版本；串联 retrieval_graph 与 generage_graph，提供 RAGApp 与 run_rag 接口 |
