# Amendment: 根据生成图重检索信号编排外循环

- Date: 2026-06-22
- Status: merged
- Target: spec/rag_app_spec.md
- Merged at: 2026-06-22

## 变更摘要

`rag_app` 在「检索 → 生成」单次串联之外，增加外层循环：当生成图返回 `should_retry_retrieval=True` 时，使用（可能已改写的）工作问题重新调用检索图，再调用生成图。保留用户原始问题用于返回；限制外层轮次，避免与检索图/生成图内层重试叠乘导致成本失控。

## 相对主 spec 的修改

### 修改前（引用主 spec 章节或简述当前行为）

- **§3 架构**：线性 `retrieval_graph` → `generage_graph`，无回流。
- **§5.2 / §5.3**：检索与生成均使用入参 `question`；生成图改写的问题不回传检索。
- **§5.4 返回结构**：无 `should_retry_retrieval`、`failure_reason`、`rag_rounds` 等字段。
- 生成失败时仅返回最后一次 `generation` 与 `is_grounded_in_docs` / `is_question_answered`，不触发重检索。

### 修改后（期望行为）

#### 5.1 新增参数

```python
def run(
    self,
    question: str,
    company_code: str = "001",
    is_direct_retrieve: bool = False,
    is_direct_generate: bool = False,
    max_rag_rounds: int = 2,
) -> dict
```

| 参数 | 说明 |
|------|------|
| `max_rag_rounds` | 外层「检索+生成」最大轮数，默认 `2`（即最多 2 次完整检索阶段 + 2 次完整生成阶段） |

`run_rag(...)` 同步增加 `max_rag_rounds` 参数并透传。

#### 5.2 问题字段约定

| 变量 | 说明 |
|------|------|
| `original_question` | 用户入参，全程不变，写入返回 `question` |
| `working_question` | 当前轮检索/生成使用的问题；首轮等于 `original_question`；若生成图改写则更新为 `generation_state["question"]` |

#### 5.3 外层循环流程

```
rag_round = 0
working_question = original_question
last_retrieval_state = None
last_generation_state = None

while rag_round < max_rag_rounds:
    rag_round += 1

    # 检索阶段（每轮独立 invoke，内层状态重置）
    retrieval_inputs = {
        "question": working_question,
        "company_code": company_code,
        "documents": [],
        "retrieval_attempts": 0,
        "has_relevant_docs": False,
        "is_direct_retrieve": is_direct_retrieve,
    }
    retrieval_state = retrieval_app.invoke(retrieval_inputs)
    last_retrieval_state = retrieval_state

    # 生成阶段（每轮 generation_attempts 从 0 开始）
    generation_inputs = {
        "question": working_question,
        "documents": retrieval_state["documents"],
        "generation": "",
        "generation_attempts": 0,
        "is_grounded_in_docs": False,
        "is_question_answered": False,
        "is_direct_generate": is_direct_generate,
        "should_retry_retrieval": False,
        "failure_reason": "skipped",
    }
    generation_state = generation_app.invoke(generation_inputs)
    last_generation_state = generation_state

    if not generation_state.get("should_retry_retrieval", False):
        break

    working_question = generation_state.get("question", working_question)

return assemble_result(...)
```

要点：

1. **每轮重置** `retrieval_attempts=0`、`generation_attempts=0`；检索图内层仍可自行 `transform_query → retrieve`（最多约 2 次检索）。
2. **跳出条件**：`should_retry_retrieval=False` 或已达 `max_rag_rounds`。
3. **重试前**用生成图输出的 `question` 更新 `working_question`（改写查询用于下一轮检索）。

#### 5.4 与直接模式的关系

| 模式 | 外层循环 |
|------|----------|
| `is_direct_generate=True` | 生成图恒 `should_retry_retrieval=False`，**只跑 1 轮** |
| `is_direct_retrieve=True` | 允许外层循环；检索图跳过文档评分，但可被多次 invoke |
| 两者均为 True | 只跑 1 轮（与现行为一致） |

#### 5.5 返回结构增补（§5.4）

```python
{
    "question": str,              # original_question
    "working_question": str,        # 最后一轮使用的工作问题（可能已改写）
    "documents": List[Document],
    "generation": str,
    "has_relevant_docs": bool,
    "is_direct_retrieve": bool,
    "is_grounded_in_docs": bool,
    "is_question_answered": bool,
    "is_direct_generate": bool,
    "should_retry_retrieval": bool,   # 最后一轮生成图出口值
    "failure_reason": str,            # 最后一轮：ok | hallucination | not_answered | skipped
    "rag_rounds": int,                # 实际执行的外层轮数（1..max_rag_rounds）
}
```

最后一轮 `retrieval_state` / `generation_state` 的文档与生成结果作为最终 `documents` / `generation`。

#### 5.6 异常与警告（§6）

| 场景 | 行为 |
|------|------|
| 最后一轮 `should_retry_retrieval=True` 且已达 `max_rag_rounds` | 不抛异常；返回当前 `generation`；`should_retry_retrieval` 仍为 `True`，日志打印警告 |
| 某轮检索结果为空 | 与现有一致：警告后继续生成；由生成图决定是否 `should_retry_retrieval` |
| `max_rag_rounds < 1` | 抛出 `ValueError` |

#### 5.7 架构图更新（§3）

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

#### 5.8 成本上限（说明性，非硬编码 token 限制）

单次 `run()` 在默认参数下近似上限：

| 层级 | 默认上限 |
|------|----------|
| 外层 `max_rag_rounds` | 2 |
| 检索图内层 | 每轮约 2 次检索 + 若干文档评分 LLM |
| 生成图内层 | 每轮约 2 次生成 + 评分 + 改写（见 generage_graph amendment） |

实现时不在 `rag_app` 内再叠一层无上限循环。

## 实现备注

- 涉及文件：`src/rag_app.py`
- 依赖生成图 amendment：`spec/amendments/pending/generage_graph/2026-06-22_01_retry-retrieval-signal.md`
- `RAGApp.run` 由单次 invoke 改为 while 循环；抽取 `_run_retrieval` / `_run_generation` 或内联保持可读即可。
- 单元测试：mock 子图，验证 (1) 成功首轮跳出 (2) 重检索信号触发第二轮 (3) `max_rag_rounds` 截断 (4) direct 模式只一轮。

## 合并检查清单

- [x] 主 spec 已更新
- [x] 测试已更新/通过
- [x] 本文件已移至 `merged/rag_app/` 且 Status 为 merged
