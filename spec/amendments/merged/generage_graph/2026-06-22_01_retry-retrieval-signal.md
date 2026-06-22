# Amendment: 生成失败时输出重检索信号

- Date: 2026-06-22
- Status: merged
- Target: spec/generage_graph_spec.md
- Merged at: 2026-06-22

## 变更摘要

生成图在质量评估结束后，除现有 `is_grounded_in_docs` / `is_question_answered` 外，新增 `should_retry_retrieval` 与 `failure_reason`，向编排层（`rag_app`）表明「当前文档上下文下无法产出合格答案，建议重新检索」。生成图仍不执行检索；内层重试策略收紧，避免同文档空转浪费 LLM。

## 相对主 spec 的修改

### 修改前（引用主 spec 章节或简述当前行为）

- **§1 概述 / §2 设计目标**：失败时在图内 `transform_query` 后回到 `generate`，最多 3 次生成；不改文档、不触发检索。
- **§4 GraphState**：无 `should_retry_retrieval`、`failure_reason`。
- **§6.2 route_after_grade**：
  - 未基于文档且 `generation_attempts < 2` → `not supported` → 同问题同文档再 `generate`。
  - 未基于文档且 `generation_attempts >= 2` → `not useful` → `transform_query`。
  - 已基于文档但未回答问题 → `not useful` → `transform_query`。
- **§10 边界**：达到 3 次生成后结束，返回最后一次 `generation` 与评分标识；无上游重检索约定。

### 修改后（期望行为）

#### 4.1 新增状态字段

```python
class GraphState(TypedDict):
  # ... 现有字段 ...
  should_retry_retrieval: bool
  failure_reason: str  # "ok" | "hallucination" | "not_answered" | "skipped"
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `should_retry_retrieval` | `bool` | 工作流结束时是否建议上游重新检索 |
| `failure_reason` | `str` | 失败类型；成功为 `ok`；直接生成未评估为 `skipped` |

初始输入可省略上述字段，默认 `should_retry_retrieval=False`、`failure_reason="skipped"`（与 `is_direct_generate` 一致时）或 `failure_reason="ok"`。

#### 4.2 出口标志计算（新增节点 `finalize` 或等价逻辑）

在工作流**结束前**统一写入 `should_retry_retrieval` 与 `failure_reason`（不分散在条件边里隐式表达）。

| 条件 | `failure_reason` | `should_retry_retrieval` |
|------|------------------|---------------------------|
| `is_direct_generate=True` | `skipped` | `False` |
| grounded 且 answered | `ok` | `False` |
| 未 grounded（幻觉） | `hallucination` | `True` |
| grounded 但未 answered | `not_answered` | `True` |

说明：

- **不准确**（未基于文档）→ `hallucination`，建议重检索。
- **不相关**（未回答问题）→ `not_answered`，建议重检索（改写问题后由 `rag_app` 带入检索图，对齐 Self-RAG 的 `transform_query → retrieve`）。
- 成功路径不触发重检索。

#### 4.3 内层路由收紧（§6.2 / §6.3）

目标：减少「同文档、同问题」无效重试；把「换文档」交给 `rag_app`。

**`route_after_grade` 新规则**：

1. grounded 且 answered → `useful`
2. 未 grounded → `not useful`（**不再**走 `not supported` 同输入再生成）
3. grounded 但未 answered → `not useful`

**`should_continue` 新规则**（仅 `not useful` 经 `transform_query` 后）：

- `generation_attempts < 2` → `generate`（同文档内最多 **2 次**生成：首次 + 改写问题后 1 次）
- `generation_attempts >= 2` → `end`（经 `finalize` 写出 `should_retry_retrieval=True`）

移除 `not supported → generate` 边；`route_after_grade` 仅返回 `useful` / `not useful`。

**更新后架构（简图）**：

```
START → generate → [direct? END | grade_generation]
grade_generation → route_after_grade
  useful → finalize → END
  not useful → transform_query → should_continue
    attempts < 2 → generate
    attempts >= 2 → finalize → END
direct 路径：generate → finalize（skipped）→ END
```

#### 4.4 设计目标表增补（§2）

| 目标 | 说明 |
|------|------|
| 重检索信号 | 失败时输出 `should_retry_retrieval`，由 `rag_app` 编排重检索，本图不调用检索器 |
| 内层有限自愈 | 同文档内仅允许「改写问题后再生成 1 次」，不做同输入盲重试 |

#### 4.5 边界行为增补（§10）

| 场景 | 行为 |
|------|------|
| 内层 2 次生成仍失败 | `should_retry_retrieval=True`，`failure_reason` 为 `hallucination` 或 `not_answered` |
| `is_direct_generate=True` | `should_retry_retrieval=False`，`failure_reason=skipped` |
| `documents` 为空且评估失败 | 同失败规则；通常 `hallucination`，建议上游重检索或接受无答案 |

#### 4.6 LLM 调用上限（正常模式）

单次 `generage_graph.invoke`（不含 `is_direct_generate`）：

| 场景 | 上限 |
|------|------|
| 一次通过 | 1 生成 + 2 评分 = **3** |
| 内层用尽仍失败 | 2 生成 + 2～3 评分 + 1 改写 = **5～6** |

（较当前最多约 12 次下降。）

## 实现备注

- 涉及文件：`src/generage_graph.py`
- 新增 `finalize(state) -> dict` 节点，所有结束路径经 `finalize` 再 `END`；`route_after_generate` 的 `end` 也指向 `finalize` 而非直接 `END`。
- `QuestionRewriter` 保留：改写后的问题写入 `state["question"]`，供 `rag_app` 下一轮检索使用。
- **不**在本模块引入 `HybridRetriever` 或 `retrieval_graph` 依赖。
- 与 `rag_app` amendment `2026-06-22_01_outer-retrieval-loop.md` 配套实现。

## 合并检查清单

- [x] 主 spec 已更新
- [x] 测试已更新/通过
- [x] 本文件已移至 `merged/generage_graph/` 且 Status 为 merged
