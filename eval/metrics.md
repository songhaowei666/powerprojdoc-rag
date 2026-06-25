# RAG 评估指标说明

本文档说明 [`eval/evaluation.py`](evaluation.py) 离线评估使用的全部指标：1 个自定义页面指标 + 5 个 ragas 标准指标。

- 分数范围：0.0 ~ 1.0（`page_recall@k` 在无法评估时为 `None`）
- 越高越好
- 规格文档：[`spec/evaluation_spec.md`](../spec/evaluation_spec.md)

---

## 1. 输入字段映射

评估时 ragas 使用的字段与项目评估集的对应关系：

| ragas 字段 | 评估集 / RAG 输出 | 说明 |
|------------|-------------------|------|
| `user_input` | `question` | 用户问题 |
| `response` | `generation` | RAG 生成答案 |
| `retrieved_contexts` | `documents[].page_content` | 检索到的文本块列表 |
| `reference` | `expected_answer` | 预期标准答案 |
| （自定义） | `expected_source_pages` | 预期来源页码，仅用于 `page_recall@k` |

---

## 2. 指标总览

| 指标 | 维度 | 依赖 LLM | 依赖 Embedding | 需要 `expected_answer` |
|------|------|:--------:|:--------------:|:---------------------:|
| `page_recall@k` | 检索（页码） | 否 | 否 | 否（需 `expected_source_pages`） |
| `context_precision` | 检索 | 是 | 否 | 是 |
| `context_recall` | 检索 | 是 | 否 | 是 |
| `faithfulness` | 生成 | 是 | 否 | 否 |
| `answer_relevancy` | 生成 | 是 | 是 | 否 |
| `answer_correctness` | 生成 | 是 | 是 | 是 |

评估模型默认读取 `settings.chat_model`；Embedding 默认读取 `settings.embedding_model`（与 RAG 共用 API Key / Base URL）。

---

## 3. 自定义指标

### 3.1 `page_recall@k`（页面召回率）

**衡量什么**：检索结果是否命中了标注的预期页码（本项目「按页检索」场景专用）。

**公式**：

```
page_recall@k = |{page ∈ expected_pages 且 page 出现在前 k 个结果的 metadata.page}| / |expected_pages|
```

**计算规则**：

- 取前 `k` 个检索结果（默认 `k=6`）
- 读取每个 `Document.metadata["page"]`，在 `expected_source_pages` 中则计为命中
- 同一页码多次出现只计一次

**边界行为**：

| 情况 | 返回值 |
|------|--------|
| `expected_source_pages` 为空 | `None`（不参与评估） |
| `documents` 为空 | `0.0` |
| 缺少 `metadata.page` | 该文档视为未命中 |

**与 ragas 指标的区别**：ragas 看「内容是否相关」；`page_recall@k` 看「页码是否命中标注」。可能出现内容相关但页码召回为 0（评估集页码标注错误，或检索到了同内容的不同页）。

**实现**：[`compute_page_recall_at_k()`](evaluation.py)

---

## 4. ragas 标准指标

以下指标由 [ragas](https://docs.ragas.io/) 计算，评判 LLM 默认 `temperature=0`。实现位于 `ragas.metrics._*` 模块。

### 4.1 `context_precision`（上下文精确率）

**衡量什么**：检索到的上下文里，有多少是对「写出预期答案」有用的（检索噪声大则偏低）。

**依赖字段**：`user_input`、`retrieved_contexts`、`reference`

**计算步骤**：

1. 对**每个**检索 chunk，LLM 判断：给定「问题 + 该 chunk + 预期答案」，该 chunk 是否有助于得到预期答案 → 0/1
2. 按检索顺序计算 **Average Precision（AP）**，排名靠前的相关 chunk 权重更高

```
AP = Σ (前 i 个结果中相关数 / i × 第 i 个是否相关) / 相关 chunk 总数
```

**典型解读**：

- 接近 1.0：检索结果几乎都有用
- 0.5 ~ 0.8：大部分有用，但混入了无关 chunk

---

### 4.2 `context_recall`（上下文召回率）

**衡量什么**：预期答案中的信息，有多少能在检索上下文中找到依据。

**依赖字段**：`user_input`、`retrieved_contexts`、`reference`

**计算步骤**：

1. LLM 将 `reference`（预期答案）拆成若干陈述句
2. 对每句判断：能否从 `retrieved_contexts` 中找到支持 → attributed 0/1
3. 得分 = 被覆盖句数 / 总句数

```
context_recall = 可归因句数 / 预期答案陈述句总数
```

**典型解读**：

- 1.0：预期答案的要点都被检索上下文覆盖
- 低于 1.0：有预期信息未被检索到

---

### 4.3 `faithfulness`（忠实度）

**衡量什么**：生成答案是否存在幻觉——答案中的陈述是否都能从检索上下文直接推断。

**依赖字段**：`user_input`、`response`、`retrieved_contexts`

**计算步骤**：

1. LLM 将 `response`（生成答案）拆成若干原子陈述
2. 对每句做 NLI 判断：能否从 `retrieved_contexts` 直接推出 → verdict 0/1
3. 得分 = 有依据句数 / 总句数

```
faithfulness = 可推断句数 / 生成答案陈述句总数
```

**典型解读**：

- 1.0：生成内容完全基于检索文档，无编造
- 低于 1.0：存在无法从上下文推出的内容

---

### 4.4 `answer_relevancy`（答案相关性）

**衡量什么**：生成答案是否在回答问题，而非答非所问、敷衍或回避。

**依赖字段**：`user_input`、`response`  
**额外依赖**：Embedding 模型

**计算步骤**：

1. LLM 根据 `response` **反推** 3 个问题（`strictness=3`）
2. 判断答案是否为 noncommittal（如「我不知道」→ 整项为 0）
3. 用 Embedding 计算**原始问题**与**各反推问题**的余弦相似度，取平均

```
answer_relevancy = mean(cos_sim(原问题, 反推问题_i)) × (非敷衍 ? 1 : 0)
```

**典型解读**：

- 接近 1.0：答案紧扣问题
- 0.7 ~ 0.9：在回答问题，但表述角度或详略与原问题有偏差

---

### 4.5 `answer_correctness`（答案正确性）

**衡量什么**：生成答案与预期标准答案在事实上和语义上有多接近。

**依赖字段**：`user_input`、`response`、`reference`  
**额外依赖**：Embedding 模型

**计算步骤**（默认权重 **75% 事实 + 25% 语义**）：

**A. 事实 F1（权重 0.75）**

1. LLM 分别将 `response` 与 `reference` 拆成陈述句
2. LLM 对生成答案每句分类：TP（与预期一致）/ FP（多余或错误）/ FN（预期有但未答）
3. 计算 F1 分数

**B. 语义相似度（权重 0.25）**

- 对整段 `response` 与 `reference` 做 Embedding 余弦相似度

```
answer_correctness = 0.75 × F1(事实) + 0.25 × cos_sim(生成, 预期)
```

**典型解读**：

- 接近 1.0：与预期答案高度一致
- 0.7 ~ 0.8：意思正确，但措辞或细节与预期答案有差异（更详或更略）

---

## 5. 指标关系

```
                    ┌─────────────────────────────────────┐
                    │           评估输入                   │
                    │  question / generation / contexts   │
                    │  expected_answer / expected_pages   │
                    └─────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
   │  检索质量     │           │  生成质量     │           │  页码命中     │
   ├──────────────┤           ├──────────────┤           ├──────────────┤
   │context_      │           │ faithfulness │           │page_recall@k │
   │  precision   │           │answer_       │           │              │
   │context_recall│           │  relevancy   │           │              │
   └──────────────┘           │answer_       │           └──────────────┘
                              │  correctness │
                              └──────────────┘
```

**常见组合解读**：

| 模式 | 可能含义 |
|------|----------|
| `context_recall` 高 + `page_recall@k` 低 | 内容找对了，但页码标注或命中不对 |
| `faithfulness` 高 + `answer_correctness` 低 | 无幻觉，但与预期答案表述不一致 |
| `context_precision` 低 + `context_recall` 高 | 检索到了所需信息，但混入了较多无关 chunk |

---

## 6. 报告中的低分阈值

[`eval/report.py`](report.py) 生成 Markdown 报告时，任一指标 **< 0.5** 或 **NaN** 的样本会列入「低分样本」章节。

---

## 7. 参考

- 项目规格：[`spec/evaluation_spec.md`](../spec/evaluation_spec.md)
- ragas 官方文档：https://docs.ragas.io/en/stable/concepts/metrics/
- ragas 源码（deepstudy 环境）：`ragas/metrics/_context_precision.py` 等
