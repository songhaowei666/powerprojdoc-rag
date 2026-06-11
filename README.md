# PowerProjDoc-RAG

一个面向长文档（财务报告、基建项目文档等）的**检索增强生成（RAG）系统**。系统支持从 PDF 解析、文本分块、向量化存储到智能检索与答案生成的完整流水线，并集成了查询重写、多角度检索、LLM 重排、父文档回溯、引用校验等多种优化策略，以提升复杂文档问答的准确性与可解释性。

---

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF 解析   │ --> │  文本分块   │ --> │  信息嵌入   │ --> │  向量存储   │
│ (MinerU)    │     │(Markdown/  │     │(OpenAI     │     │(ChromaDB + │
│             │     │  JSON)      │     │ Embedding) │     │  BM25)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                              ┌─────────────────────────────────────┘
                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  答案生成   │ <-- │ 检索后处理  │ <-- │  混合检索   │ <-- │ 检索前处理  │
│(LLM + 引用  │     │(重排 + 校正)│     │(向量+BM25) │     │(重写+路由) │
│  校验)      │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 核心功能模块

### 1. 数据导入（Ingestion）

- **PDF 解析**：调用 [MinerU](https://mineru.net/) API 将 PDF 转换为结构化 Markdown / JSON，保留原始排版与表格信息。
- **报告合并**：将解析结果按页索引合并为规整的 Markdown 文档，便于后续分块处理。

> 相关代码：`src/pdf_mineru.py`、`src/markdown_reports_merging.py`

### 2. 文本分块（Chunking）

- 支持 **按页分块** 与 **Markdown 行级分块**两种模式。
- 使用 `RecursiveCharacterTextSplitter`（基于 `tiktoken`）控制 chunk_size 与 chunk_overlap。
- 可选插入**序列化表格块**，保证表格信息不丢失。
- 自动关联 `company_name`、`sha1`、`file_name` 等元数据。

> 相关代码：`src/text_splitter.py`

### 3. 信息嵌入与向量存储（Embedding & Vector Store）

- 基于 **OpenAI Embedding API**（默认 `text-embedding-3-large`）生成文本向量。
- 使用 **ChromaDB** 作为向量数据库，持久化存储，支持按 `company_name` 等元数据过滤检索。
- 同时构建 **BM25 索引**（基于 `jieba` 中文分词），实现稀疏检索能力。

> 相关代码：`src/openai_embedding.py`、`src/ingestion.py`

### 4. 检索前处理（Pre-Retrieval）

- **元数据过滤**：利用 LLM 从用户查询中抽取省公司编码（`unit_code`）等过滤条件，生成 ChromaDB `where` 子句，缩小检索范围。
- **多角度查询扩展**：基于原始查询生成 3 个检索友好变体（语义扩展 / 关键词聚焦 / 结构化条件），提升召回率。
- **查询重写**：通过 LLM 重写用户问题，使其更适合文档检索。

> 相关代码：`src/pre_retrieval_processing.py`、`src/retrieval_graph.py`

### 5. 索引优化（Index Optimization）

- **父文档检索（Parent Document Retrieval）**：检索时使用子文本块进行相似度匹配，但返回该子块所属的**整页内容**作为上下文，避免信息碎片化，提升答案完整性。

> 相关代码：`src/retrieval.py`（`return_parent_pages` 参数）

### 6. 检索与检索后处理（Retrieval & Post-Retrieval）

- **混合检索（Hybrid Retrieval）**：
  - `VectorRetriever`：基于 ChromaDB 的语义相似度检索，支持公司名过滤。
  - `BM25Retriever`：基于关键词的稀疏检索。
  - `HybridRetriever`：向量检索 + **LLM 重排**，融合向量相似度与 LLM 相关性评分。
- **重排策略**：
  - **Jina Reranker**：调用 Jina API 进行多语言重排。
  - **LLM Reranker**：使用 GPT-4o-mini / Qwen-Turbo 对检索结果进行单条或批量相关性评分，按加权融合分数重新排序。
- **检索后校正**：通过 LLM 对检索文档与问题进行二元相关性评分（`yes/no`），过滤低质量上下文，降低模型幻觉。

> 相关代码：`src/retrieval.py`、`src/reranking.py`、`src/post_retrieval_correction.py`

### 7. 答案生成（Generation）

- 支持 **单公司问答** 与 **多公司比较问答**（自动重写为单公司子问题并行处理）。
- 基于结构化 Prompt 调用 LLM（OpenAI GPT-4o / Qwen-Turbo）生成答案，输出包含：
  - `final_answer`：最终答案
  - `step_by_step_analysis`：逐步分析过程
  - `relevant_pages`：相关页码
- **引用校验**：自动验证 LLM 引用的页码是否真实存在于检索结果中，过滤幻觉引用，并补充必要的页码。
- 支持 **全上下文模式**（`full_context`）：直接加载整篇报告作为上下文，适用于高精度场景。

> 相关代码：`src/questions_processing.py`、`src/api_requests.py`

### 8. 检索工作流（Retrieval Graph）

基于 **LangGraph** 实现的状态机工作流：

```
[开始] --> [检索] --> [文档相关性评分]
                              │
              ┌───────────────┘
              ▼
    [有相关文档?] --是--> [结束]
              │
              否（第1次）
              ▼
        [查询重写] --> [重新检索]
                              │
              ┌───────────────┘（第2次仍无相关文档则直接返回）
```

> 相关代码：`src/retrieval_graph.py`

---

## 快速开始

### 环境配置

1. 复制环境变量模板并填写你的 API Key：

```bash
cp .env_example .env
```

2. 编辑 `.env`：

```env
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1   # 如需自定义 base url

# 可选
JINA_API_KEY=...
MINERU_API_KEY=...
DASHSCOPE_API_KEY=...

# 数据目录（可选，留空使用默认值）
CHROMA_PERSIST_DIR=data/stock_data/databases/vector_dbs
BM25_OUTPUT_DIR=data/stock_data/databases/bm25_index
REPORTS_INPUT_DIR=data/stock_data/databases/chunked_reports
```

3. 安装依赖（请根据项目实际依赖安装）：

```bash
pip install -r requirements.txt
```

### 运行完整流水线

```python
from pyprojroot import here
from src.pipeline import Pipeline, RunConfig

# 配置运行参数
config = RunConfig(
    parent_document_retrieval=True,   # 启用父文档检索
    llm_reranking=True,               # 启用 LLM 重排
    parallel_requests=4,
    answering_model="gpt-4o-2024-08-06",
    pipeline_details="PDF解析 + 向量库 + 查询路由 + 父文档检索 + LLM重排 + CoT"
)

# 初始化流水线
pipeline = Pipeline(
    root_path=here() / "data" / "stock_data",
    run_config=config
)

# 步骤1：PDF -> Markdown（按需执行）
# pipeline.export_reports_to_markdown("report.pdf")

# 步骤2：文本分块
pipeline.chunk_reports()

# 步骤3：构建向量库
pipeline.create_vector_dbs()

# 步骤4（可选）：构建 BM25 索引
pipeline.create_bm25_db()

# 步骤5：批量处理问题并生成答案
pipeline.process_questions()

# 单条问题即时推理
answer = pipeline.answer_single_question(
    question="中芯国际2024年营业收入是多少？",
    kind="number"
)
print(answer)
```

---

## 项目结构

```
.
├── data/                       # 数据目录
│   └── stock_data/
│       ├── debug_data/         # 调试中间产物（解析结果、合并报告、Markdown）
│       └── databases/          # 向量库、分块报告、BM25 索引
├── spec/                       # 规格文档
├── src/                        # 核心源码
│   ├── pdf_mineru.py           # PDF 解析（MinerU API）
│   ├── text_splitter.py        # 文本分块
│   ├── openai_embedding.py     # OpenAI 嵌入封装
│   ├── ingestion.py            # ChromaDB / BM25 索引构建
│   ├── pre_retrieval_processing.py  # 检索前处理（过滤、多角度查询）
│   ├── retrieval.py            # 检索器（向量 / BM25 / 混合）
│   ├── reranking.py            # 重排器（Jina / LLM）
│   ├── post_retrieval_correction.py # 检索后校正
│   ├── retrieval_graph.py      # LangGraph 检索工作流
│   ├── questions_processing.py # 问题处理与答案生成
│   ├── pipeline.py             # 主流程编排
│   └── config.py               # 全局配置（.env）
├── tests/                      # 测试用例
├── examples/                   # 使用示例
└── README.md                   # 本文件
```

---

## 技术亮点（简历适用）

### 生成器（Generator）

- 实现**多公司比较问题的自动拆分与并行推理**：将含 N 家公司的复杂查询拆分为单公司子问题，并行调用 LLM 后汇总生成对比结论，避免长上下文干扰。
- 构建**引用校验层**：反验 LLM 给出的页码引用，自动剔除幻觉页码，并补充 Top 检索结果中的高相关页码，将引用数控制在 **2~8 页**，保证答案可追溯。

### 检索器（Retriever）

- 构建**多路混合检索架构**：Dense Retrieval（ChromaDB + OpenAI Embedding，`text-embedding-3-large`）与 Sparse Retrieval（BM25 + jieba 中文分词）相结合；支持按公司名元数据过滤，将首轮候选池缩至目标文档。
- 引入**检索前查询优化**：基于 LLM 自动生成 **3 个角度**的检索变体（语义扩展 / 关键词聚焦 / 结构化条件），并结合元数据过滤构建 ChromaDB `where` 子句，解决用户表达模糊导致的召回不足。
- 实现**检索后精排与校正**：首轮向量检索召回 **Top-28** 候选，经 LLM（GPT-4o-mini / Qwen-Turbo）批量重排后取 **Top-6**；LLM 相关性评分与向量相似度按 **0.7:0.3** 加权融合；结合 LangGraph 状态机实现“检索 → 相关性评分 → 查询重写 → 再检索”的闭环反馈，最多 **2 轮**检索自动终止。
- 支持**父文档检索（Parent Document Retrieval）**：以 **300 tokens 子块**召回、返回整页父文档作为上下文，兼顾检索精度与上下文完整性。

### 系统评估（Evaluator）【规划中】

- 计划构建自动化评测体系：检索侧关注**页面精确率**与**上下文精确度**；生成侧关注**忠实度**（答案是否基于检索上下文）与**答案相关性**（是否回应用户查询），用于量化迭代各模块改进对端到端效果的影响。

---

## License

MIT
