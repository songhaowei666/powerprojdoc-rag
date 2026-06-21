# Pipeline Spec

> **同步声明**：本文档严格反向推导自 `src/pipeline.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`Pipeline` 类是项目主流程编排入口，负责将 PDF 报告从解析、规整、分块到构建索引、回答问题的全流程串联起来。配置通过 `PipelineConfig`（路径）与 `RunConfig`（运行参数）分离管理。

---

## 2. 依赖

```
python >= 3.10
pandas
pyprojroot
```

**内部依赖**：
- `src.pdf_mineru`：PDF 转 Markdown（远程 URL 或本地上传，详见 `spec/pdf_mineru_spec.md`）
- `src.parsed_reports_merging.PageTextPreparation`：页面文本规整
- `src.markdown_reports_merging.MinerUReportMerger`：MinerU JSON 规整为标准报告结构
- `src.text_splitter.TextSplitter`：报告分块
- `src.ingestion.VectorDBIngestor` / `BM25Ingestor`：索引构建
- `src.questions_processing.QuestionsProcessor`：问题处理与答案生成
- `src.config.settings`：配置读取
- `src.tables_serialization.TableSerializer`：表格序列化

---

## 3. 配置

### 3.1 PipelineConfig

路径与目录配置类，根据 `root_path` 与运行参数初始化所有流程目录与文件路径。

**字段**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_path` | `Path` | — | 数据集根目录 |
| `subset_name` | `str` | `"subset.csv"` | 子集元数据文件名 |
| `questions_file_name` | `str` | `"questions.json"` | 问题文件名 |
| `pdf_reports_dir_name` | `str` | `"pdf_reports"` | PDF 报告目录名 |
| `use_serialized_tables` | `bool` | `False` | 是否使用序列化表格，影响数据库和 merged 目录后缀 |
| `config_suffix` | `str` | `""` | 配置后缀，影响 answers 文件名 |
| `company_code` | `str` | `"001"` | 公司编码，写入 merge 后报告的 metainfo |

**派生路径**（在 `__post_init__` 中计算）：
- `subset_path`：`root_path / subset_name`
- `questions_file_path`：`root_path / questions_file_name`
- `pdf_reports_dir`：`root_path / pdf_reports_dir_name`
- `debug_data_path`：`root_path / "debug_data"`
- `databases_path`：`root_path / f"databases{suffix}"`
- `vector_db_dir`：`databases_path / settings.vector_db_subdir`
- `documents_dir`：`databases_path / settings.chunked_reports_subdir`
- `bm25_db_path`：`databases_path / settings.bm25_dbs_subdir`
- `merged_reports_path`：`debug_data_path / f"02_merged_reports{suffix}"`
- `reports_markdown_path`：`debug_data_path / f"03_reports_markdown{suffix}"`

**便捷构造**：

```python
PipelineConfig.from_root(root_path, **overrides)  # 等价于 PipelineConfig(root_path=root_path, ...)
```

## 4. Pipeline 方法

### 4.1 初始化与路径

```python
class Pipeline:
    def __init__(self, config: PipelineConfig)
    
    def _convert_json_to_csv_if_needed(self)
```

**初始化流程**：
1. 接收 `PipelineConfig` 实例并赋值给 `self.config`
2. 调用 `_convert_json_to_csv_if_needed()`

### 4.2 PDF 与 Markdown 处理

- `parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10)`：解析 PDF 报告
- `parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10)`：多进程并行解析
- `export_reports_to_markdown(self, file_name)`：将指定 PDF 转换为 Markdown

### 4.3 报告规整

#### merge_mineru_reports

**签名**：
```python
def merge_mineru_reports(
    self,
    reports_dir: Path = None,
    reports_paths: List[Path] = None,
) -> List[Dict]
```

**功能**：将 MinerU 解析后的 JSON 报告批量规整为标准报告结构（metainfo + content.pages），便于后续分块与索引。

**参数**：
- `reports_dir`：输入 JSON 文件目录，自动收集该目录下所有 `*.json`
- `reports_paths`：输入 JSON 文件路径列表
- 两者互斥提供，至少传一个；同时传入时，`reports_paths` 优先

**默认行为**：
- `output_dir` 固定使用 `self.config.merged_reports_path`
- `company_code` 固定使用 `self.config.company_code`（默认 `"001"`）

**处理流程**：
1. 延迟导入 `src.markdown_reports_merging.MinerUReportMerger`
2. 实例化 `MinerUReportMerger`
3. 调用 `process_reports(...)`，传入 `reports_dir`、`reports_paths`、`output_dir`、`company_code`
4. 打印处理数量
5. 返回规整后的报告对象列表

**返回**：`List[Dict]`，每个元素为 `{"metainfo": ..., "content": ...}`

### 4.4 分块与索引

- `chunk_reports(self, include_serialized_tables: bool = False)`：Markdown 报告分块
- `chunk_reports2(self, include_serialized_tables: bool = False)`：通用报告分块
- `create_vector_dbs(self)`：创建向量数据库，chunk metadata 的 `company_code` 取自分块 JSON 的 `metainfo`
- `create_bm25_db(self)`：创建 BM25 索引，同上

### 4.5 向量检索

#### vector_retrieve

```python
def vector_retrieve(
    self,
    query: str,
    company_code: str = "",
    top_n: int = 3,
    return_parent_pages: bool = False,
    index_name: str = "default",
) -> list[dict]
```

在 `self.config.vector_db_dir` 的 ChromaDB 向量库中检索与 query 最相关的文本块。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `str` | — | 查询文本 |
| `company_code` | `str` | `""` | 公司编码，用于 metadata 过滤；为空时使用 `PipelineConfig.company_code` |
| `top_n` | `int` | `3` | 返回结果数量上限 |
| `return_parent_pages` | `bool` | `False` | 为 True 时返回整页内容 |
| `index_name` | `str` | `"default"` | ChromaDB collection 名称 |

**实现**：实例化 `VectorRetriever`，传入 `vector_db_dir`、`documents_dir`，调用 `retrieve(...)`。

**返回**：`list[dict]`，每项含 `distance`、`page`、`text`。

#### bm25_retrieve

```python
def bm25_retrieve(
    self,
    query: str,
    top_n: int = 3,
    return_parent_pages: bool = False,
    index_name: str = "default",
) -> list[dict]
```

在 `self.config.bm25_db_path` 的 BM25 索引中做关键词检索。

**实现**：实例化 `BM25Retriever`，传入 `bm25_db_path`、`documents_dir`，调用 `retrieve(...)`。

**返回**：`list[dict]`，每项含 `distance`（BM25 分数）、`page`、`text`。

#### hybrid_retrieve

```python
def hybrid_retrieve(
    self,
    query: str,
    company_code: str = "",
    llm_reranking_sample_size: int = 28,
    documents_batch_size: int = 10,
    top_n: int = 6,
    llm_weight: float = 0.7,
    return_parent_pages: bool = False,
) -> list[dict]
```

向量召回 + LLM 重排的混合检索。

**实现**：实例化 `HybridRetriever`，传入 `vector_db_dir`、`documents_dir`，调用 `retrieve(...)`。

**返回**：`list[dict]`，经重排后的文档列表（含 `distance`、`page`、`text` 等）。

---

## 5. 测试要点

### 5.1 merge_mineru_reports

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T1 | 传入单个 MinerU JSON 文件路径 | 调用 MinerUReportMerger.process_reports，返回单元素列表 |
| T2 | 传入输入目录 | 自动收集目录下所有 JSON，返回规整后列表 |
| T3 | 同时传入 reports_dir 和 reports_paths | 按 MinerUReportMerger 行为处理（reports_dir 被 reports_paths 覆盖或共同处理） |
| T4 | 输出目录不存在 | 由 MinerUReportMerger 自动创建 |
| T5 | subset.csv 不存在 | MinerUReportMerger 使用空元数据映射，不报错 |
| T6 | 输入为空 | 返回空列表 |

### 5.2 检索方法

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T7 | 调用 `vector_retrieve` | 返回 ChromaDB 相似度检索结果 |
| T8 | 调用 `bm25_retrieve` | 返回 BM25 关键词检索结果 |
| T9 | 调用 `hybrid_retrieve` | 返回 LLM 重排后的混合检索结果 |

---

## 6. 异常与边界行为

| 场景 | 行为 |
|------|------|
| `reports_dir` 和 `reports_paths` 均为空 | `MinerUReportMerger.process_reports` 返回空列表 |
| 输入 JSON 格式非法 | 由 `json.load` 抛出异常 |
| 输出目录为文件 | `mkdir(parents=True, exist_ok=True)` 行为取决于操作系统 |

---

## 7. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始实现，支持 PDF 解析、分块、索引、问答流程 |
| v1.1 | 2026-06-12 | 新增 `merge_mineru_reports` 方法，支持调用 MinerUReportMerger 批量规整 JSON 报告 |
| v1.6 | 2026-06-12 | 移除 `process_questions`；新增 `bm25_retrieve`、`hybrid_retrieve` |
