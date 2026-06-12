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
- `src.pdf_mineru`：PDF 转 Markdown
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

路径配置类，根据 `root_path` 与运行参数初始化所有流程目录与文件路径。

关键路径：
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

## 4. Pipeline 方法

### 4.1 初始化与路径

```python
class Pipeline:
    def __init__(
        self,
        root_path: Path,
        subset_name: str = "subset.csv",
        questions_file_name: str = "questions.json",
        pdf_reports_dir_name: str = "pdf_reports",
        use_serialized_tables: bool = False,
        config_suffix: str = "",
    )
    
    def _convert_json_to_csv_if_needed(self)
```

**初始化流程**：
1. 保存 `use_serialized_tables` 和 `config_suffix`
2. 直接实例化 `PipelineConfig(...)` 并赋值给 `self.paths`
3. 调用 `_convert_json_to_csv_if_needed()`

**初始化参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_path` | `Path` | — | 数据集根目录 |
| `subset_name` | `str` | `"subset.csv"` | 子集元数据文件名 |
| `questions_file_name` | `str` | `"questions.json"` | 问题文件名 |
| `pdf_reports_dir_name` | `str` | `"pdf_reports"` | PDF 报告目录名 |
| `use_serialized_tables` | `bool` | `False` | 是否使用序列化表格，影响数据库和 merged 目录后缀 |
| `config_suffix` | `str` | `""` | 配置后缀，影响 answers 文件名和数据库目录名 |

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
- `output_dir` 固定使用 `self.paths.merged_reports_path`
- `subset_csv` 固定使用 `self.paths.subset_path`

**处理流程**：
1. 延迟导入 `src.markdown_reports_merging.MinerUReportMerger`
2. 实例化 `MinerUReportMerger`
3. 调用 `process_reports(...)`，传入 `reports_dir`、`reports_paths`、`output_dir`、`subset_csv`
4. 打印处理数量
5. 返回规整后的报告对象列表

**返回**：`List[Dict]`，每个元素为 `{"metainfo": ..., "content": ...}`

### 4.4 分块与索引

- `chunk_reports(self, include_serialized_tables: bool = False)`：Markdown 报告分块
- `chunk_reports2(self, include_serialized_tables: bool = False)`：通用报告分块
- `create_vector_dbs(self)`：创建向量数据库
- `create_bm25_db(self)`：创建 BM25 索引

### 4.5 问题回答

#### process_questions

```python
def process_questions(
    self,
    parent_document_retrieval: bool = False,
    llm_reranking: bool = False,
    llm_reranking_sample_size: int = 30,
    top_n_retrieval: int = 10,
    parallel_requests: int = 1,
    pipeline_details: str = "",
    submission_file: bool = True,
    full_context: bool = False,
    api_provider: str = "openai",
    answering_model: str = "gpt-4-turbo",
)
```

批量处理问题并生成答案文件。所有问题处理相关参数均在调用时显式传入，不再依赖全局 RunConfig。

#### answer_single_question

```python
def answer_single_question(
    self,
    question: str,
    kind: str = "string",
    parent_document_retrieval: bool = False,
    llm_reranking: bool = False,
    llm_reranking_sample_size: int = 30,
    top_n_retrieval: int = 10,
    parallel_requests: int = 1,
    api_provider: str = "openai",
    answering_model: str = "gpt-4-turbo",
    full_context: bool = False,
)
```

单条问题即时推理，返回结构化答案。

#### _get_next_available_filename

```python
def _get_next_available_filename(self, base_path: Path) -> Path
```

获取不重复文件名。

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

### 5.2 问题回答参数

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T7 | 使用默认参数调用 `process_questions` | 使用默认模型和默认检索参数正常执行 |
| T8 | 显式传入 `answering_model` 和 `parallel_requests` | QuestionsProcessor 接收对应参数 |
| T9 | 启用 `llm_reranking` | QuestionsProcessor 接收 `llm_reranking=True` |
| T10 | 启用 `parent_document_retrieval` | QuestionsProcessor 接收 `parent_document_retrieval=True` |

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
| v1.2 | 2026-06-12 | 移除 `RunConfig`，将路径相关参数平铺到 `Pipeline.__init__`，将问题处理参数平铺到 `process_questions` / `answer_single_question` 方法参数 |
