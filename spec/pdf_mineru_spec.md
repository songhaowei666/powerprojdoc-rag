# pdf_mineru Spec

> **同步声明**：本文档严格反向推导自 `src/pdf_mineru.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`pdf_mineru` 封装 MinerU 精准解析 API（v4），支持两种 PDF 提交方式：

| 方式 | 入口函数 | MinerU 接口 | 轮询函数 |
|------|----------|-------------|----------|
| 远程 URL | `get_task_id` | `POST /api/v4/extract/task` | `get_result(task_id)` |
| 本地上传 | `upload_local_file` | `POST /api/v4/file-urls/batch` + PUT | `get_batch_result(batch_id)` |

配置通过 `src.config.settings.mineru_api_key`（环境变量 `MINERU_API_KEY`）读取。

---

## 2. 远程 URL 提交

### get_task_id

```python
def get_task_id(file_name: str) -> str
```

- 将 `file_name` 拼接到固定 OSS 前缀 `https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/` 作为 `url`
- 请求体：`is_ocr=True`，`enable_formula=False`
- 返回 `task_id`

### get_result

```python
def get_result(task_id: str) -> None
```

- 轮询 `GET /api/v4/extract/task/{task_id}`，间隔 5 秒
- `state` 为 `pending` / `running` 时继续等待
- `state=done` 时下载 `full_zip_url` 到 `{task_id}.zip` 并解压到 `{task_id}/`
- 出错或未知状态时打印信息后返回

---

## 3. 本地上传

### upload_local_file

```python
def upload_local_file(
    file_path: str,
    *,
    is_ocr: bool = True,
    enable_formula: bool = False,
    enable_table: bool = True,
    model_version: str = "vlm",
    data_id: str | None = None,
) -> str
```

**流程**：

1. 校验本地文件存在
2. `POST /api/v4/file-urls/batch` 申请预签名上传 URL
3. `PUT` 本地文件二进制到预签名 URL（**不设置 Content-Type**）
4. 返回 `batch_id`（上传完成后 MinerU 自动提交解析任务，无需再调 `/extract/task`）

**异常**：

| 场景 | 异常 |
|------|------|
| 文件不存在 | `FileNotFoundError` |
| 未配置 API Key | `ValueError` |
| API 返回非成功 | `RuntimeError`（含 MinerU 错误信息） |
| PUT 上传失败 | `RuntimeError` |

### get_batch_result

```python
def get_batch_result(batch_id: str, *, file_index: int = 0) -> None
```

- 轮询 `GET /api/v4/extract-results/batch/{batch_id}`，间隔 5 秒
- 取 `extract_result[file_index]` 作为单文件结果
- `state` 为 `waiting-file` / `pending` / `running` / `converting` 时继续等待
- `state=done` 时下载 `full_zip_url` 到 `{batch_id}.zip` 并解压到 `{batch_id}/`
- `state=failed` 或存在 `err_msg` 时打印错误并返回

---

## 4. 工具函数

### unzip_file

```python
def unzip_file(zip_path: str, extract_dir: str | None = None) -> None
```

解压 zip 到指定目录；`extract_dir` 默认为去掉 `.zip` 后缀的路径。

---

## 5. 与 Pipeline 集成

`Pipeline.export_reports_to_markdown` 当前使用 URL 方式（`get_task_id` + `get_result`），解压目录名为 `task_id`。

若改用本地上传，调用顺序为：

```python
batch_id = pdf_mineru.upload_local_file(str(pdf_path))
pdf_mineru.get_batch_result(batch_id)
# 解压目录为 {batch_id}/，其中 full.md 为 Markdown 结果
```

---

## 6. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-06-21 | 初版：URL 提交 + 本地上传双路径 |
