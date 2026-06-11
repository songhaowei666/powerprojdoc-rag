# Markdown Reports Merging Spec

> **同步声明**：本文档严格反向推导自 `src/markdown_reports_merging.py` 与 `src/parsed_reports_merging.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

本模块负责将多种来源的解析后报告 JSON（主要为 MinerU 输出格式）统一规整为标准报告结构（metainfo + content.pages），便于后续分块、索引与检索流程消费。

核心处理流程：
1. 读取原始解析 JSON（单文件或批量目录）
2. 识别输入格式（MinerU 原生格式 或已规整格式），若为 MinerU 格式则执行格式转换
3. 逐页逐块映射内容类型（标题、正文、表格、图片、列表等）
4. 调用页面文本规整组件清洗与格式化
5. 输出规整后的报告对象列表，可选持久化到指定目录

---

## 2. 依赖

```
python >= 3.10
pandas
```

**内部依赖**：
- `src.parsed_reports_merging.PageTextPreparation`：负责页面块的清洗、表格/列表分组、Markdown 渲染等文本规整逻辑

---

## 3. 输入数据格式

### 3.1 MinerU 解析 JSON

```json
{
  "pdf_info": [
    {
      "page_idx": 0,
      "page_size": [595, 841],
      "para_blocks": [
        {
          "type": "title",
          "index": 0,
          "lines": [
            {
              "spans": [
                {"type": "text", "content": "年度报告"}
              ]
            }
          ]
        },
        {
          "type": "text",
          "index": 1,
          "lines": [...]
        },
        {
          "type": "table",
          "index": 2,
          "lines": [
            {
              "spans": [
                {"html": "<table>...</table>"}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**关键字段**：
- `pdf_info`：页面数组，每个页面含 `page_idx`、`page_size`、`para_blocks`
- `para_blocks`：页面内容块数组，每个块含 `type`、`index`、`lines`
- `lines`：行数组，每行含 `spans`，span 含 `type`/`content` 或 `html`

### 3.2 已规整格式（Docling 风格）

```json
{
  "metainfo": {...},
  "content": [
    {
      "page": 1,
      "content": [
        {"type": "section_header", "text": "..."},
        {"type": "text", "text": "..."},
        {"type": "table", "table_id": 0}
      ],
      "page_dimensions": {"width": 595, "height": 841}
    }
  ],
  "tables": [
    {"table_id": 0, "page": 1, "markdown": "...", "html": "..."}
  ]
}
```

### 3.3 辅助元数据（subset.csv）

可选的 CSV 文件，用于补充报告元信息：

| 列名 | 说明 |
|------|------|
| `file_name` | 原始文件名（含扩展名），用于匹配 |
| `sha1` | 文件哈希标识 |
| `company_name` | 公司名称 |

CSV 编码支持 `utf-8`，解码失败时回退 `gbk`。

---

## 4. 功能需求

### 4.1 批量报告处理入口（MinerUReportMerger）

```python
class MinerUReportMerger:
    def __init__(self)
    
    def process_reports(
        self,
        reports_dir: Path = None,
        reports_paths: List[Path] = None,
        output_dir: Path = None,
        subset_csv: Path = None,
    ) -> List[Dict]
```

**输入**：
- 报告来源：目录路径（自动收集 `*.json`）或文件路径列表
- 可选的输出目录路径
- 可选的 subset.csv 路径

**处理流程**：

1. **元数据加载**：若提供 subset.csv，建立「去扩展名文件名 → {sha1, company_name, file_name}」的映射表
2. **文件遍历**：遍历所有 JSON 报告文件
3. **格式识别与转换**：
   - 若 JSON 含 `pdf_info` 键，判定为 MinerU 格式，执行转换流程
   - 否则视为已规整格式，直接透传
4. **文件名解析**（MinerU 格式）：
   - 从文件名提取报告主名
   - 若文件名以 `MinerU_` 开头，移除前缀并按 `__` 切分取第一部分
5. **元信息填充**（MinerU 格式）：
   - 通过解析后的主名在 subset.csv 映射表中查找元数据
   - 组装 `metainfo`，包含：sha1_name、sha1、company_name、file_name、pages_amount、text_blocks_amount、tables_amount、pictures_amount、equations_amount、footnotes_amount
6. **页面文本规整**：调用 `PageTextPreparation.process_report` 处理 content 内容，生成按页组织的清洗后文本
7. **结果组装**：每条报告输出为 `{metainfo, content}` 结构
8. **可选持久化**：若指定输出目录，自动创建目录，按规则生成文件名并写入 JSON

**输出文件名规则**（按优先级）：
1. `metainfo.file_name` 的去扩展名 + `.json`
2. `metainfo.sha1_name` + `.json`
3. 解析后的报告主名 + `.json`

**返回**：规整后的报告对象列表

### 4.2 MinerU 格式转换流程

将 MinerU 原生 JSON 转为内部标准格式，核心步骤：

1. **页面遍历**：按 `page_idx` 升序遍历所有页面
2. **块遍历**：每页内按 `index` 升序遍历 `para_blocks`
3. **块类型映射**：

| MinerU type | 映射为 | 附加行为 |
|-------------|--------|----------|
| `title` | `section_header` | 提取 span 文本 |
| `text` | `text` / `paragraph` | 若以 `:` 结尾则映射为 `paragraph` |
| `table` | `table` | 提取 HTML → 转 Markdown → 分配全局 table_id → 元数据存入 tables 数组 |
| `image` | `picture` | picture_id 固定为 0 |
| `list` | 递归展开 | 对子块按 index 排序后递归处理，合并统计 |
| 其他 | `text` | 提取 span 文本 |

4. **页面维度**：从 `page_size` 提取宽高，默认回退 `[595, 841]`
5. **统计计数**：分别统计 text、table、picture 块数量，写入 metainfo

### 4.3 文本提取规则

- 逐行遍历 `lines`，每行内按顺序拼接 `spans` 中 `type == "text"` 且 `content` 非空的字符串
- 行内 spans 直接拼接，行间也直接拼接（无分隔符）

### 4.4 表格处理规则

1. **HTML 提取**：优先在当前块的 `lines → spans → html` 中查找；若未找到，递归搜索子块
2. **Markdown 转换**：
   - 使用 pandas `read_html` 解析 HTML
   - 取第一个 DataFrame，调用 `to_markdown(index=False)`
   - 解析失败时原样返回 HTML 字符串
3. **全局 ID 分配**：每张表格分配递增的整数 `table_id`

---

## 5. 页面文本规整（PageTextPreparation）

```python
class PageTextPreparation:
    def __init__(self, use_serialized_tables: bool = False, serialized_tables_instead_of_markdown: bool = False)
    
    def process_reports(self, reports_dir: Path = None, reports_paths: List[Path] = None, output_dir: Path = None) -> List[Dict]
    
    def process_report(self, report_data: dict) -> dict
    
    def prepare_page_text(self, page_number: int) -> str
    
    def export_to_markdown(self, reports_dir: Path, output_dir: Path)
```

### 5.1 __init__

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_serialized_tables` | `bool` | `False` | 是否使用序列化表格替代 Markdown 表格 |
| `serialized_tables_instead_of_markdown` | `bool` | `False` | `True` 时完全用序列化文本替代；`False` 时拼接 Markdown + 序列化描述 |

### 5.2 process_report

处理单份报告，返回规整后的每页文本结构：

```json
{
  "chunks": null,
  "pages": [
    {"page": 1, "text": "..."},
    {"page": 2, "text": "..."}
  ]
}
```

处理步骤：
1. 逐页调用 `prepare_page_text(page_number)`
2. 调用 `_clean_text(page_text)` 用正则清洗文本（替换 slash command、glyph、cap 等）
3. 若存在修正，打印修正数量与详情

### 5.3 prepare_page_text

主流程：处理页面块并组装为字符串。

1. 根据页码获取页面数据
2. `_filter_blocks`：移除 `page_footer`、`picture` 类型块
3. `_apply_formatting_rules`：按规则处理块，合并表格组、列表组、脚注等
4. 首尾块去除多余空白，行拼接为字符串

### 5.4 _apply_formatting_rules

核心格式化逻辑，按块类型分组处理：

| 块类型 | 处理行为 |
|--------|----------|
| `page_header` | 前 3 个块用 `# `，其余用 `## ` |
| `section_header` | 同 page_header 规则 |
| `paragraph` | 若不以冒号结尾或下一块不是 table/list_item，包装为 `### ` |
| `table` | 触发表格组渲染（含前置冒号标题、脚注） |
| `list_item` | 触发列表组渲染（含前置冒号标题、脚注） |
| `text` / `caption` / `footnote` / `checkbox_*` / `formula` | 直接追加文本 |

**表格组渲染**（`_render_table_group`）：
- 标题/说明文本前置
- 表格内容通过 `_get_table_by_id(table_id)` 获取
- 若启用 `use_serialized_tables`，调用 `_get_serialized_table_text` 获取序列化描述
- 脚注追加

**列表组渲染**（`_render_list_group`）：
- 标题/说明文本前置
- `list_item` 转为 `- ` 前缀
- `checkbox_selected` 转为 `[x] `， `checkbox_unselected` 转为 `[ ] `
- 脚注追加

### 5.5 _get_serialized_table_text

```python
def _get_serialized_table_text(self, table: dict, serialized_tables_instead_of_markdown: bool) -> str
```

- 若 `table` 无 `serialized` 字段，回退 `table.get("markdown", "")`
- 若 `serialized_tables_instead_of_markdown=True`，返回纯序列化文本（`information_blocks` 拼接）
- 若 `serialized_tables_instead_of_markdown=False`，返回 `markdown + "\nDescription of the table entities:\n" + serialized_text`

### 5.6 _clean_text

用正则清洗文本，统计修正次数：

| 模式 | 示例 | 替换 |
|------|------|------|
| `/zero.pl.tnum` 等 slash command | `/zero.pl.tnum` | `0` |
| `glyph<...>` | `glyph<0x0041>` | 空字符串 |
| `/A.cap` | `/A.cap` | `A` |

完整 command_mapping 包含：`zero`~`nine`, `period`, `comma`, `colon`, `hyphen`, `percent`, `dollar`, `space`, `plus`, `minus`, `slash`, `asterisk`, `lparen`, `rparen`, `parenleft`, `parenright`, `wedge.1_E`

### 5.7 export_to_markdown

将处理后的报告导出为 Markdown 文件：
- 每页之间用 `\n\n---\n\n# Page N\n\n` 分隔
- 输出文件名为 `{sha1_name}.md`

---

## 6. 测试要点

> 以下测试场景按 TDD 流程单独列出，spec 中不绑定具体方法名。

### 6.1 批量处理基础场景

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T1 | 传入空目录 | 返回空列表，不报错 |
| T2 | 传入空路径列表 | 返回空列表，不报错 |
| T3 | 传入单个 MinerU 格式 JSON 文件 | 正确识别格式，完成转换并返回单元素列表 |
| T4 | 传入单个已规整格式 JSON 文件 | 跳过转换，直接规整后返回 |
| T5 | 混合传入 MinerU 格式和已规整格式 | 分别正确处理，返回合并列表 |

### 6.2 格式转换与内容映射

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T6 | MinerU JSON 含 title 块 | 映射为 section_header，文本正确提取 |
| T7 | MinerU JSON 含 text 块 | 映射为 text，文本正确提取 |
| T8 | text 块文本以冒号结尾 | 映射为 paragraph 类型 |
| T9 | MinerU JSON 含 table 块 | 分配递增 table_id，HTML 正确提取并转为 Markdown，tables 数组正确填充 |
| T10 | table 块的 HTML 解析失败 | Markdown 字段回退为原始 HTML 字符串 |
| T11 | MinerU JSON 含 image 块 | 映射为 picture，picture_id 为 0 |
| T12 | MinerU JSON 含 list 块（含子块） | 递归展开子块，内容正确合并，统计准确 |
| T13 | 未知类型的 block | 按 text 处理，尝试提取文本 |
| T14 | 多页报告 | 按 page_idx 升序处理，每页内容独立 |
| T15 | 页面含空 text span 或缺失 content | 忽略空内容，不生成对应块 |

### 6.3 元信息解析与填充

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T16 | 提供有效的 subset.csv | 按文件名匹配，正确填充 sha1、company_name、file_name |
| T17 | subset.csv 编码为 gbk | 正确解码并读取 |
| T18 | subset.csv 缺少 file_name 列 | 忽略 CSV，返回空映射，metainfo 使用默认值 |
| T19 | subset.csv 文件不存在 | 返回空映射，metainfo 使用默认值 |
| T20 | MinerU 文件名以 `MinerU_` 开头并含 `__` | 正确提取报告主名（移除前缀，取 `__` 前部分） |
| T21 | MinerU 文件名无特殊前缀 | 直接使用文件 stem 作为主名 |
| T22 | metainfo 中 file_name 非空 | 输出文件名为 file_name 去扩展名 + `.json` |
| T23 | metainfo 中 file_name 为空但 sha1_name 非空 | 输出文件名为 sha1_name + `.json` |
| T24 | metainfo 中 file_name 和 sha1_name 均为空 | 输出文件名为解析的报告主名 + `.json` |

### 6.4 统计与边界

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T25 | 空页面（无 para_blocks） | 页面内容为空块列表，pages_amount 正确 |
| T26 | 页面缺少 page_size | 宽高回退为默认值 595×841 |
| T27 | 多表格场景 | table_id 全局递增不重复，各表格 Markdown 独立 |
| T28 | 表格 HTML 为空字符串 | Markdown 返回空字符串 |
| T29 | 输出目录已存在 | 不报错，正常写入 |
| T30 | 输出目录不存在 | 自动递归创建 |

### 6.5 页面文本规整

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T31 | 页面含 page_footer | 被过滤，不出现在最终文本 |
| T32 | 页面含 picture | 被过滤，不出现在最终文本 |
| T33 | 表格前有冒号结尾的 paragraph | 表格组包含该 paragraph 作为标题 |
| T34 | 列表前有冒号结尾的 paragraph | 列表组包含该 paragraph 作为标题 |
| T35 | 文本含 slash command | `_clean_text` 正确替换并统计修正次数 |
| T36 | 启用 use_serialized_tables | `_get_table_by_id` 返回序列化文本而非纯 Markdown |

---

## 7. 异常与边界行为

| 场景 | 行为 |
|------|------|
| 输入 JSON 文件无法解析 | 由 `json.load` 抛出异常，外层未捕获 |
| MinerU 块缺少 `lines` | `_extract_mineru_text` 返回空字符串，通常被忽略 |
| table 块在 tables 数组中找不到对应 ID | 由 `PageTextPreparation` 阶段抛出 `ValueError` |
| subset.csv 编码既非 utf-8 也非 gbk | 第二次 `pd.read_csv` 仍可能抛异常 |
| 输出目录路径为文件而非目录 | `mkdir(parents=True, exist_ok=True)` 行为取决于操作系统 |

---

## 8. 当前实现缺陷（需后续优化）

1. **picture_id 硬编码**：所有图片块的 `picture_id` 固定为 0，无法区分多图
2. **equations_amount / footnotes_amount 恒为 0**：MinerU 转换流程未统计公式和脚注
3. **HTML 解析依赖 pandas**：`pd.read_html` 对复杂表格支持有限，且异常处理较粗糙
4. **文件名解析规则硬编码**：`MinerU_` 前缀和 `__` 分隔符与外部约定强耦合
5. **递归无深度限制**：list 块的嵌套递归在极端情况下可能导致栈溢出
6. **page_size 回退值硬编码**：默认 595×841 未参数化
7. **输出文件名冲突**：不同来源报告若解析出相同主名，后者会覆盖前者
8. **PageTextPreparation 类名与文件路径不一致**：定义在 `parsed_reports_merging.py` 中，而非 `markdown_reports_merging.py`

---

## 9. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始实现，支持 MinerU JSON 批量转换为规整报告结构 |
| v1.1 | 2026-06-11 | 补充 PageTextPreparation 详细接口（序列化表格、export_to_markdown、_clean_text 等） |
