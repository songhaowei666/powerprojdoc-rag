# Markdown Reports Merging Spec

> **同步声明**：本文档严格反向推导自 `src/markdown_reports_merging.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

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

### 4.1 批量报告处理入口

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
6. **页面文本规整**：调用文本规整组件处理 content 内容，生成按页组织的清洗后文本
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

## 5. 测试要点

> 以下测试场景按 TDD 流程单独列出，spec 中不绑定具体方法名。

### 5.1 批量处理基础场景

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T1 | 传入空目录 | 返回空列表，不报错 |
| T2 | 传入空路径列表 | 返回空列表，不报错 |
| T3 | 传入单个 MinerU 格式 JSON 文件 | 正确识别格式，完成转换并返回单元素列表 |
| T4 | 传入单个已规整格式 JSON 文件 | 跳过转换，直接规整后返回 |
| T5 | 混合传入 MinerU 格式和已规整格式 | 分别正确处理，返回合并列表 |

### 5.2 格式转换与内容映射

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

### 5.3 元信息解析与填充

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

### 5.4 统计与边界

| # | 测试场景 | 预期行为 |
|---|---------|---------|
| T25 | 空页面（无 para_blocks） | 页面内容为空块列表，pages_amount 正确 |
| T26 | 页面缺少 page_size | 宽高回退为默认值 595×841 |
| T27 | 多表格场景 | table_id 全局递增不重复，各表格 Markdown 独立 |
| T28 | 表格 HTML 为空字符串 | Markdown 返回空字符串 |
| T29 | 输出目录已存在 | 不报错，正常写入 |
| T30 | 输出目录不存在 | 自动递归创建 |

---

## 6. 异常与边界行为

| 场景 | 行为 |
|------|------|
| 输入 JSON 文件无法解析 | 由 `json.load` 抛出异常，外层未捕获 |
| MinerU 块缺少 `lines` | `_extract_mineru_text` 返回空字符串，通常被忽略 |
| table 块在 tables 数组中找不到对应 ID | 由 `PageTextPreparation` 阶段抛出 `ValueError` |
| subset.csv 编码既非 utf-8 也非 gbk | 第二次 `pd.read_csv` 仍可能抛异常 |
| 输出目录路径为文件而非目录 | `mkdir(parents=True, exist_ok=True)` 行为取决于操作系统 |

---

## 7. 当前实现缺陷（需后续优化）

1. **picture_id 硬编码**：所有图片块的 `picture_id` 固定为 0，无法区分多图
2. **equations_amount / footnotes_amount 恒为 0**：MinerU 转换流程未统计公式和脚注
3. **HTML 解析依赖 pandas**：`pd.read_html` 对复杂表格支持有限，且异常处理较粗糙
4. **文件名解析规则硬编码**：`MinerU_` 前缀和 `__` 分隔符与外部约定强耦合
5. **递归无深度限制**：list 块的嵌套递归在极端情况下可能导致栈溢出
6. **page_size 回退值硬编码**：默认 595×841 未参数化
7. **输出文件名冲突**：不同来源报告若解析出相同主名，后者会覆盖前者

---

## 8. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始实现，支持 MinerU JSON 批量转换为规整报告结构 |
