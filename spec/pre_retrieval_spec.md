# Pre-Retrieval Processing Module Spec

## 1. 概述

`pre_retrieval_processing.py` 是 RAG 检索流程的前置处理模块，位于 `src/` 目录下。它负责在正式检索（BM25 / Vector / Hybrid）之前，对用户输入进行两项核心预处理：

1. **元数据过滤条件生成**：借鉴 SelfQueryRetriever 思路，根据用户查询自动生成 ChromaDB 的 `where` 元数据过滤条件（含 `unit_code` 等字段）。若调用时已传入 `unit_code`，则直接复用、跳过 LLM。
2. **多角度查询构建**：将原始查询扩展为 3 个不同检索角度的变体查询，以提升召回率。

处理结果以 Python 对象形式返回，由调用方直接使用，不持久化到文件。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 元数据智能过滤 | 用户未传编码时，通过 LLM 从查询文本推断目标省公司并生成 `where` 条件 |
| 编码短路复用 | 用户已传编码时，直接组装为 `{"unit_code": {"$eq": "xxx"}}`，零 LLM 调用 |
| 多字段支持 | 除 `unit_code` 外，支持 `year`、`report_type` 等元数据字段的自动推断 |
| 多角度召回 | 从语义、关键词、结构化条件三个角度生成检索变体 |
| 结果可审计 | 返回结构化对象，便于调用方日志记录与调试 |
| LangGraph 兼容 | 提供 `invoke(inputs: dict)` 统一接口 |

---

## 3. 架构设计

```
┌─────────────────────────────────────┐
│     PreRetrievalProcessor           │
│  (统一入口，兼容 LangGraph invoke)    │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
┌──────────────────┐  ┌───────────────────┐
│MetadataFilterBuilder│ │MultiAngleQueryBuilder
│ (元数据过滤条件)  │  │  (3角度查询生成)   │
└──────────────────┘  └───────────────────┘
         │                     │
         ▼                     ▼
   MetadataFieldInfo       QueryAngle
   (元数据字段定义)        (Pydantic模型)
```

---

## 4. 依赖清单

```
python >= 3.10
pydantic
pyyaml
src.api_requests.APIProcessor
```

**内部依赖**：
- `src.api_requests.APIProcessor`：LLM 调用封装

---

## 5. 数据模型

### 5.1 MetadataField

```python
class MetadataField(BaseModel):
    name: str         # 字段名，如 "unit_code"
    description: str  # 字段说明，供 LLM 理解
    type: Literal["string", "integer", "float"]
```

### 5.2 预定义元数据字段

当前仅支持 `unit_code` 字段。后续如需扩展（如 `year`、`company_name` 等），可在此列表中追加。

```python
METADATA_FIELDS = [
    MetadataField(
        name="unit_code",
        description="省公司编码，27家省公司之一，可选值：001~027",
        type="string",
    ),
]
```

### 5.3 ProvinceCompanyCode（省公司编码映射）

```python
class ProvinceCompanyCode(BaseModel):
    name: str   # 省公司简称，如"北京"
    code: str   # 编码占位符，如"001"
```

模块内常量 `PROVINCE_CODE_BASE`，包含全部 27 家省公司。当前使用 `"001" ~ "027"` 字符串序号占位，后续可直接替换为真实编码。

| 省公司 | 编码 |
|--------|------|
| 北京 | 001 |
| 天津 | 002 |
| 河北 | 003 |
| 山西 | 004 |
| 山东 | 005 |
| 上海 | 006 |
| 江苏 | 007 |
| 浙江 | 008 |
| 安徽 | 009 |
| 福建 | 010 |
| 湖北 | 011 |
| 湖南 | 012 |
| 河南 | 013 |
| 江西 | 014 |
| 四川 | 015 |
| 重庆 | 016 |
| 辽宁 | 017 |
| 吉林 | 018 |
| 黑龙江 | 019 |
| 陕西 | 020 |
| 甘肃 | 021 |
| 青海 | 022 |
| 宁夏 | 023 |
| 新疆 | 024 |
| 西藏 | 025 |
| 内蒙古 | 026 |
| 广西 | 027 |

### 5.4 QueryAngle

```python
class QueryAngle(BaseModel):
    angle_name: str      # 角度名称，如"semantic_expansion"
    query_text: str      # 该角度下的查询文本
    rationale: str       # 生成理由简述
```

### 5.5 PreRetrievalResult

```python
class PreRetrievalResult(BaseModel):
    original_query: str
    metadata_filter: dict    # ChromaDB where 条件，如 {"unit_code": {"$eq": "001"}}
    angles: List[QueryAngle]
```

---

## 6. 类接口定义

### 6.1 MetadataFilterBuilder

```python
class MetadataFilterBuilder:
    def __init__(self, provider: str = "openai")
    
    def build(self, query: str, unit_code: str = None) -> dict
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `str` | - | 用户原始查询 |
| `unit_code` | `str` | `None` | 若已提供，直接组装为 `{"unit_code": {"$eq": unit_code}}`，不调用 LLM |

**返回**：ChromaDB 格式的 `where` 条件字典。

**LLM 推断策略**（借鉴 SelfQueryRetriever）：
- System Prompt 注入 `METADATA_FIELDS` 定义 + `PROVINCE_CODE_BASE` 列表。
- 要求 LLM 分析查询，推断需要过滤的元数据字段及其值。
- 输出格式使用 Pydantic 结构化：`{"filters": [{"field": "unit_code", "operator": "$eq", "value": "001"}]}`
- 模块内部将 LLM 输出转换为 ChromaDB `where` 格式：`{"unit_code": {"$eq": "001"}, "year": {"$gte": 2024}}`
- 若查询中无法匹配任何元数据条件，返回 `{}`（空过滤）。

### 6.2 MultiAngleQueryBuilder

```python
class MultiAngleQueryBuilder:
    def __init__(self, provider: str = "openai")
    
    def build(self, query: str) -> List[QueryAngle]
```

**三个角度定义**：

| 角度 | 名称 | 说明 |
|------|------|------|
| ① | `semantic_expansion` | 语义扩展：对原查询进行同义改写，保留原意但换表述 |
| ② | `keyword_focus` | 关键词聚焦：提取查询中的核心实体、指标、时间，重组为关键词导向的查询 |
| ③ | `structured_condition` | 结构化条件：补充隐含的时间范围、对比维度、限定条件，使查询更结构化 |

**返回**：`List[QueryAngle]`，长度固定为 3，顺序与上表一致。

### 6.3 输出模型

```python
class PreRetrievalResult(BaseModel):
    original_query: str
    metadata_filter: dict       # ChromaDB where 条件
    angles: List[QueryAngle]    # 三个角度的查询变体
```

两个处理器各自独立使用，由调用方按需组合。无需统一集成类。

---

## 7. Prompt 策略

### 7.1 元数据过滤条件生成 Prompt

**System**：
```
你是元数据过滤条件生成助手。请根据用户的查询，分析需要过滤的元数据字段。

当前可用元数据字段：
- unit_code（string）：省公司编码，可选值：001~027，对应省公司列表如下：
  001: 北京, 002: 天津, 003: 河北, ...（全部27家）

请仅从上述字段中选择，生成过滤条件。每个条件包含 field、operator、value。
支持的 operator：$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin。

若查询中无法确定任何过滤条件，返回空列表。
```

**Response Format**：
```python
class FilterCondition(BaseModel):
    field: str
    operator: str
    value: Union[str, int, float, List]

class MetadataFilterResponse(BaseModel):
    filters: List[FilterCondition]
    reasoning: str
```

### 7.2 多角度查询 Prompt

**System**：
```
你是查询扩展专家。请基于用户的原始查询，从以下三个角度生成检索友好的查询变体：

1. semantic_expansion：语义扩展，使用同义词、近义表达改写原查询。
2. keyword_focus：关键词聚焦，提取核心实体、指标、时间，去掉冗余修饰。
3. structured_condition：结构化条件，补充隐含的时间范围、对比维度、限定词。

每个角度输出查询文本和生成理由。
```

**Response Format**：
```python
class AngleItem(BaseModel):
    angle_name: str
    query_text: str
    rationale: str

class MultiAngleResponse(BaseModel):
    angles: List[AngleItem]
```

---

## 8. 输出示例

### 8.1 PreRetrievalResult（Python 字典形式）

```python
{
    "original_query": "北京公司2024年营业收入增长原因是什么？",
    "metadata_filter": {
        "unit_code": {"$eq": "001"}
    },
    "angles": [
        {
            "angle_name": "semantic_expansion",
            "query_text": "北京电力公司2024年收入增长的主要驱动因素有哪些？",
            "rationale": "将'营业收入'扩展为'收入'，'增长原因'扩展为'主要驱动因素'，保持语义一致。"
        },
        {
            "angle_name": "keyword_focus",
            "query_text": "北京 2024 营业收入 增长原因",
            "rationale": "提取核心实体'北京'、时间'2024'、指标'营业收入'和意图'增长原因'，去除冗余结构。"
        },
        {
            "angle_name": "structured_condition",
            "query_text": "北京公司2024年度营业收入同比增减变动原因分析",
            "rationale": "补充'年度'和'同比增减变动'等结构化条件，使查询更匹配年报中的正式表述。"
        }
    ]
}
```

---

## 9. 使用示例

```python
from src.pre_retrieval_processing import MetadataFilterBuilder, MultiAngleQueryBuilder

# 功能一：生成元数据过滤条件
filter_builder = MetadataFilterBuilder(provider="openai")

# 未传编码，LLM 推断
metadata_filter = filter_builder.build(
    query="北京公司2024年营收增长原因"
)
# => {"unit_code": {"$eq": "001"}, "year": {"$eq": 2024}}

# 已传编码，直接复用
metadata_filter = filter_builder.build(
    query="上海公司未来三年战略规划",
    unit_code="006"
)
# => {"unit_code": {"$eq": "006"}}

# 功能二：构建多角度查询
angle_builder = MultiAngleQueryBuilder(provider="openai")
angles = angle_builder.build("北京公司2024年营收增长原因")
# => [QueryAngle(...), QueryAngle(...), QueryAngle(...)]

# 调用方按需组合两者结果
result = PreRetrievalResult(
    original_query="北京公司2024年营收增长原因",
    metadata_filter=metadata_filter,
    angles=angles
)
```

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v0.1 | 2026-06-05 | 初始 Spec，引入 SelfQueryRetriever 思路，将功能一从"编码解析"升级为"元数据过滤条件生成"；27家省公司编码使用占位符 |
