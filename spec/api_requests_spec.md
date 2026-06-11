# API Requests Module Spec

> **同步声明**：本文档严格反向推导自 `src/api_requests.py` 与 `src/api_request_parallel_processor.py` 当前实现，用于后续代码修改时保持行为一致。若代码实现变更，必须同步更新本文档。

---

## 1. 概述

`api_requests.py` 是项目内所有大语言模型（LLM）API 调用的统一封装层。它屏蔽了不同厂商（OpenAI、IBM、Google Gemini、阿里云 DashScope）的 API 差异，为上层业务提供一致的对话接口，并额外封装了 RAG（检索增强生成）场景的专用方法。

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| 多厂商统一接入 | 支持 OpenAI、IBM、Gemini、DashScope 四家 LLM 提供商，切换仅需修改 `provider` 参数 |
| 结构化输出支持 | 除 DashScope 外，均支持通过 Pydantic Schema 约束模型返回 JSON 格式 |
| 错误自动修复 | IBM 与 Gemini 处理器在结构化输出解析失败时，自动调用 LLM 重新解析 |
| 异步批量处理 | 提供 `AsyncOpenaiProcessor` 支持基于 JSONL 的批量异步请求，带速率限制 |
| RAG 场景封装 | 内置 `get_answer_from_rag_context` 与 `get_rephrased_questions`，减少上层重复代码 |

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        APIProcessor                          │
│                   (统一路由 / RAG 封装)                        │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
           ▼              ▼              ▼                    ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ BaseOpenai      │ │ BaseIBMAPI   │ │ BaseGemini      │ │ BaseDashscope   │
│ Processor       │ │ Processor    │ │ Processor       │ │ Processor       │
│ (LangChain)     │ │ (IBM RAG)    │ │ (Google Gemini) │ │ (DashScope)     │
└─────────────────┘ └──────────────┘ └─────────────────┘ └─────────────────┘
                                                          
┌─────────────────────────────────────────────────────────────┐
│                  AsyncOpenaiProcessor                        │
│              (JSONL 批量异步 + 速率限制)                       │
└─────────────────────────────────────────────────────────────┘
```

- **`APIProcessor`**：对外统一入口，根据 `provider` 参数自动实例化底层处理器。
- **`BaseOpenaiProcessor`**：封装 LangChain `ChatOpenAI`，支持 `invoke` 与 `with_structured_output`（结构化输出）。配置来自 `src.config.settings`。
- **`BaseIBMAPIProcessor`**：基于 `requests` 直接调用 IBM 私有 REST API，额外支持余额查询、模型列表、Embedding。
- **`BaseGeminiProcessor`**：基于 `google.generativeai`（当前代码中 import 被注释，实际可能不可用），内置重试机制（3 次，间隔 20 秒）。
- **`BaseDashscopeProcessor`**：基于 `dashscope.Generation.call`，面向阿里云通义千问系列。
- **`AsyncOpenaiProcessor`**：基于 `api_request_parallel_processor`，将批量请求写入 JSONL 后并行消费。

---

## 4. 依赖清单

```
python >= 3.10
langchain-openai
langchain-core
openai
requests
google-generativeai  # 当前代码中 import 被注释
dashscope
tiktoken
pydantic
tenacity
json_repair
pydantic-settings
```

**内部依赖**：
- `src.api_request_parallel_processor`：`AsyncOpenaiProcessor` 使用的并行请求处理器
- `src.prompts`：RAG 场景的 system prompt、user prompt、结构化 Schema 定义
- `src.config.settings`：`BaseOpenaiProcessor` 读取 API 密钥与模型配置

---

## 5. 环境变量 / 配置

| 变量名 | 用途 | 使用方 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI 鉴权 | `BaseOpenaiProcessor`, `AsyncOpenaiProcessor`（通过 `settings`） |
| `OPENAI_API_BASE` | OpenAI 自定义 Base URL（可选） | `BaseOpenaiProcessor`, `AsyncOpenaiProcessor`（通过 `settings`） |
| `CHAT_MODEL` | 默认对话模型 | `BaseOpenaiProcessor`, `APIProcessor`（通过 `settings.chat_model`） |
| `IBM_API_KEY` | IBM API 鉴权 | `BaseIBMAPIProcessor` |
| `GEMINI_API_KEY` | Google Gemini 鉴权 | `BaseGeminiProcessor` |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope 鉴权 | `BaseDashscopeProcessor` |

> OpenAI 相关配置通过 `src.config.settings`（pydantic-settings）读取，不再直接调用 `load_dotenv()`。

---

## 6. 类接口定义

### 6.1 BaseOpenaiProcessor

```python
class BaseOpenaiProcessor:
    def __init__(self)
    
    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
    ) -> Union[str, dict]
    
    @staticmethod
    def count_tokens(string, encoding_name="o200k_base") -> int
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `settings.chat_model` | 模型 ID |
| `temperature` | `float` | `0.5` | 采样温度；**当前代码固定传入 `ChatOpenAI`** |
| `seed` | `int` | `None` | 兼容性参数，当前未实际使用 |
| `system_content` | `str` | 默认提示 | system 角色内容 |
| `human_content` | `str` | `"Hello!"` | user 角色内容 |
| `is_structured` | `bool` | `False` | 是否启用结构化输出 |
| `response_format` | `Type[BaseModel]` | `None` | Pydantic Schema，仅在 `is_structured=True` 时生效 |

**实现细节**：
- `__init__` 从 `src.config.settings` 读取 `chat_model`, `openai_api_key`, `openai_api_base`，初始化 `ChatOpenAI(model=..., api_key=..., base_url=..., timeout=5, max_retries=2, temperature=0.5)`
- 非结构化输出：调用 `llm.invoke([SystemMessage, HumanMessage])`，返回 `response.content`
- 结构化输出：调用 `llm.with_structured_output(response_format).invoke(...)`，返回 `response.model_dump()`
- `self.response_data` 记录 `model`, `input_tokens`, `output_tokens`（结构化输出时 usage 为空字典）

**Token 统计**：
- 使用 `tiktoken` 的 `o200k_base` 编码器统计字符串 token 数。

---

### 6.2 BaseIBMAPIProcessor

```python
class BaseIBMAPIProcessor:
    def __init__(self)
    
    def check_balance(self) -> Optional[dict]
    
    def get_available_models(self) -> Optional[dict]
    
    def get_embeddings(
        self,
        texts: List[str],
        model_id="ibm/granite-embedding-278m-multilingual"
    ) -> Optional[dict]
    
    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ) -> Union[str, dict, None]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `meta-llama/llama-3-3-70b-instruct` | IBM 模型 ID |
| `max_new_tokens` | `int` | `5000` | 最大生成 token 数 |
| `min_new_tokens` | `int` | `1` | 最小生成 token 数 |
| `**kwargs` | — | — | 透传至 IBM API `parameters` 字段 |

**结构化输出处理流程**：
1. 先尝试 `json_repair.repair_json` + `json.loads` + `response_format.model_validate`
2. 若失败，调用 `_reparse_response()`：将原始响应与 system prompt 一起重新喂给 LLM，要求修复 JSON
3. 重解析后再尝试验证，若仍失败则返回原始字符串或解析后的字典

**IBM API 端点**（`base_url = https://rag.timetoact.at/ibm`）：
- `GET /balance` — 余额查询
- `GET /foundation_model_specs` — 可用模型列表
- `POST /embeddings` — 文本嵌入
- `POST /text_generation` — 文本生成

---

### 6.3 BaseGeminiProcessor

```python
class BaseGeminiProcessor:
    def __init__(self)
    
    def list_available_models(self) -> None
    
    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=12345,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None
    ) -> Union[str, dict, None]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `gemini-2.0-flash-001` | Gemini 模型 ID |
| `seed` | `int` | `12345` | 兼容性参数，当前未实际使用 |

> **注意**：当前代码中 `import google.generativeai as genai` 被注释，但 `_set_up_llm` 仍调用 `genai.configure`，运行时可能抛出 `NameError`。

**重试策略**：
```python
@retry(wait=wait_fixed(20), stop=stop_after_attempt(3), before_sleep=_log_retry_attempt)
def _generate_with_retry(...)
```
- 失败时等待 20 秒后重试，最多 3 次尝试
- 每次重试前打印异常信息

**Prompt 拼接方式**：
- Gemini 不区分 system/user 角色，代码中将两者拼接为：
  ```
  {system_content}

  ---

  {human_content}
  ```

---

### 6.4 BaseDashscopeProcessor

```python
class BaseDashscopeProcessor:
    def __init__(self)
    
    def send_message(
        self,
        model="qwen-turbo-latest",
        temperature=0.1,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ) -> Union[dict, str]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `qwen-turbo-latest` | DashScope 模型 ID |
| `temperature` | `float` | `0.1` | 采样温度 |

**特性**：
- 暂**不支持**原生结构化输出（`is_structured` 与 `response_format` 被接收但无实际作用）
- 返回结果**始终尝试解析为 JSON**：
  1. 自动去除 Markdown 代码块标记（` ```json...``` `）
  2. 调用 `json.loads` 解析
  3. 若解析成功，返回解析后的字典
  4. 若解析失败，返回兜底字典：`{"final_answer": content, "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": []}`

**响应数据**：
- `self.response_data = {"model": model, "input_tokens": ..., "output_tokens": ...}`

---

### 6.5 APIProcessor

```python
class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "dashscope"] = "dashscope")
    
    def send_message(self, model=None, temperature=0.5, seed=None,
                     system_content="...", human_content="...",
                     is_structured=False, response_format=None, **kwargs)
    
    def get_answer_from_rag_context(
        self,
        question: str,
        rag_context: str,
        schema: str,
        model: str
    ) -> dict
    
    def get_rephrased_questions(
        self,
        original_question: str,
        companies: List[str]
    ) -> Dict[str, str]
```

#### 6.5.1 send_message

统一路由方法。特别地，当 `model=None` 时，从 `settings.chat_model` 读取默认模型，再透传至底层处理器。

#### 6.5.2 get_answer_from_rag_context

**RAG 问答专用方法**，根据 `schema` 选择不同的 Prompt 与输出结构：

| schema | 用途 | 结构化 Schema |
|--------|------|--------------|
| `"name"` | 单一人名/实体 | `AnswerWithRAGContextNamePrompt.AnswerSchema` |
| `"number"` | 数值型答案 | `AnswerWithRAGContextNumberPrompt.AnswerSchema` |
| `"boolean"` | 是/否判断 | `AnswerWithRAGContextBooleanPrompt.AnswerSchema` |
| `"names"` | 多人名/实体列表 | `AnswerWithRAGContextNamesPrompt.AnswerSchema` |
| `"comparative"` | 对比分析 | `ComparativeAnswerPrompt.AnswerSchema` |
| `"string"` | 开放性文本 | `AnswerWithRAGContextStringPrompt.AnswerSchema` |

**Prompt 选择策略**：
- IBM 与 Gemini 使用带 Schema 描述的 system prompt（`system_prompt_with_schema`）
- OpenAI 与 DashScope 使用不带 Schema 描述的 system prompt（`system_prompt`）

**返回结构兜底**：
- 若返回非字典或缺少 `step_by_step_analysis` 字段，代码会尝试从 `final_answer` 中提取 JSON
- 最终兜底为：
  ```python
  {"step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": [], "final_answer": "N/A"}
  ```

#### 6.5.3 get_rephrased_questions

将对比类问题（如"A 公司和 B 公司的营收对比"）拆分为针对每个公司的独立子问题。

**返回格式**：
```python
{"A公司": "A公司的营收是多少？", "B公司": "B公司的营收是多少？"}
```

---

### 6.6 AsyncOpenaiProcessor

```python
class AsyncOpenaiProcessor:
    def _get_unique_filepath(self, base_filepath: str) -> str
    
    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ) -> List[dict]
```

**批量处理流程**：
1. 将 `queries` 列表中的每个查询封装为 JSONL 请求对象（含 `response_format` 与 `metadata.original_index`）
2. 写入 `requests_filepath`（自动处理文件名冲突）
3. 调用 `process_api_requests_from_file` 并行消费，同时启动 `monitor_progress` 协程监控进度
4. 读取 `save_filepath`，按 `original_index` 排序后返回

**结果校验**：
- 检查 `finish_reason`，非 `"stop"` 时打印警告
- 尝试将 `message.content` 解析为 JSON 并用 `response_format` 验证；失败则 `answer=""`

---

## 7. 异常与边界行为

| 场景 | 行为 |
|------|------|
| OpenAI：结构化输出 Schema 不匹配 | 由 LangChain / OpenAI SDK 在服务端校验，失败时抛出 API 异常 |
| OpenAI：`model=None` | 从 `settings.chat_model` 读取默认值 |
| IBM：结构化输出 JSON 解析失败 | 自动调用 `_reparse_response` 重试一次；仍失败则返回原始字符串 |
| IBM：API 返回 HTTP 错误 | 打印错误日志，返回 `None` |
| Gemini：API 调用失败 | 触发 tenacity 重试，3 次后抛出 `Exception` |
| Gemini：JSON 修复失败 | 与 IBM 类似，自动重解析一次 |
| Gemini：`google.generativeai` 未导入 | 因 import 被注释，初始化时抛出 `NameError` |
| DashScope：返回非 JSON 字符串 | 自动返回兜底字典，不会抛异常 |
| DashScope：`DASHSCOPE_API_KEY` 未设置 | `dashscope.api_key = None`，后续调用由 SDK 抛错 |
| AsyncOpenai：`queries` 为空 | 生成空 JSONL，处理结果为空列表 |
| AsyncOpenai：某条结果解析失败 | 该条 `answer=""`，继续处理其余请求 |

---

## 8. 当前实现缺陷（需后续优化）

1. **DashScope 不支持结构化输出**：`is_structured` 与 `response_format` 参数被接收但无实际作用，仅靠正则/JSON 解析兜底
2. **DashScope JSON 解析过于宽松**：使用字符串匹配去除 Markdown 代码块，鲁棒性有限
3. **IBM 与 Gemini 重解析耦合**：`_reparse_response` 依赖 `prompts.AnswerSchemaFixPrompt`，但重解析时仍使用同一模型，若模型本身生成质量差则陷入循环
4. **BaseDashscopeProcessor 在 APIProcessor 之后定义**：类定义顺序上，`APIProcessor.__init__` 中引用 `BaseDashscopeProcessor` 时该类尚未定义（运行时 Python 能处理，但不符合编码规范）
5. **AsyncOpenaiProcessor 仅支持 OpenAI**：异步批量能力未覆盖 IBM、Gemini、DashScope
6. **Hardcoded 参数过多**：
   - 重试次数、间隔时间
   - IBM 的 `max_new_tokens=5000`
   - 各处理器的默认模型
7. **Gemini 的 `seed` 参数未实际使用**：仅作兼容性占位
8. **异常处理不一致**：OpenAI 直接抛 SDK 异常；IBM 捕获后打印并返回 `None`；Gemini 重试后抛异常；DashScope 几乎不抛异常
9. **BaseOpenaiProcessor 的 temperature 未动态调整**：代码中 `o3-mini` 剔除 temperature 的逻辑被注释，当前固定传入
10. **Gemini import 被注释**：`google.generativeai` 导入被注释，但类实现仍依赖 `genai`，导致运行时不可用
11. **代码末尾的 `MetadataFilterResponse2`**：未在任何地方使用，属于残留代码

---

## 9. 使用示例

### 9.1 统一路由调用

```python
from src.api_requests import APIProcessor

# OpenAI
processor = APIProcessor(provider="openai")
answer = processor.send_message(
    model="gpt-4o-2024-08-06",
    system_content="You are a helpful assistant.",
    human_content="What is the capital of France?"
)

# DashScope（默认）
processor = APIProcessor(provider="dashscope")
answer = processor.send_message(
    model="qwen-turbo-latest",
    human_content="你好"
)
```

### 9.2 结构化输出

```python
from pydantic import BaseModel
from src.api_requests import APIProcessor

class AnswerSchema(BaseModel):
    step_by_step_analysis: str
    final_answer: str

processor = APIProcessor(provider="openai")
result = processor.send_message(
    system_content="Analyze the question step by step.",
    human_content="What is 2 + 2?",
    is_structured=True,
    response_format=AnswerSchema
)
# result -> {"step_by_step_analysis": "...", "final_answer": "4"}
```

### 9.3 RAG 问答

```python
from src.api_requests import APIProcessor

processor = APIProcessor(provider="gemini")
answer = processor.get_answer_from_rag_context(
    question="公司营收是多少？",
    rag_context="根据 2024 年报，公司营收为 100 亿元...",
    schema="number",
    model="gemini-2.0-flash-001"
)
# answer -> {"step_by_step_analysis": "...", "final_answer": "100", "reasoning_summary": "...", "relevant_pages": [...]}
```

### 9.4 异步批量请求

```python
import asyncio
from src.api_requests import AsyncOpenaiProcessor

async def main():
    processor = AsyncOpenaiProcessor()
    results = await processor.process_structured_ouputs_requests(
        queries=["What is AI?", "Explain ML."],
        response_format=AnswerSchema,
        max_requests_per_minute=100
    )
    # results -> [{"question": [...], "answer": {...}}, ...]

asyncio.run(main())
```

### 9.5 IBM Embedding

```python
from src.api_requests import BaseIBMAPIProcessor

ibm = BaseIBMAPIProcessor()
embeddings = ibm.get_embeddings(
    texts=["hello world", "test sentence"],
    model_id="ibm/granite-embedding-278m-multilingual"
)
```

---

## 10. 版本记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2024-XX-XX | 初始实现，支持 OpenAI / IBM / Gemini / DashScope 四家 LLM 提供商统一接入 |
| v1.1 | 2026-06-11 | BaseOpenaiProcessor 迁移至 LangChain ChatOpenAI；配置来源改为 `src.config.settings`；同步当前代码状态 |
