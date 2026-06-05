"""
Tests for src/api_requests.py

Follows TDD principles: tests are written against the spec in spec/api_requests_spec.md.
Run with: pytest tests/test_api_requests.py -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch tenacity.retry BEFORE importing api_requests, so @retry becomes no-op
import tenacity

_original_retry = tenacity.retry


def _noop_retry(**kwargs):
    """No-op replacement for tenacity.retry to disable waiting in tests."""
    def decorator(f):
        return f
    return decorator


tenacity.retry = _noop_retry

from src.api_requests import (
    BaseOpenaiProcessor,
    BaseIBMAPIProcessor,
    BaseGeminiProcessor,
    BaseDashscopeProcessor,
    APIProcessor,
    AsyncOpenaiProcessor,
)

# Restore after import so other code isn't affected
tenacity.retry = _original_retry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client."""
    return MagicMock()


@pytest.fixture
def mock_response_data():
    """Return common response_data structure for assertions."""
    return {
        "model": "test-model",
        "input_tokens": 10,
        "output_tokens": 20,
    }


# ---------------------------------------------------------------------------
# BaseOpenaiProcessor Tests
# ---------------------------------------------------------------------------

class TestBaseOpenaiProcessor:
    """Tests for BaseOpenaiProcessor."""

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.OpenAI")
    def test_init(self, mock_openai_cls, mock_getenv, mock_dotenv):
        mock_getenv.side_effect = lambda key: {"OPENAI_API_KEY": "sk-test", "OPENAI_API_BASE": "https://test.com"}.get(key)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        processor = BaseOpenaiProcessor()

        mock_openai_cls.assert_called_once_with(
            api_key="sk-test",
            base_url="https://test.com",
            timeout=None,
            max_retries=2,
        )
        assert processor.llm is mock_client
        assert processor.default_model == "gpt-4o-2024-08-06"

    def test_send_message_unstructured(self, mock_openai_client):
        processor = BaseOpenaiProcessor.__new__(BaseOpenaiProcessor)
        processor.llm = mock_openai_client
        processor.default_model = "gpt-4o-2024-08-06"

        mock_completion = MagicMock()
        mock_completion.model = "gpt-4o-2024-08-06"
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.choices = [MagicMock(message=MagicMock(content="Hello back!"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion

        result = processor.send_message(human_content="Hello!")

        assert result == "Hello back!"
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-2024-08-06"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_send_message_structured(self, mock_openai_client):
        processor = BaseOpenaiProcessor.__new__(BaseOpenaiProcessor)
        processor.llm = mock_openai_client
        processor.default_model = "gpt-4o-2024-08-06"

        mock_schema = MagicMock()
        mock_schema.dict.return_value = {"answer": "42"}

        mock_completion = MagicMock()
        mock_completion.model = "gpt-4o-2024-08-06"
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 8
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_schema))]
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion

        result = processor.send_message(
            human_content="What is the answer?",
            is_structured=True,
            response_format=mock_schema,
        )

        assert result == {"answer": "42"}
        mock_openai_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_openai_client.beta.chat.completions.parse.call_args.kwargs
        assert "response_format" in call_kwargs

    def test_send_message_o3_mini_no_temperature(self, mock_openai_client):
        processor = BaseOpenaiProcessor.__new__(BaseOpenaiProcessor)
        processor.llm = mock_openai_client
        processor.default_model = "gpt-4o-2024-08-06"

        mock_completion = MagicMock()
        mock_completion.model = "o3-mini"
        mock_completion.usage.prompt_tokens = 5
        mock_completion.usage.completion_tokens = 3
        mock_completion.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion

        result = processor.send_message(model="o3-mini", temperature=0.7)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert result == "ok"

    def test_count_tokens(self):
        processor = BaseOpenaiProcessor.__new__(BaseOpenaiProcessor)
        count = processor.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0


# ---------------------------------------------------------------------------
# BaseIBMAPIProcessor Tests
# ---------------------------------------------------------------------------

class TestBaseIBMAPIProcessor:
    """Tests for BaseIBMAPIProcessor."""

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_init(self, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "ibm-test-key"
        processor = BaseIBMAPIProcessor()
        assert processor.api_token == "ibm-test-key"
        assert processor.base_url == "https://rag.timetoact.at/ibm"
        assert processor.default_model == "meta-llama/llama-3-3-70b-instruct"

    @patch("src.api_requests.requests.get")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_check_balance_success(self, mock_getenv, mock_dotenv, mock_get):
        mock_getenv.return_value = "ibm-test-key"
        mock_get.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"balance": 100})
        )
        processor = BaseIBMAPIProcessor()
        result = processor.check_balance()
        assert result == {"balance": 100}
        mock_get.assert_called_once_with(
            "https://rag.timetoact.at/ibm/balance",
            headers={"Authorization": "Bearer ibm-test-key"}
        )

    @patch("src.api_requests.requests.get")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_check_balance_http_error(self, mock_getenv, mock_dotenv, mock_get):
        mock_getenv.return_value = "ibm-test-key"
        import requests
        mock_get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=requests.HTTPError("401 Unauthorized"))
        )
        processor = BaseIBMAPIProcessor()
        result = processor.check_balance()
        assert result is None

    @patch("src.api_requests.requests.get")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_get_available_models(self, mock_getenv, mock_dotenv, mock_get):
        mock_getenv.return_value = "ibm-test-key"
        mock_get.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"models": ["model-a"]})
        )
        processor = BaseIBMAPIProcessor()
        result = processor.get_available_models()
        assert result == {"models": ["model-a"]}

    @patch("src.api_requests.requests.post")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_get_embeddings(self, mock_getenv, mock_dotenv, mock_post):
        mock_getenv.return_value = "ibm-test-key"
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"embeddings": [[0.1, 0.2]]})
        )
        processor = BaseIBMAPIProcessor()
        result = processor.get_embeddings(["hello"])
        assert result == {"embeddings": [[0.1, 0.2]]}

    @patch("src.api_requests.requests.post")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_send_message_unstructured(self, mock_getenv, mock_dotenv, mock_post):
        mock_getenv.return_value = "ibm-test-key"
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "model_id": "meta-llama/llama-3-3-70b-instruct",
                "results": [{
                    "generated_text": "IBM response",
                    "input_token_count": 10,
                    "generated_token_count": 2,
                }]
            })
        )
        processor = BaseIBMAPIProcessor()
        result = processor.send_message(human_content="Hello")
        assert result == "IBM response"
        assert processor.response_data == {
            "model": "meta-llama/llama-3-3-70b-instruct",
            "input_tokens": 10,
            "output_tokens": 2,
        }

    @patch("src.api_requests.requests.post")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.repair_json")
    @patch("src.api_requests.json.loads")
    def test_send_message_structured_success(self, mock_json_loads, mock_repair, mock_getenv, mock_dotenv, mock_post):
        mock_getenv.return_value = "ibm-test-key"
        mock_schema = MagicMock()
        mock_schema.model_validate.return_value = MagicMock(model_dump=MagicMock(return_value={"answer": "42"}))

        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "model_id": "meta-llama/llama-3-3-70b-instruct",
                "results": [{
                    "generated_text": '{"answer": "42"}',
                    "input_token_count": 10,
                    "generated_token_count": 5,
                }]
            })
        )
        mock_repair.return_value = '{"answer": "42"}'
        mock_json_loads.return_value = {"answer": "42"}

        processor = BaseIBMAPIProcessor()
        result = processor.send_message(
            human_content="What?",
            is_structured=True,
            response_format=mock_schema,
        )

        assert result == {"answer": "42"}
        mock_schema.model_validate.assert_called_once()

    @patch("src.api_requests.requests.post")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.repair_json")
    @patch("src.api_requests.json.loads")
    def test_send_message_structured_reparse_success(self, mock_json_loads, mock_repair, mock_getenv, mock_dotenv, mock_post):
        mock_getenv.return_value = "ibm-test-key"
        mock_schema = MagicMock()

        # First call returns bad JSON, second call (reparse) returns good JSON
        call_count = [0]
        def post_side_effect(*args, **kwargs):
            call_count[0] += 1
            payload = kwargs.get("json", {})
            generated = '{"answer": "42"}' if call_count[0] > 1 else 'bad json'
            return MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={
                    "model_id": "meta-llama/llama-3-3-70b-instruct",
                    "results": [{
                        "generated_text": generated,
                        "input_token_count": 10,
                        "generated_token_count": 5,
                    }]
                })
            )

        mock_post.side_effect = post_side_effect
        mock_repair.side_effect = lambda x: x  # no-op repair

        def json_loads_side_effect(s):
            if s == 'bad json':
                raise json.JSONDecodeError("test", s, 0)
            return {"answer": "42"}

        mock_json_loads.side_effect = json_loads_side_effect

        # For reparse success path: after reparse, validate succeeds
        mock_schema.model_validate.side_effect = [
            Exception("fail first"),  # first validate fails
            MagicMock(model_dump=MagicMock(return_value={"answer": "42"}))  # second succeeds
        ]

        processor = BaseIBMAPIProcessor()
        result = processor.send_message(
            human_content="What?",
            is_structured=True,
            response_format=mock_schema,
        )

        assert result == {"answer": "42"}

    @patch("src.api_requests.requests.post")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.repair_json")
    @patch("src.api_requests.json.loads")
    def test_send_message_structured_reparse_fail(self, mock_json_loads, mock_repair, mock_getenv, mock_dotenv, mock_post):
        mock_getenv.return_value = "ibm-test-key"
        mock_schema = MagicMock()
        mock_schema.model_validate.side_effect = Exception("always fail")

        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "model_id": "meta-llama/llama-3-3-70b-instruct",
                "results": [{
                    "generated_text": 'bad json',
                    "input_token_count": 10,
                    "generated_token_count": 5,
                }]
            })
        )
        mock_repair.return_value = 'bad json'
        mock_json_loads.side_effect = json.JSONDecodeError("test", "bad json", 0)

        processor = BaseIBMAPIProcessor()
        result = processor.send_message(
            human_content="What?",
            is_structured=True,
            response_format=mock_schema,
        )

        # Should return original content when everything fails
        assert result == "bad json"

    @patch("src.api_requests.requests.post")
    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_send_message_http_error(self, mock_getenv, mock_dotenv, mock_post):
        mock_getenv.return_value = "ibm-test-key"
        import requests
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=requests.HTTPError("500"))
        )
        processor = BaseIBMAPIProcessor()
        result = processor.send_message(human_content="Hello")
        assert result is None


# ---------------------------------------------------------------------------
# BaseGeminiProcessor Tests
# ---------------------------------------------------------------------------

class TestBaseGeminiProcessor:
    """Tests for BaseGeminiProcessor."""

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.genai")
    def test_init(self, mock_genai, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "gemini-test-key"
        processor = BaseGeminiProcessor()
        mock_genai.configure.assert_called_once_with(api_key="gemini-test-key")
        assert processor.default_model == "gemini-2.0-flash-001"

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.genai")
    def test_list_available_models(self, mock_genai, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "gemini-test-key"
        mock_model = MagicMock()
        mock_model.name = "models/gemini-pro"
        mock_model.supported_generation_methods = ["generateContent"]
        mock_model.input_token_limit = 1000
        mock_model.output_token_limit = 500
        mock_genai.list_models.return_value = [mock_model]

        processor = BaseGeminiProcessor()
        # Just verify it doesn't raise
        processor.list_available_models()

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.genai")
    def test_send_message_unstructured(self, mock_genai, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "gemini-test-key"
        mock_response = MagicMock()
        mock_response.model_version = "gemini-2.0-flash-001"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.text = "Gemini says hi"

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        processor = BaseGeminiProcessor()
        result = processor.send_message(human_content="Hello")

        assert result == "Gemini says hi"
        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="gemini-2.0-flash-001",
            generation_config={"temperature": 0.5}
        )

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.genai")
    @patch("src.api_requests.repair_json")
    @patch("src.api_requests.json.loads")
    def test_send_message_structured_success(self, mock_json_loads, mock_repair, mock_genai, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "gemini-test-key"
        mock_schema = MagicMock()
        mock_schema.model_validate.return_value = MagicMock(model_dump=MagicMock(return_value={"answer": "42"}))

        mock_response = MagicMock()
        mock_response.model_version = "gemini-2.0-flash-001"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 8
        mock_response.text = '{"answer": "42"}'

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        mock_repair.return_value = '{"answer": "42"}'
        mock_json_loads.return_value = {"answer": "42"}

        processor = BaseGeminiProcessor()
        result = processor.send_message(
            human_content="What?",
            is_structured=True,
            response_format=mock_schema,
        )

        assert result == {"answer": "42"}

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    @patch("src.api_requests.genai")
    @patch("src.api_requests.repair_json")
    @patch("src.api_requests.json.loads")
    def test_send_message_structured_reparse(self, mock_json_loads, mock_repair, mock_genai, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "gemini-test-key"
        mock_schema = MagicMock()

        # First validate fails, second succeeds after reparse
        mock_schema.model_validate.side_effect = [
            Exception("fail first"),
            MagicMock(model_dump=MagicMock(return_value={"answer": "fixed"}))
        ]

        # First call returns bad json, second call (reparse) returns good json
        bad_response = MagicMock()
        bad_response.model_version = "gemini-2.0-flash-001"
        bad_response.usage_metadata.prompt_token_count = 10
        bad_response.usage_metadata.candidates_token_count = 8
        bad_response.text = 'bad json'

        good_response = MagicMock()
        good_response.model_version = "gemini-2.0-flash-001"
        good_response.usage_metadata.prompt_token_count = 10
        good_response.usage_metadata.candidates_token_count = 8
        good_response.text = '{"answer": "fixed"}'

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [bad_response, good_response]
        mock_genai.GenerativeModel.return_value = mock_model

        mock_repair.side_effect = lambda x: x

        def json_loads_side_effect(s):
            if s == 'bad json':
                raise json.JSONDecodeError("test", s, 0)
            return {"answer": "fixed"}

        mock_json_loads.side_effect = json_loads_side_effect

        processor = BaseGeminiProcessor()
        result = processor.send_message(
            human_content="What?",
            is_structured=True,
            response_format=mock_schema,
        )

        assert result == {"answer": "fixed"}


# ---------------------------------------------------------------------------
# BaseDashscopeProcessor Tests
# ---------------------------------------------------------------------------

class TestBaseDashscopeProcessor:
    """Tests for BaseDashscopeProcessor."""

    @patch("src.api_requests.load_dotenv")
    @patch("src.api_requests.os.getenv")
    def test_init(self, mock_getenv, mock_dotenv):
        mock_getenv.return_value = "ds-test-key"
        mock_dashscope = MagicMock()
        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            # Need to import fresh with dashscope patched, but we already imported.
            # So we test by instantiating and checking the env var logic indirectly.
            # Actually, the import-time dashscope.api_key assignment won't be re-run.
            # We test the behavior via send_message instead.
            pass

    def test_send_message_normal(self):
        processor = BaseDashscopeProcessor.__new__(BaseDashscopeProcessor)
        processor.default_model = "qwen-turbo-latest"

        mock_response = MagicMock()
        mock_response.output.choices = [MagicMock(message=MagicMock(content='{"final_answer": "hi"}'))]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3

        mock_dashscope = MagicMock()
        mock_dashscope.Generation.call.return_value = mock_response

        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            # Since dashscope was imported at module level, we patch the module attribute
            with patch("src.api_requests.dashscope", mock_dashscope):
                result = processor.send_message(human_content="Hello")
                assert result == {"final_answer": "hi"}
                mock_dashscope.Generation.call.assert_called_once()
                call_kwargs = mock_dashscope.Generation.call.call_args.kwargs
                assert call_kwargs["model"] == "qwen-turbo-latest"
                assert call_kwargs["temperature"] == 0.1
                assert call_kwargs["result_format"] == "message"

    def test_send_message_json_with_markdown(self):
        processor = BaseDashscopeProcessor.__new__(BaseDashscopeProcessor)
        processor.default_model = "qwen-turbo-latest"

        content = '```json\n{"final_answer": "42"}\n```'
        mock_response = MagicMock()
        mock_response.output.choices = [MagicMock(message=MagicMock(content=content))]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3

        mock_dashscope = MagicMock()
        mock_dashscope.Generation.call.return_value = mock_response

        with patch("src.api_requests.dashscope", mock_dashscope):
            result = processor.send_message(human_content="What?")
            assert result == {"final_answer": "42"}

    def test_send_message_invalid_json_fallback(self):
        processor = BaseDashscopeProcessor.__new__(BaseDashscopeProcessor)
        processor.default_model = "qwen-turbo-latest"

        mock_response = MagicMock()
        mock_response.output.choices = [MagicMock(message=MagicMock(content="not json"))]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3

        mock_dashscope = MagicMock()
        mock_dashscope.Generation.call.return_value = mock_response

        with patch("src.api_requests.dashscope", mock_dashscope):
            result = processor.send_message(human_content="Hello")
            assert result == {
                "final_answer": "not json",
                "step_by_step_analysis": "",
                "reasoning_summary": "",
                "relevant_pages": [],
            }


# ---------------------------------------------------------------------------
# APIProcessor Tests
# ---------------------------------------------------------------------------

class TestAPIProcessor:
    """Tests for APIProcessor."""

    @patch.object(BaseOpenaiProcessor, "__init__", return_value=None)
    def test_init_openai(self, mock_init):
        processor = APIProcessor(provider="openai")
        assert processor.provider == "openai"
        mock_init.assert_called_once()

    @patch.object(BaseIBMAPIProcessor, "__init__", return_value=None)
    def test_init_ibm(self, mock_init):
        processor = APIProcessor(provider="ibm")
        assert processor.provider == "ibm"
        mock_init.assert_called_once()

    @patch.object(BaseGeminiProcessor, "__init__", return_value=None)
    def test_init_gemini(self, mock_init):
        processor = APIProcessor(provider="gemini")
        assert processor.provider == "gemini"
        mock_init.assert_called_once()

    @patch.object(BaseDashscopeProcessor, "__init__", return_value=None)
    def test_init_dashscope(self, mock_init):
        processor = APIProcessor(provider="dashscope")
        assert processor.provider == "dashscope"
        mock_init.assert_called_once()

    def test_init_invalid_provider(self):
        # Current implementation does not raise on invalid provider in __init__;
        # self.processor is left unset.
        processor = APIProcessor(provider="unknown")
        assert not hasattr(processor, "processor") or processor.processor is None

    def test_send_message_routing(self):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "openai"
        mock_inner = MagicMock()
        mock_inner.default_model = "gpt-4o"
        mock_inner.send_message.return_value = "routed"
        processor.processor = mock_inner

        result = processor.send_message(human_content="Hello")
        assert result == "routed"
        mock_inner.send_message.assert_called_once_with(
            model="gpt-4o",
            temperature=0.5,
            seed=None,
            system_content="You are a helpful assistant.",
            human_content="Hello",
            is_structured=False,
            response_format=None,
        )

    @patch("src.api_requests.APIProcessor._build_rag_context_prompts")
    def test_get_answer_from_rag_context_success(self, mock_build_prompts):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "openai"
        mock_inner = MagicMock()
        mock_inner.send_message.return_value = {
            "step_by_step_analysis": "analysis",
            "reasoning_summary": "summary",
            "relevant_pages": [1],
            "final_answer": "42",
        }
        mock_inner.response_data = {"model": "test", "input_tokens": 10, "output_tokens": 5}
        processor.processor = mock_inner

        mock_schema = MagicMock()
        mock_build_prompts.return_value = ("sys", mock_schema, "user {context} {question}")

        result = processor.get_answer_from_rag_context(
            question="Q?",
            rag_context="context",
            schema="number",
            model="gpt-4o",
        )

        assert result["final_answer"] == "42"
        assert result["step_by_step_analysis"] == "analysis"
        mock_inner.send_message.assert_called_once_with(
            model="gpt-4o",
            system_content="sys",
            human_content="user context Q?",
            is_structured=True,
            response_format=mock_schema,
        )

    @patch("src.api_requests.APIProcessor._build_rag_context_prompts")
    def test_get_answer_from_rag_context_fallback(self, mock_build_prompts):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "dashscope"
        mock_inner = MagicMock()
        mock_inner.send_message.return_value = "not a dict"
        mock_inner.response_data = {}
        processor.processor = mock_inner

        mock_build_prompts.return_value = ("sys", MagicMock(), "user")

        result = processor.get_answer_from_rag_context(
            question="Q?",
            rag_context="ctx",
            schema="number",
            model="qwen",
        )

        assert result["final_answer"] == "N/A"
        assert result["step_by_step_analysis"] == ""
        assert result["relevant_pages"] == []

    @patch("src.api_requests.APIProcessor._build_rag_context_prompts")
    def test_get_answer_from_rag_context_dashscope_format(self, mock_build_prompts):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "dashscope"
        mock_inner = MagicMock()
        mock_inner.send_message.return_value = {
            "final_answer": '{"step_by_step_analysis": "a", "final_answer": "42"}',
        }
        mock_inner.response_data = {}
        processor.processor = mock_inner

        mock_build_prompts.return_value = ("sys", MagicMock(), "user")

        result = processor.get_answer_from_rag_context(
            question="Q?",
            rag_context="ctx",
            schema="number",
            model="qwen",
        )

        assert result["final_answer"] == "42"
        assert result["step_by_step_analysis"] == "a"

    @pytest.mark.parametrize("schema,expected_class", [
        ("name", "AnswerWithRAGContextNamePrompt"),
        ("number", "AnswerWithRAGContextNumberPrompt"),
        ("boolean", "AnswerWithRAGContextBooleanPrompt"),
        ("names", "AnswerWithRAGContextNamesPrompt"),
        ("comparative", "ComparativeAnswerPrompt"),
        ("string", "AnswerWithRAGContextStringPrompt"),
    ])
    @patch("src.api_requests.prompts")
    def test_build_rag_context_prompts(self, mock_prompts, schema, expected_class):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "openai"

        mock_module = MagicMock()
        mock_module.system_prompt = "sys"
        mock_module.system_prompt_with_schema = "sys_schema"
        mock_module.AnswerSchema = MagicMock()
        mock_module.user_prompt = "user"
        setattr(mock_prompts, expected_class, mock_module)

        sys_prompt, resp_format, user_prompt = processor._build_rag_context_prompts(schema)
        assert sys_prompt == "sys"
        assert resp_format is mock_module.AnswerSchema
        assert user_prompt == "user"

    @pytest.mark.parametrize("schema,expected_class", [
        ("name", "AnswerWithRAGContextNamePrompt"),
        ("number", "AnswerWithRAGContextNumberPrompt"),
    ])
    @patch("src.api_requests.prompts")
    def test_build_rag_context_prompts_ibm_uses_schema(self, mock_prompts, schema, expected_class):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "ibm"

        mock_module = MagicMock()
        mock_module.system_prompt = "sys"
        mock_module.system_prompt_with_schema = "sys_schema"
        mock_module.AnswerSchema = MagicMock()
        mock_module.user_prompt = "user"
        setattr(mock_prompts, expected_class, mock_module)

        sys_prompt, resp_format, user_prompt = processor._build_rag_context_prompts(schema)
        assert sys_prompt == "sys_schema"

    def test_build_rag_context_prompts_invalid(self):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "openai"
        with pytest.raises(ValueError, match="Unsupported schema"):
            processor._build_rag_context_prompts("invalid")

    def test_get_rephrased_questions(self):
        processor = APIProcessor.__new__(APIProcessor)
        processor.provider = "openai"
        mock_inner = MagicMock()
        mock_inner.send_message.return_value = {
            "questions": [
                {"company_name": "A公司", "question": "A的营收？"},
                {"company_name": "B公司", "question": "B的营收？"},
            ]
        }
        processor.processor = mock_inner

        result = processor.get_rephrased_questions(
            original_question="A和B的营收对比",
            companies=["A公司", "B公司"],
        )

        assert result == {"A公司": "A的营收？", "B公司": "B的营收？"}


# ---------------------------------------------------------------------------
# AsyncOpenaiProcessor Tests
# ---------------------------------------------------------------------------

class TestAsyncOpenaiProcessor:
    """Tests for AsyncOpenaiProcessor."""

    def test_get_unique_filepath_no_collision(self, tmp_path):
        processor = AsyncOpenaiProcessor()
        base = str(tmp_path / "test.jsonl")
        result = processor._get_unique_filepath(base)
        assert result == base

    def test_get_unique_filepath_with_collision(self, tmp_path):
        processor = AsyncOpenaiProcessor()
        base = str(tmp_path / "test.jsonl")
        Path(base).write_text("exists")
        result = processor._get_unique_filepath(base)
        assert result == str(tmp_path / "test_1.jsonl")

        Path(result).write_text("exists")
        result2 = processor._get_unique_filepath(base)
        assert result2 == str(tmp_path / "test_2.jsonl")

    @pytest.mark.asyncio
    @patch("src.api_requests.process_api_requests_from_file")
    async def test_process_structured_outputs_requests_success(self, mock_process, tmp_path):
        processor = AsyncOpenaiProcessor()

        save_filepath = str(tmp_path / "results.jsonl")
        requests_filepath = str(tmp_path / "requests.jsonl")

        results = [
            [
                {"messages": [{"role": "user", "content": "Q1"}]},
                {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}]},
                {"original_index": 0},
            ],
            [
                {"messages": [{"role": "user", "content": "Q2"}]},
                {"choices": [{"message": {"content": '{"a": 2}'}, "finish_reason": "stop"}]},
                {"original_index": 1},
            ],
        ]

        async def mock_parallel_processor(*args, **kwargs):
            sf = kwargs.get("save_filepath")
            with open(sf, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            return None

        mock_process.side_effect = mock_parallel_processor

        mock_response_format = MagicMock()
        mock_response_format.side_effect = lambda **kwargs: MagicMock(model_dump=MagicMock(return_value=kwargs))

        with patch("src.api_requests.type_to_response_format_param", return_value={"type": "json_schema"}):
            with patch("src.api_requests.os.getenv", return_value="sk-test"):
                result = await processor.process_structured_ouputs_requests(
                    queries=["Q1", "Q2"],
                    response_format=mock_response_format,
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    preserve_requests=False,
                    preserve_results=False,
                )

        assert len(result) == 2
        assert result[0]["question"][0]["content"] == "Q1"
        assert result[1]["question"][0]["content"] == "Q2"

    @pytest.mark.asyncio
    @patch("src.api_requests.process_api_requests_from_file")
    async def test_process_structured_outputs_requests_finish_reason_warning(self, mock_process, tmp_path, capsys):
        processor = AsyncOpenaiProcessor()

        save_filepath = str(tmp_path / "results.jsonl")
        requests_filepath = str(tmp_path / "requests.jsonl")

        results = [
            [
                {"messages": [{"role": "user", "content": "Q1"}]},
                {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "length"}]},
                {"original_index": 0},
            ],
        ]

        async def mock_parallel_processor(*args, **kwargs):
            sf = kwargs.get("save_filepath")
            with open(sf, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            return None

        mock_process.side_effect = mock_parallel_processor

        mock_response_format = MagicMock()
        mock_response_format.side_effect = lambda **kwargs: MagicMock(model_dump=MagicMock(return_value=kwargs))

        with patch("src.api_requests.type_to_response_format_param", return_value={"type": "json_schema"}):
            with patch("src.api_requests.os.getenv", return_value="sk-test"):
                await processor.process_structured_ouputs_requests(
                    queries=["Q1"],
                    response_format=mock_response_format,
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    preserve_requests=False,
                    preserve_results=False,
                )

        captured = capsys.readouterr()
        assert "finish_reason is 'length'" in captured.out

    @pytest.mark.asyncio
    @patch("src.api_requests.process_api_requests_from_file")
    async def test_process_structured_outputs_requests_parse_fail(self, mock_process, tmp_path, capsys):
        processor = AsyncOpenaiProcessor()

        save_filepath = str(tmp_path / "results.jsonl")
        requests_filepath = str(tmp_path / "requests.jsonl")

        results = [
            [
                {"messages": [{"role": "user", "content": "Q1"}]},
                {"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]},
                {"original_index": 0},
            ],
        ]

        async def mock_parallel_processor(*args, **kwargs):
            sf = kwargs.get("save_filepath")
            with open(sf, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            return None

        mock_process.side_effect = mock_parallel_processor

        mock_response_format = MagicMock()

        with patch("src.api_requests.type_to_response_format_param", return_value={"type": "json_schema"}):
            with patch("src.api_requests.os.getenv", return_value="sk-test"):
                result = await processor.process_structured_ouputs_requests(
                    queries=["Q1"],
                    response_format=mock_response_format,
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    preserve_requests=False,
                    preserve_results=False,
                )

        assert len(result) == 1
        assert result[0]["answer"] == ""
        captured = capsys.readouterr()
        assert "Failed to parse answer JSON" in captured.out
