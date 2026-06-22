"""RAGApp 编排层单元测试。"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.rag_app import RAGApp


def _make_doc(text: str = "doc") -> Document:
    return Document(page_content=text, metadata={})


class TestRAGApp:
    def test_empty_question_raises(self):
        app = RAGApp(retrieval_app=MagicMock(), generation_app=MagicMock())
        with pytest.raises(ValueError, match="问题不能为空"):
            app.run("")

    def test_invalid_max_rag_rounds_raises(self):
        app = RAGApp(retrieval_app=MagicMock(), generation_app=MagicMock())
        with pytest.raises(ValueError, match="max_rag_rounds"):
            app.run("问题", max_rag_rounds=0)

    def test_success_first_round_no_retry(self):
        retrieval_app = MagicMock()
        retrieval_app.invoke.return_value = {
            "documents": [_make_doc("a")],
            "has_relevant_docs": True,
            "is_direct_retrieve": False,
        }
        generation_app = MagicMock()
        generation_app.invoke.return_value = {
            "documents": [_make_doc("a")],
            "generation": "答案",
            "question": "原始问题",
            "is_grounded_in_docs": True,
            "is_question_answered": True,
            "is_direct_generate": False,
            "should_retry_retrieval": False,
            "failure_reason": "ok",
        }

        app = RAGApp(retrieval_app=retrieval_app, generation_app=generation_app)
        result = app.run("原始问题", company_code="001")

        assert result["question"] == "原始问题"
        assert result["generation"] == "答案"
        assert result["rag_rounds"] == 1
        assert result["should_retry_retrieval"] is False
        assert result["failure_reason"] == "ok"
        retrieval_app.invoke.assert_called_once()
        generation_app.invoke.assert_called_once()

    def test_retry_retrieval_second_round(self):
        retrieval_app = MagicMock()
        retrieval_app.invoke.side_effect = [
            {"documents": [_make_doc("round1")], "has_relevant_docs": False},
            {"documents": [_make_doc("round2")], "has_relevant_docs": True},
        ]
        generation_app = MagicMock()
        generation_app.invoke.side_effect = [
            {
                "documents": [_make_doc("round1")],
                "generation": "差答案",
                "question": "改写后问题",
                "is_grounded_in_docs": False,
                "is_question_answered": False,
                "should_retry_retrieval": True,
                "failure_reason": "hallucination",
            },
            {
                "documents": [_make_doc("round2")],
                "generation": "好答案",
                "question": "改写后问题",
                "is_grounded_in_docs": True,
                "is_question_answered": True,
                "should_retry_retrieval": False,
                "failure_reason": "ok",
            },
        ]

        app = RAGApp(retrieval_app=retrieval_app, generation_app=generation_app)
        result = app.run("原始问题")

        assert result["rag_rounds"] == 2
        assert result["working_question"] == "改写后问题"
        assert result["generation"] == "好答案"
        assert retrieval_app.invoke.call_count == 2
        second_retrieval_inputs = retrieval_app.invoke.call_args_list[1][0][0]
        assert second_retrieval_inputs["question"] == "改写后问题"

    def test_max_rag_rounds_stops_loop(self):
        retrieval_app = MagicMock()
        retrieval_app.invoke.return_value = {
            "documents": [_make_doc()],
            "has_relevant_docs": False,
        }
        generation_app = MagicMock()
        generation_app.invoke.return_value = {
            "documents": [_make_doc()],
            "generation": "仍失败",
            "question": "改写问题",
            "is_grounded_in_docs": False,
            "is_question_answered": False,
            "should_retry_retrieval": True,
            "failure_reason": "hallucination",
        }

        app = RAGApp(retrieval_app=retrieval_app, generation_app=generation_app)
        result = app.run("原始问题", max_rag_rounds=2)

        assert result["rag_rounds"] == 2
        assert result["should_retry_retrieval"] is True
        assert retrieval_app.invoke.call_count == 2
        assert generation_app.invoke.call_count == 2

    def test_direct_generate_single_round(self):
        retrieval_app = MagicMock()
        retrieval_app.invoke.return_value = {"documents": [], "has_relevant_docs": False}
        generation_app = MagicMock()
        generation_app.invoke.return_value = {
            "documents": [],
            "generation": "直接答案",
            "question": "问题",
            "should_retry_retrieval": False,
            "failure_reason": "skipped",
        }

        app = RAGApp(retrieval_app=retrieval_app, generation_app=generation_app)
        result = app.run("问题", is_direct_generate=True, max_rag_rounds=3)

        assert result["rag_rounds"] == 1
        retrieval_app.invoke.assert_called_once()
        generation_app.invoke.assert_called_once()
