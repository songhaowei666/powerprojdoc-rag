"""
Tests for src/evaluation.py

Follows TDD principles: tests are written against the spec in spec/evaluation_spec.md.
Run with: pytest tests/test_evaluation.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    EvalDataset,
    RAGEvaluator,
    SingleTurnEvaluator,
    compute_page_precision_at_k,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_eval_json(tmp_path):
    """Write a sample evaluation JSON and return its path."""
    data = [
        {
            "question": "Q1",
            "expected_answer": "A1",
            "expected_source_doc": "doc1",
            "expected_source_pages": [1, 2],
            "company_name": "示例科技",
        },
        {
            "question": "Q2",
            "expected_answer": "A2",
            "expected_source_doc": "doc2",
            "expected_source_pages": [5],
            "company_name": "示例科技",
        },
    ]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def mock_ragas_result():
    """Return a mock ragas Result whose to_pandas() yields a DataFrame."""
    class MockResult:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "context_precision": [0.8, 0.7],
                    "context_recall": [0.7, 0.6],
                    "faithfulness": [0.9, 0.85],
                    "answer_relevancy": [0.85, 0.8],
                    "answer_correctness": [0.75, 0.7],
                }
            )

    return MockResult()


@pytest.fixture
def mock_llm():
    """Return a minimal mock LLM for evaluator injection."""
    return MagicMock()


# ---------------------------------------------------------------------------
# EvalDataset Tests
# ---------------------------------------------------------------------------

class TestEvalDataset:
    """Tests for EvalDataset."""

    def test_from_json(self, sample_eval_json):
        dataset = EvalDataset.from_json(sample_eval_json)
        assert len(dataset) == 2
        assert dataset[0]["question"] == "Q1"
        assert dataset[1]["expected_source_pages"] == [5]

    def test_to_list_returns_copy(self, sample_eval_json):
        dataset = EvalDataset.from_json(sample_eval_json)
        lst = dataset.to_list()
        assert len(lst) == 2
        lst.pop()
        assert len(dataset) == 2

    def test_getitem_index_error(self, sample_eval_json):
        dataset = EvalDataset.from_json(sample_eval_json)
        with pytest.raises(IndexError):
            _ = dataset[10]


# ---------------------------------------------------------------------------
# compute_page_precision_at_k Tests
# ---------------------------------------------------------------------------

class TestComputePagePrecisionAtK:
    """Tests for compute_page_precision_at_k."""

    def test_full_hit(self):
        docs = [
            Document(page_content="p1", metadata={"page": 1}),
            Document(page_content="p2", metadata={"page": 2}),
        ]
        score = compute_page_precision_at_k(docs, [1, 2], top_k=2)
        assert score == 1.0

    def test_partial_hit(self):
        docs = [
            Document(page_content="p1", metadata={"page": 1}),
            Document(page_content="p3", metadata={"page": 3}),
        ]
        score = compute_page_precision_at_k(docs, [1, 2], top_k=2)
        assert score == 0.5

    def test_no_hit(self):
        docs = [
            Document(page_content="p3", metadata={"page": 3}),
        ]
        score = compute_page_precision_at_k(docs, [1, 2], top_k=2)
        assert score == 0.0

    def test_empty_expected_pages_returns_none(self):
        docs = [Document(page_content="p1", metadata={"page": 1})]
        score = compute_page_precision_at_k(docs, [], top_k=2)
        assert score is None

    def test_empty_documents(self):
        score = compute_page_precision_at_k([], [1], top_k=2)
        assert score == 0.0

    def test_missing_page_key(self):
        docs = [
            Document(page_content="p1", metadata={}),
            Document(page_content="p2", metadata={"page": 2}),
        ]
        score = compute_page_precision_at_k(docs, [2], top_k=2)
        assert score == 0.5

    def test_top_k_larger_than_docs(self):
        docs = [
            Document(page_content="p1", metadata={"page": 1}),
        ]
        score = compute_page_precision_at_k(docs, [1, 2], top_k=3)
        # 1 hit out of 3 positions
        assert score == pytest.approx(1 / 3)

    def test_top_k_zero(self):
        docs = [Document(page_content="p1", metadata={"page": 1})]
        score = compute_page_precision_at_k(docs, [1], top_k=0)
        assert score == 0.0


# ---------------------------------------------------------------------------
# SingleTurnEvaluator Tests
# ---------------------------------------------------------------------------

class TestSingleTurnEvaluator:
    """Tests for SingleTurnEvaluator."""

    @patch("src.evaluation.evaluate")
    def test_evaluate_returns_expected_keys(self, mock_evaluate, mock_ragas_result, mock_llm):
        mock_evaluate.return_value = mock_ragas_result

        evaluator = SingleTurnEvaluator(evaluator_llm=mock_llm)
        docs = [Document(page_content="ctx", metadata={"page": 1})]
        result = evaluator.evaluate(
            question="Q",
            generation="A",
            documents=docs,
            expected_answer="EA",
            expected_pages=[1],
            top_k=2,
        )

        assert result["page_precision@k"] == 0.5
        assert result["context_precision"] == 0.8
        assert result["context_recall"] == 0.7
        assert result["faithfulness"] == 0.9
        assert result["answer_relevancy"] == 0.85
        assert result["answer_correctness"] == 0.75

    @patch("src.evaluation.evaluate")
    def test_evaluate_no_expected_pages(self, mock_evaluate, mock_ragas_result, mock_llm):
        mock_evaluate.return_value = mock_ragas_result

        evaluator = SingleTurnEvaluator(evaluator_llm=mock_llm)
        result = evaluator.evaluate(
            question="Q",
            generation="A",
            documents=[Document(page_content="ctx")],
        )

        assert result["page_precision@k"] is None

    @patch("src.evaluation.evaluate")
    def test_evaluate_ragas_failure(self, mock_evaluate, mock_llm):
        mock_evaluate.side_effect = RuntimeError("ragas error")

        evaluator = SingleTurnEvaluator(evaluator_llm=mock_llm)
        with pytest.raises(RuntimeError, match="ragas error"):
            evaluator.evaluate(
                question="Q",
                generation="A",
                documents=[Document(page_content="ctx")],
                expected_answer="EA",
            )


# ---------------------------------------------------------------------------
# RAGEvaluator Tests
# ---------------------------------------------------------------------------

class TestRAGEvaluator:
    """Tests for RAGEvaluator."""

    @patch("src.evaluation.evaluate")
    def test_run_batch_returns_dataframe(self, mock_evaluate, mock_ragas_result, mock_llm, sample_eval_json):
        mock_evaluate.return_value = mock_ragas_result

        mock_rag_app = MagicMock()
        mock_rag_app.run.side_effect = [
            {
                "question": "Q1",
                "generation": "G1",
                "documents": [
                    Document(page_content="c1", metadata={"page": 1}),
                    Document(page_content="c2", metadata={"page": 3}),
                ],
            },
            {
                "question": "Q2",
                "generation": "G2",
                "documents": [
                    Document(page_content="c3", metadata={"page": 5}),
                ],
            },
        ]

        dataset = EvalDataset.from_json(sample_eval_json)
        evaluator = RAGEvaluator(rag_app=mock_rag_app, evaluator_llm=mock_llm)
        df = evaluator.run_batch(dataset, top_k=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "page_precision@k" in df.columns
        assert "faithfulness" in df.columns
        # Sample 0: pages [1,3] vs expected [1,2] -> 1 hit / 2 = 0.5
        assert df.loc[0, "page_precision@k"] == 0.5
        # Sample 1: pages [5] vs expected [5] -> 1 hit / 2 = 0.5 (top_k=2, only 1 doc)
        assert df.loc[1, "page_precision@k"] == 0.5
        mock_rag_app.run.assert_any_call("Q1", "示例科技")
        mock_rag_app.run.assert_any_call("Q2", "示例科技")

    @patch("src.evaluation.evaluate")
    def test_run_batch_rag_failure_continues(self, mock_evaluate, mock_ragas_result, mock_llm, sample_eval_json):
        mock_evaluate.return_value = mock_ragas_result

        mock_rag_app = MagicMock()
        mock_rag_app.run.side_effect = [
            RuntimeError("检索失败"),
            {
                "question": "Q2",
                "generation": "G2",
                "documents": [Document(page_content="c3", metadata={"page": 5})],
            },
        ]

        dataset = EvalDataset.from_json(sample_eval_json)
        evaluator = RAGEvaluator(rag_app=mock_rag_app, evaluator_llm=mock_llm)
        df = evaluator.run_batch(dataset, top_k=2)

        assert len(df) == 2
        # First sample failed: empty docs -> 0.0
        assert df.loc[0, "page_precision@k"] == 0.0
        assert pd.isna(df.loc[0, "faithfulness"])
        # Second sample succeeded
        assert df.loc[1, "page_precision@k"] == 0.5
        assert df.loc[1, "faithfulness"] == 0.85

    @patch("src.evaluation.evaluate")
    def test_run_batch_ragas_failure_raises(self, mock_evaluate, mock_llm, sample_eval_json):
        mock_evaluate.side_effect = RuntimeError("ragas 失败")

        mock_rag_app = MagicMock()
        mock_rag_app.run.return_value = {
            "question": "Q",
            "generation": "G",
            "documents": [],
        }

        dataset = EvalDataset.from_json(sample_eval_json)
        evaluator = RAGEvaluator(rag_app=mock_rag_app, evaluator_llm=mock_llm)
        with pytest.raises(RuntimeError, match="ragas 失败"):
            evaluator.run_batch(dataset)
