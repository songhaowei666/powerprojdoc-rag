"""
Tests for src/retrieval.py

Follows TDD principles: tests are written against the spec in spec/retrieval_spec.md.
Run with: pytest tests/test_retrieval.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.ingestion import BM25Ingestor
from src.retrieval import BM25Retriever, HybridRetriever, VectorRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document():
    """Return a minimal valid document dict."""
    return {
        "metainfo": {
            "company_name": "示例科技",
            "sha1": "abc123def456",
            "file_name": "示例科技_2024年报.pdf",
        },
        "content": {
            "chunks": [
                {"page": 1, "text": "chunk A text"},
                {"page": 2, "text": "chunk B text"},
                {"page": 3, "text": "chunk C text"},
            ],
            "pages": [
                {"page": 1, "text": "full page 1 text"},
                {"page": 2, "text": "full page 2 text"},
                {"page": 3, "text": "full page 3 text"},
            ],
        },
    }


@pytest.fixture
def mock_bm25_index():
    """Return a mocked BM25Okapi-like object."""
    bm25 = MagicMock()
    # Simulate scores for 3 chunks
    bm25.get_scores.return_value = np.array([2.5, 1.0, 3.2])
    return bm25


@pytest.fixture
def mock_faiss_index():
    """Return a mocked faiss index."""
    idx = MagicMock()
    # Simulate search returning distances and indices
    # distances shape: (1, k), indices shape: (1, k)
    idx.search.return_value = (
        np.array([[0.1, 0.5]]),   # distances
        np.array([[2, 0]]),       # indices
    )
    return idx


# ---------------------------------------------------------------------------
# BM25Retriever Tests
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    """Tests for BM25Retriever."""

    def test_init(self, tmp_path):
        retriever = BM25Retriever(bm25_db_dir=tmp_path, documents_dir=tmp_path)
        assert retriever.bm25_db_dir == tmp_path
        assert retriever.documents_dir == tmp_path
        assert retriever.index_name == "default"

    def test_init_custom_index_name(self, tmp_path):
        retriever = BM25Retriever(bm25_db_dir=tmp_path, documents_dir=tmp_path, index_name="custom")
        assert retriever.index_name == "custom"

    @patch.object(BM25Ingestor, "search")
    def test_retrieve_success(self, mock_search, tmp_path):
        mock_search.return_value = (
            np.array([2.5, 1.0, 3.2]),
            [
                {"page": 1, "sha1": "sha1"},
                {"page": 2, "sha1": "sha1"},
                {"page": 3, "sha1": "sha1"},
            ],
            ["chunk A text", "chunk B text", "chunk C text"],
        )

        bm25_db_dir = tmp_path / "bm25"
        bm25_db_dir.mkdir()
        (bm25_db_dir / "default.pkl").write_text("dummy")

        retriever = BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=tmp_path)
        results = retriever.retrieve(query="营业收入", top_n=2)

        assert len(results) == 2
        # Scores sorted desc: 3.2 (idx 2), 2.5 (idx 0)
        assert results[0]["page"] == 3
        assert results[0]["distance"] == 3.2
        assert results[0]["text"] == "chunk C text"
        assert results[1]["page"] == 1
        assert results[1]["distance"] == 2.5
        assert results[1]["text"] == "chunk A text"

    @patch.object(BM25Ingestor, "search")
    def test_retrieve_return_parent_pages(self, mock_search, tmp_path):
        # 创建真实文档 JSON，包含 pages，用于 _load_pages_mapping
        doc = {
            "metainfo": {"sha1": "sha1", "company_name": "示例科技"},
            "content": {
                "pages": [
                    {"page": 1, "text": "full page 1 text"},
                    {"page": 2, "text": "full page 2 text"},
                    {"page": 3, "text": "full page 3 text"},
                ]
            },
        }
        (tmp_path / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        mock_search.return_value = (
            np.array([2.5, 1.0, 3.2]),
            [
                {"page": 1, "sha1": "sha1"},
                {"page": 2, "sha1": "sha1"},
                {"page": 3, "sha1": "sha1"},
            ],
            ["chunk A text", "chunk B text", "chunk C text"],
        )

        bm25_db_dir = tmp_path / "bm25"
        bm25_db_dir.mkdir()
        (bm25_db_dir / "default.pkl").write_text("dummy")

        retriever = BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=tmp_path)
        results = retriever.retrieve(query="test", top_n=3, return_parent_pages=True)

        # All chunks map to distinct pages, so 3 results
        assert len(results) == 3
        assert results[0]["text"] == "full page 3 text"
        assert results[1]["text"] == "full page 1 text"

    @patch.object(BM25Ingestor, "search")
    def test_retrieve_top_n_larger_than_chunks(self, mock_search, tmp_path):
        mock_search.return_value = (
            np.array([2.5, 1.0, 3.2]),
            [
                {"page": 1, "sha1": "sha1"},
                {"page": 2, "sha1": "sha1"},
                {"page": 3, "sha1": "sha1"},
            ],
            ["chunk A text", "chunk B text", "chunk C text"],
        )

        bm25_db_dir = tmp_path / "bm25"
        bm25_db_dir.mkdir()
        (bm25_db_dir / "default.pkl").write_text("dummy")

        retriever = BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=tmp_path)
        results = retriever.retrieve(query="test", top_n=100)

        # Should cap at number of chunks (3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# VectorRetriever Tests
# ---------------------------------------------------------------------------

class TestVectorRetriever:
    """Tests for VectorRetriever."""

    # -- Constructor / Setup --

    @patch("src.retrieval.VectorRetriever._load_vectorstore")
    def test_init(self, mock_load_store, tmp_path):
        mock_load_store.return_value = MagicMock()
        retriever = VectorRetriever(
            vector_db_dir=tmp_path, documents_dir=tmp_path, index_name="custom"
        )
        assert retriever.vector_db_dir == tmp_path
        assert retriever.documents_dir == tmp_path
        assert retriever.index_name == "custom"

    @patch("src.retrieval.VectorRetriever._load_vectorstore")
    def test_init_default_index_name(self, mock_load_store, tmp_path):
        mock_load_store.return_value = MagicMock()
        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        assert retriever.index_name == "default"

    # -- _load_pages_mapping --

    def test_load_pages_mapping(self, tmp_path, sample_document):
        (tmp_path / "doc.json").write_text(
            json.dumps(sample_document), encoding="utf-8"
        )
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.documents_dir = tmp_path
        mapping = retriever._load_pages_mapping()
        sha1 = sample_document["metainfo"]["sha1"]
        assert mapping[sha1][1] == "full page 1 text"
        assert mapping[sha1][2] == "full page 2 text"
        assert mapping[sha1][3] == "full page 3 text"

    # -- _find_report_by_company --

    def test_find_report_by_company(self, tmp_path, sample_document):
        (tmp_path / "doc.json").write_text(
            json.dumps(sample_document), encoding="utf-8"
        )
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.documents_dir = tmp_path
        doc = retriever._find_report_by_company("示例科技")
        assert doc["metainfo"]["company_name"] == "示例科技"

    def test_find_report_not_found(self, tmp_path):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.documents_dir = tmp_path
        with pytest.raises(ValueError, match="No report found with '未知公司' company name"):
            retriever._find_report_by_company("未知公司")

    # -- retrieve --

    @patch("src.retrieval.VectorRetriever._load_vectorstore")
    def test_vector_retrieve_success(self, mock_load_store, tmp_path):
        mock_doc1 = Document(
            page_content="chunk C text",
            metadata={"page": 3, "sha1": "sha1", "company_code": "001"},
        )
        mock_doc2 = Document(
            page_content="chunk A text",
            metadata={"page": 1, "sha1": "sha1", "company_code": "001"},
        )
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (mock_doc1, 0.1),
            (mock_doc2, 0.5),
        ]
        mock_load_store.return_value = mock_store

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve(
            company_code="001", query="test", top_n=2
        )

        assert len(results) == 2
        assert results[0]["page"] == 3
        assert results[0]["distance"] == 0.1
        assert results[0]["text"] == "chunk C text"
        assert results[1]["page"] == 1
        assert results[1]["distance"] == 0.5
        assert results[1]["text"] == "chunk A text"
        mock_store.similarity_search_with_score.assert_called_once_with(
            "test", k=2, filter={"company_code": "001"}
        )

    @patch("src.retrieval.VectorRetriever._load_vectorstore")
    def test_vector_retrieve_return_parent_pages(self, mock_load_store, tmp_path, sample_document):
        (tmp_path / "doc.json").write_text(
            json.dumps(sample_document), encoding="utf-8"
        )
        sha1 = sample_document["metainfo"]["sha1"]
        mock_doc1 = Document(
            page_content="chunk C text",
            metadata={"page": 3, "sha1": sha1, "company_name": "示例科技"},
        )
        mock_doc2 = Document(
            page_content="chunk A text",
            metadata={"page": 1, "sha1": sha1, "company_name": "示例科技"},
        )
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (mock_doc1, 0.1),
            (mock_doc2, 0.5),
        ]
        mock_load_store.return_value = mock_store

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve(
            company_code="001", query="test", top_n=2, return_parent_pages=True
        )

        assert len(results) == 2
        assert results[0]["text"] == "full page 3 text"
        assert results[1]["text"] == "full page 1 text"

    # -- get_strings_cosine_similarity --

    @patch("src.retrieval.VectorRetriever.set_up_llm")
    def test_cosine_similarity(self, mock_setup_llm):
        mock_llm = MagicMock()
        mock_llm.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[1.0, 0.0]),
                MagicMock(embedding=[1.0, 0.0]),
            ]
        )
        mock_setup_llm.return_value = mock_llm

        score = VectorRetriever.get_strings_cosine_similarity("a", "b")
        assert score == 1.0


# ---------------------------------------------------------------------------
# HybridRetriever Tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @patch("src.retrieval.VectorRetriever")
    @patch("src.retrieval.LLMReranker")
    def test_init(self, mock_reranker_cls, mock_vector_cls, tmp_path):
        mock_vector_instance = MagicMock()
        mock_vector_cls.return_value = mock_vector_instance
        mock_reranker_instance = MagicMock()
        mock_reranker_cls.return_value = mock_reranker_instance

        retriever = HybridRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)

        mock_vector_cls.assert_called_once_with(tmp_path, tmp_path)
        mock_reranker_cls.assert_called_once_with()
        assert retriever.vector_retriever is mock_vector_instance
        assert retriever.reranker is mock_reranker_instance

    @patch("src.retrieval.VectorRetriever")
    @patch("src.retrieval.LLMReranker")
    def test_retrieve(
        self, mock_reranker_cls, mock_vector_cls, tmp_path
    ):
        mock_vector_instance = MagicMock()
        mock_vector_instance.retrieve.return_value = [
            {"distance": 0.1, "page": 1, "text": "a"},
            {"distance": 0.2, "page": 2, "text": "b"},
        ]
        mock_vector_cls.return_value = mock_vector_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank_documents.return_value = [
            {"distance": 0.1, "page": 1, "text": "a", "combined_score": 0.9},
            {"distance": 0.2, "page": 2, "text": "b", "combined_score": 0.8},
        ]
        mock_reranker_cls.return_value = mock_reranker_instance

        retriever = HybridRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve(
            company_code="示例科技",
            query="test",
            llm_reranking_sample_size=10,
            documents_batch_size=5,
            top_n=2,
            llm_weight=0.7,
        )

        assert len(results) == 2
        mock_vector_instance.retrieve.assert_called_once_with(
            company_code="示例科技",
            query="test",
            top_n=10,
            return_parent_pages=False,
        )
        mock_reranker_instance.rerank_documents.assert_called_once_with(
            query="test",
            documents=mock_vector_instance.retrieve.return_value,
            documents_batch_size=5,
            llm_weight=0.7,
        )

    @patch("src.retrieval.VectorRetriever")
    @patch("src.retrieval.LLMReranker")
    def test_retrieve_top_n_clipping(
        self, mock_reranker_cls, mock_vector_cls, tmp_path
    ):
        mock_vector_instance = MagicMock()
        mock_vector_instance.retrieve.return_value = [
            {"distance": 0.1, "page": 1, "text": "a"},
        ]
        mock_vector_cls.return_value = mock_vector_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank_documents.return_value = [
            {"distance": 0.1, "page": 1, "text": "a", "combined_score": 0.9},
            {"distance": 0.2, "page": 2, "text": "b", "combined_score": 0.8},
        ]
        mock_reranker_cls.return_value = mock_reranker_instance

        retriever = HybridRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve(
            company_code="示例科技", query="test", top_n=1
        )

        # Even though reranker returned 2, we clip to top_n=1
        assert len(results) == 1
        assert results[0]["combined_score"] == 0.9
