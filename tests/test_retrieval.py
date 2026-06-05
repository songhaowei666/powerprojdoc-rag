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

    @patch("src.retrieval.pickle.load")
    def test_retrieve_by_company_name_success(self, mock_pickle_load, tmp_path, sample_document, mock_bm25_index):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text(json.dumps(sample_document), encoding="utf-8")

        bm25_db_dir = tmp_path / "bm25"
        bm25_db_dir.mkdir()
        bm25_path = bm25_db_dir / f"{sample_document['metainfo']['sha1']}.pkl"
        bm25_path.write_text("dummy pickle data")
        mock_pickle_load.return_value = mock_bm25_index

        retriever = BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=tmp_path)
        results = retriever.retrieve_by_company_name(
            company_name="示例科技", query="营业收入", top_n=2
        )

        assert len(results) == 2
        # Scores sorted desc: 3.2 (idx 2), 2.5 (idx 0)
        assert results[0]["page"] == 3
        assert results[0]["distance"] == 3.2
        assert results[0]["text"] == "chunk C text"
        assert results[1]["page"] == 1
        assert results[1]["distance"] == 2.5
        assert results[1]["text"] == "chunk A text"

    def test_retrieve_company_not_found(self, tmp_path):
        retriever = BM25Retriever(bm25_db_dir=tmp_path, documents_dir=tmp_path)
        with pytest.raises(ValueError, match="No report found with '未知公司' company name"):
            retriever.retrieve_by_company_name(company_name="未知公司", query="test")

    @patch("src.retrieval.pickle.load")
    def test_retrieve_return_parent_pages(self, mock_pickle_load, tmp_path, sample_document, mock_bm25_index):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text(json.dumps(sample_document), encoding="utf-8")

        bm25_db_dir = tmp_path / "bm25"
        bm25_db_dir.mkdir()
        bm25_path = bm25_db_dir / f"{sample_document['metainfo']['sha1']}.pkl"
        bm25_path.write_text("dummy")
        mock_pickle_load.return_value = mock_bm25_index

        retriever = BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=tmp_path)
        results = retriever.retrieve_by_company_name(
            company_name="示例科技", query="test", top_n=3, return_parent_pages=True
        )

        # All chunks map to distinct pages, so 3 results
        assert len(results) == 3
        assert results[0]["text"] == "full page 3 text"
        assert results[1]["text"] == "full page 1 text"

    @patch("src.retrieval.pickle.load")
    def test_retrieve_top_n_larger_than_chunks(self, mock_pickle_load, tmp_path, sample_document, mock_bm25_index):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text(json.dumps(sample_document), encoding="utf-8")

        bm25_db_dir = tmp_path / "bm25"
        bm25_db_dir.mkdir()
        bm25_path = bm25_db_dir / f"{sample_document['metainfo']['sha1']}.pkl"
        bm25_path.write_text("dummy")
        mock_pickle_load.return_value = mock_bm25_index

        retriever = BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=tmp_path)
        results = retriever.retrieve_by_company_name(
            company_name="示例科技", query="test", top_n=100
        )

        # Should cap at number of chunks (3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# VectorRetriever Tests
# ---------------------------------------------------------------------------

class TestVectorRetriever:
    """Tests for VectorRetriever."""

    # -- Constructor / Setup --

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    def test_init_default_provider(self, mock_setup_llm, mock_load_dbs, tmp_path):
        mock_load_dbs.return_value = []
        mock_setup_llm.return_value = None
        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        assert retriever.embedding_provider == "dashscope"
        mock_load_dbs.assert_called_once()
        mock_setup_llm.assert_called_once()

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    def test_init_openai_provider(self, mock_setup_llm, mock_load_dbs, tmp_path):
        mock_load_dbs.return_value = []
        mock_setup_llm.return_value = MagicMock()
        retriever = VectorRetriever(
            vector_db_dir=tmp_path, documents_dir=tmp_path, embedding_provider="OpenAI"
        )
        assert retriever.embedding_provider == "openai"

    # -- _set_up_llm --

    @patch("src.retrieval.load_dotenv")
    @patch("src.retrieval.os.getenv")
    @patch("src.retrieval.OpenAI")
    def test_setup_llm_openai(self, mock_openai_cls, mock_getenv, mock_dotenv, tmp_path):
        mock_getenv.return_value = "sk-test"
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "openai"
        llm = retriever._set_up_llm()

        mock_openai_cls.assert_called_once_with(
            api_key="sk-test", timeout=None, max_retries=2
        )
        assert llm is mock_client

    @patch("src.retrieval.load_dotenv")
    @patch("src.retrieval.os.getenv")
    def test_setup_llm_dashscope(self, mock_getenv, mock_dotenv, tmp_path):
        mock_getenv.return_value = "ds-test"
        mock_dashscope = MagicMock()
        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            retriever = VectorRetriever.__new__(VectorRetriever)
            retriever.embedding_provider = "dashscope"
            llm = retriever._set_up_llm()
            assert mock_dashscope.api_key == "ds-test"
            assert llm is None

    @patch("src.retrieval.load_dotenv")
    def test_setup_llm_invalid_provider(self, mock_dotenv, tmp_path):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "unknown"
        with pytest.raises(ValueError, match="不支持的 embedding provider"):
            retriever._set_up_llm()

    # -- _get_embedding (OpenAI) --

    def test_get_embedding_openai(self):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "openai"
        mock_llm = MagicMock()
        mock_llm.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        retriever.llm = mock_llm

        emb = retriever._get_embedding("hello")
        assert emb == [0.1, 0.2, 0.3]
        mock_llm.embeddings.create.assert_called_once_with(
            input="hello", model="text-embedding-3-large"
        )

    # -- _get_embedding (DashScope) --

    def test_get_embedding_dashscope_embeddings_format(self):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "dashscope"
        retriever.llm = None

        mock_rsp = {
            "output": {
                "embeddings": [
                    {"text_index": 0, "embedding": [0.4, 0.5, 0.6]}
                ]
            }
        }
        mock_dashscope = MagicMock()
        mock_dashscope.TextEmbedding.call.return_value = mock_rsp
        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            emb = retriever._get_embedding("hello")
            assert emb == [0.4, 0.5, 0.6]

    def test_get_embedding_dashscope_embedding_format(self):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "dashscope"
        retriever.llm = None

        mock_rsp = {"output": {"embedding": [0.7, 0.8, 0.9]}}
        mock_dashscope = MagicMock()
        mock_dashscope.TextEmbedding.call.return_value = mock_rsp
        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            emb = retriever._get_embedding("hello")
            assert emb == [0.7, 0.8, 0.9]

    def test_get_embedding_dashscope_empty_embedding(self):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "dashscope"
        retriever.llm = None

        mock_rsp = {"output": {"embeddings": [{"text_index": 0, "embedding": []}]}}
        mock_dashscope = MagicMock()
        mock_dashscope.TextEmbedding.call.return_value = mock_rsp
        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            with pytest.raises(RuntimeError, match="DashScope返回的embedding为空"):
                retriever._get_embedding("hello")

    def test_get_embedding_dashscope_malformed_response(self):
        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.embedding_provider = "dashscope"
        retriever.llm = None

        mock_rsp = {"unexpected": "key"}
        mock_dashscope = MagicMock()
        mock_dashscope.TextEmbedding.call.return_value = mock_rsp
        with patch.dict(sys.modules, {"dashscope": mock_dashscope}):
            with pytest.raises(RuntimeError, match="DashScope embedding API返回格式异常"):
                retriever._get_embedding("hello")

    # -- _load_dbs --

    @patch("src.retrieval.faiss.read_index")
    def test_load_dbs_success(self, mock_read_index, tmp_path, sample_document):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text(json.dumps(sample_document), encoding="utf-8")

        vector_db_dir = tmp_path / "vector"
        vector_db_dir.mkdir()
        faiss_path = vector_db_dir / f"{sample_document['metainfo']['sha1']}.faiss"
        faiss_path.write_text("dummy")

        mock_index = MagicMock()
        mock_read_index.return_value = mock_index

        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.vector_db_dir = vector_db_dir
        retriever.documents_dir = tmp_path
        dbs = retriever._load_dbs()

        assert len(dbs) == 1
        assert dbs[0]["name"] == sample_document["metainfo"]["sha1"]
        assert dbs[0]["vector_db"] is mock_index
        assert dbs[0]["document"] == sample_document

    @patch("src.retrieval.faiss.read_index")
    def test_load_dbs_missing_sha1(self, mock_read_index, tmp_path):
        doc_path = tmp_path / "doc.json"
        bad_doc = {"metainfo": {"company_name": "test"}}  # no sha1
        doc_path.write_text(json.dumps(bad_doc), encoding="utf-8")

        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.vector_db_dir = tmp_path
        retriever.documents_dir = tmp_path
        dbs = retriever._load_dbs()
        assert len(dbs) == 0

    @patch("src.retrieval.faiss.read_index")
    def test_load_dbs_missing_faiss_file(self, mock_read_index, tmp_path, sample_document):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text(json.dumps(sample_document), encoding="utf-8")

        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.vector_db_dir = tmp_path
        retriever.documents_dir = tmp_path
        dbs = retriever._load_dbs()
        assert len(dbs) == 0

    def test_load_dbs_json_error(self, tmp_path, caplog):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text("not valid json", encoding="utf-8")

        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.vector_db_dir = tmp_path
        retriever.documents_dir = tmp_path
        with caplog.at_level("ERROR"):
            dbs = retriever._load_dbs()
        assert len(dbs) == 0
        assert "Error loading JSON" in caplog.text

    @patch("src.retrieval.faiss.read_index")
    def test_load_dbs_faiss_read_error(self, mock_read_index, tmp_path, sample_document, caplog):
        doc_path = tmp_path / "doc.json"
        doc_path.write_text(json.dumps(sample_document), encoding="utf-8")

        vector_db_dir = tmp_path / "vector"
        vector_db_dir.mkdir()
        faiss_path = vector_db_dir / f"{sample_document['metainfo']['sha1']}.faiss"
        faiss_path.write_text("dummy")

        mock_read_index.side_effect = RuntimeError("corrupted index")

        retriever = VectorRetriever.__new__(VectorRetriever)
        retriever.vector_db_dir = vector_db_dir
        retriever.documents_dir = tmp_path
        with caplog.at_level("ERROR"):
            dbs = retriever._load_dbs()
        assert len(dbs) == 0
        assert "Error reading vector DB" in caplog.text

    # -- retrieve_by_company_name --

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    @patch("src.retrieval.VectorRetriever._get_embedding")
    def test_vector_retrieve_success(
        self, mock_get_emb, mock_setup, mock_load, tmp_path, sample_document, mock_faiss_index
    ):
        mock_load.return_value = [
            {
                "name": sample_document["metainfo"]["sha1"],
                "vector_db": mock_faiss_index,
                "document": sample_document,
            }
        ]
        mock_setup.return_value = None
        mock_get_emb.return_value = [0.1, 0.2]

        # create dummy faiss file to pass the exists() check in retrieve_by_company_name
        faiss_file = tmp_path / f"{sample_document['metainfo']['sha1']}.faiss"
        faiss_file.write_text("dummy")

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve_by_company_name(
            company_name="示例科技", query="test", top_n=2
        )

        assert len(results) == 2
        # indices returned by mock: [2, 0]
        assert results[0]["page"] == 3
        assert results[0]["distance"] == 0.1
        assert results[0]["text"] == "chunk C text"
        assert results[1]["page"] == 1
        assert results[1]["distance"] == 0.5
        assert results[1]["text"] == "chunk A text"

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    def test_vector_retrieve_company_not_found(self, mock_setup, mock_load, tmp_path):
        mock_load.return_value = []
        mock_setup.return_value = None

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        with pytest.raises(ValueError, match="No report found with '未知公司' company name"):
            retriever.retrieve_by_company_name(company_name="未知公司", query="test")

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    @patch("src.retrieval.VectorRetriever._get_embedding")
    def test_vector_retrieve_return_parent_pages(
        self, mock_get_emb, mock_setup, mock_load, tmp_path, sample_document, mock_faiss_index
    ):
        mock_load.return_value = [
            {
                "name": sample_document["metainfo"]["sha1"],
                "vector_db": mock_faiss_index,
                "document": sample_document,
            }
        ]
        mock_setup.return_value = None
        mock_get_emb.return_value = [0.1, 0.2]

        faiss_file = tmp_path / f"{sample_document['metainfo']['sha1']}.faiss"
        faiss_file.write_text("dummy")

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve_by_company_name(
            company_name="示例科技", query="test", top_n=2, return_parent_pages=True
        )

        assert len(results) == 2
        assert results[0]["text"] == "full page 3 text"
        assert results[1]["text"] == "full page 1 text"

    # -- retrieve_all --

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    def test_retrieve_all_success(self, mock_setup, mock_load, tmp_path, sample_document):
        mock_load.return_value = [
            {
                "name": sample_document["metainfo"]["sha1"],
                "vector_db": MagicMock(),
                "document": sample_document,
            }
        ]
        mock_setup.return_value = None

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        results = retriever.retrieve_all(company_name="示例科技")

        assert len(results) == 3
        for r in results:
            assert r["distance"] == 0.5
        assert results[0]["page"] == 1
        assert results[1]["page"] == 2
        assert results[2]["page"] == 3

    @patch("src.retrieval.VectorRetriever._load_dbs")
    @patch("src.retrieval.VectorRetriever._set_up_llm")
    def test_retrieve_all_not_found(self, mock_setup, mock_load, tmp_path):
        mock_load.return_value = []
        mock_setup.return_value = None

        retriever = VectorRetriever(vector_db_dir=tmp_path, documents_dir=tmp_path)
        with pytest.raises(ValueError, match="No report found with '未知公司' company name"):
            retriever.retrieve_all(company_name="未知公司")

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
    def test_retrieve_by_company_name(
        self, mock_reranker_cls, mock_vector_cls, tmp_path
    ):
        mock_vector_instance = MagicMock()
        mock_vector_instance.retrieve_by_company_name.return_value = [
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
        results = retriever.retrieve_by_company_name(
            company_name="示例科技",
            query="test",
            llm_reranking_sample_size=10,
            documents_batch_size=5,
            top_n=2,
            llm_weight=0.7,
        )

        assert len(results) == 2
        mock_vector_instance.retrieve_by_company_name.assert_called_once_with(
            company_name="示例科技",
            query="test",
            top_n=10,
            return_parent_pages=False,
        )
        mock_reranker_instance.rerank_documents.assert_called_once_with(
            query="test",
            documents=mock_vector_instance.retrieve_by_company_name.return_value,
            documents_batch_size=5,
            llm_weight=0.7,
        )

    @patch("src.retrieval.VectorRetriever")
    @patch("src.retrieval.LLMReranker")
    def test_retrieve_by_company_name_top_n_clipping(
        self, mock_reranker_cls, mock_vector_cls, tmp_path
    ):
        mock_vector_instance = MagicMock()
        mock_vector_instance.retrieve_by_company_name.return_value = [
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
        results = retriever.retrieve_by_company_name(
            company_name="示例科技", query="test", top_n=1
        )

        # Even though reranker returned 2, we clip to top_n=1
        assert len(results) == 1
        assert results[0]["combined_score"] == 0.9
