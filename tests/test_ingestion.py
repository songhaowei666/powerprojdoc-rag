"""
Tests for src/ingestion.py

Follows TDD principles: tests are written against the spec in spec/ingestion_spec.md.
Run with: pytest tests/test_ingestion.py -v
"""

import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch tenacity.retry BEFORE importing ingestion, so @retry becomes no-op
import tenacity

_original_retry = tenacity.retry


def _noop_retry(**kwargs):
    """No-op replacement for tenacity.retry to disable waiting in tests."""
    def decorator(f):
        return f
    return decorator


tenacity.retry = _noop_retry

from src.ingestion import BM25Ingestor, VectorDBIngestor

# Restore after import so other code isn't affected
tenacity.retry = _original_retry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_report():
    """Return a minimal valid report dict for ingestion."""
    return {
        "metainfo": {
            "sha1": "abc123def456",
            "company_name": "示例科技",
        },
        "content": {
            "chunks": [
                {"text": "first chunk text"},
                {"text": "second chunk text"},
                {"text": "third chunk text"},
            ]
        },
    }


# ---------------------------------------------------------------------------
# BM25Ingestor Tests
# ---------------------------------------------------------------------------

class TestBM25Ingestor:
    """Tests for BM25Ingestor."""

    def test_init(self):
        ingestor = BM25Ingestor()
        assert ingestor is not None

    def test_create_bm25_index(self):
        ingestor = BM25Ingestor()
        chunks = ["hello world", "foo bar baz", "test example"]
        index = ingestor.create_bm25_index(chunks)
        # BM25Okapi should have get_scores method
        assert hasattr(index, "get_scores")
        scores = index.get_scores(["hello"])
        assert len(scores) == 3
        assert isinstance(scores, np.ndarray)

    def test_process_reports_success(self, tmp_path, sample_report):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        report_path.write_text(json.dumps(sample_report), encoding="utf-8")

        output_dir = tmp_path / "bm25_output"
        ingestor = BM25Ingestor()
        ingestor.process_reports(reports_dir, output_dir)

        # Verify output file exists (default index_name = "default")
        expected_file = output_dir / "default.pkl"
        assert expected_file.exists()

        # Verify new format: dict with index and metadatas
        with open(expected_file, "rb") as f:
            loaded_data = pickle.load(f)
        assert isinstance(loaded_data, dict)
        assert "index" in loaded_data
        assert "metadatas" in loaded_data
        assert hasattr(loaded_data["index"], "get_scores")

        # Verify metadata preserved
        metadatas = loaded_data["metadatas"]
        assert len(metadatas) == 3
        for m in metadatas:
            assert m["sha1"] == sample_report["metainfo"]["sha1"]
            assert m["company_name"] == sample_report["metainfo"]["company_name"]

    def test_process_reports_empty_dir(self, tmp_path, capsys):
        reports_dir = tmp_path / "empty_reports"
        reports_dir.mkdir()
        output_dir = tmp_path / "bm25_output"

        ingestor = BM25Ingestor()
        ingestor.process_reports(reports_dir, output_dir)

        captured = capsys.readouterr()
        assert "Processed 0 reports" in captured.out

    def test_load_bm25_index(self, tmp_path):
        ingestor = BM25Ingestor()
        chunks = ["hello world", "foo bar baz"]
        index = ingestor.create_bm25_index(chunks)
        metadatas = [{"page": 1}, {"page": 2}]

        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"index": index, "metadatas": metadatas, "texts": chunks}, f)

        loaded_index, loaded_metadatas, loaded_texts = BM25Ingestor.load_bm25_index(pkl_path)
        assert hasattr(loaded_index, "get_scores")
        assert loaded_metadatas == metadatas
        assert loaded_texts == chunks

    def test_load_bm25_index_legacy_format(self, tmp_path):
        """兼容旧格式：纯 BM25Okapi 对象"""
        ingestor = BM25Ingestor()
        chunks = ["hello world"]
        index = ingestor.create_bm25_index(chunks)

        pkl_path = tmp_path / "legacy.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(index, f)

        loaded_index, loaded_metadatas, loaded_texts = BM25Ingestor.load_bm25_index(pkl_path)
        assert hasattr(loaded_index, "get_scores")
        assert loaded_metadatas == []
        assert loaded_texts == []

    def test_search_success(self, tmp_path):
        ingestor = BM25Ingestor()
        chunks = ["hello world", "foo bar baz", "test example"]
        index = ingestor.create_bm25_index(chunks)
        metadatas = [{"page": 1}, {"page": 2}, {"page": 3}]

        pkl_path = tmp_path / "default.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"index": index, "metadatas": metadatas, "texts": chunks}, f)

        scores, result_metadatas, result_texts = ingestor.search("hello", output_dir=tmp_path)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert result_metadatas == metadatas
        assert result_texts == chunks

    def test_search_file_not_found(self, tmp_path):
        ingestor = BM25Ingestor()
        with pytest.raises(FileNotFoundError, match="BM25 index not found"):
            ingestor.search("hello", index_name="nonexistent", output_dir=tmp_path)

    def test_process_reports_multiple_reports(self, tmp_path):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        for i in range(3):
            report = {
                "metainfo": {"sha1": f"sha{i}", "company_name": f"公司{i}"},
                "content": {"chunks": [{"text": f"text {i}", "page": i + 1, "id": i}]},
            }
            (reports_dir / f"report_{i}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

        output_dir = tmp_path / "bm25_output"
        ingestor = BM25Ingestor()
        ingestor.process_reports(reports_dir, output_dir, index_name="merged")

        # All reports merged into a single index file
        assert len(list(output_dir.glob("*.pkl"))) == 1
        expected_file = output_dir / "merged.pkl"
        assert expected_file.exists()

        with open(expected_file, "rb") as f:
            loaded_data = pickle.load(f)
        assert len(loaded_data["metadatas"]) == 3
        assert loaded_data["metadatas"][1]["page"] == 2
        assert loaded_data["metadatas"][1]["chunk_id"] == 1
        assert loaded_data["metadatas"][1]["sha1"] == "sha1"


# ---------------------------------------------------------------------------
# VectorDBIngestor Tests
# ---------------------------------------------------------------------------

class TestVectorDBIngestor:
    """Tests for VectorDBIngestor."""

    # -- __init__ --

    @patch("src.ingestion.os.getenv")
    def test_init_sets_api_key(self, mock_getenv):
        mock_getenv.return_value = "test-api-key"
        with patch("dashscope.api_key", None) as mock_api_key_attr:
            # We need to patch the assignment target. Since ingestion.py does:
            #   dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            # we patch os.getenv and then check dashscope.api_key after init.
            import dashscope
            original = dashscope.api_key
            try:
                ingestor = VectorDBIngestor()
                assert dashscope.api_key == "test-api-key"
            finally:
                dashscope.api_key = original

    # -- _get_embeddings --

    @patch("src.ingestion.TextEmbedding.call")
    def test_get_embeddings_single_text(self, mock_call):
        mock_call.return_value = {
            "output": {"embeddings": [{"text_index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        }
        ingestor = VectorDBIngestor()
        result = ingestor._get_embeddings("hello world")
        assert result == [[0.1, 0.2, 0.3]]
        mock_call.assert_called_once()

    @patch("src.ingestion.TextEmbedding.call")
    def test_get_embeddings_batch_texts(self, mock_call):
        # simulate 30 texts -> 2 batches (25 + 5)
        def side_effect(*args, **kwargs):
            batch = kwargs.get("input", [])
            return {
                "output": {
                    "embeddings": [
                        {"text_index": i, "embedding": [float(i), float(i + 1)]}
                        for i in range(len(batch))
                    ]
                }
            }

        mock_call.side_effect = side_effect
        ingestor = VectorDBIngestor()
        texts = [f"text {i}" for i in range(30)]
        result = ingestor._get_embeddings(texts)

        assert len(result) == 30
        assert mock_call.call_count == 2  # 25 + 5
        # first batch size 25
        first_call_input = mock_call.call_args_list[0][1]["input"]
        assert len(first_call_input) == 25
        # second batch size 5
        second_call_input = mock_call.call_args_list[1][1]["input"]
        assert len(second_call_input) == 5

    @patch("src.ingestion.TextEmbedding.call")
    @patch("src.ingestion.retry", _noop_retry)
    def test_get_embeddings_empty_string(self, mock_call):
        ingestor = VectorDBIngestor()
        with pytest.raises(ValueError, match="Input text cannot be an empty string"):
            ingestor._get_embeddings("   ")
        mock_call.assert_not_called()

    @patch("src.ingestion.TextEmbedding.call")
    @patch("src.ingestion.retry", _noop_retry)
    def test_get_embeddings_all_empty_list(self, mock_call):
        ingestor = VectorDBIngestor()
        with pytest.raises(ValueError, match="所有待嵌入文本均为空字符串"):
            ingestor._get_embeddings(["", "   ", "\t"])
        mock_call.assert_not_called()

    @patch("src.ingestion.TextEmbedding.call")
    @patch("src.ingestion.retry", _noop_retry)
    def test_get_embeddings_non_string_items(self, mock_call):
        ingestor = VectorDBIngestor()
        with pytest.raises(ValueError, match="所有待嵌入文本必须为字符串类型"):
            ingestor._get_embeddings(["valid", 123, None])
        mock_call.assert_not_called()

    @patch("src.ingestion.TextEmbedding.call")
    @patch("src.ingestion.retry", _noop_retry)
    def test_get_embeddings_empty_embedding_response(self, mock_call, tmp_path):
        mock_call.return_value = {
            "output": {"embeddings": [{"text_index": 0, "embedding": None}]}
        }
        ingestor = VectorDBIngestor()
        with pytest.raises(RuntimeError, match="DashScope返回的embedding为空"):
            ingestor._get_embeddings("hello")

    @patch("src.ingestion.TextEmbedding.call")
    @patch("src.ingestion.retry", _noop_retry)
    def test_get_embeddings_malformed_response(self, mock_call):
        mock_call.return_value = {"unexpected": "key"}
        ingestor = VectorDBIngestor()
        with pytest.raises(RuntimeError, match="DashScope embedding API返回格式异常"):
            ingestor._get_embeddings("hello")

    @patch("src.ingestion.TextEmbedding.call")
    def test_get_embeddings_single_embedding_format(self, mock_call):
        mock_call.return_value = {"output": {"embedding": [0.4, 0.5, 0.6]}}
        ingestor = VectorDBIngestor()
        result = ingestor._get_embeddings("hello")
        assert result == [[0.4, 0.5, 0.6]]

    @patch("src.ingestion.TextEmbedding.call")
    @patch("src.ingestion.retry", _noop_retry)
    def test_get_embeddings_single_empty_embedding(self, mock_call):
        mock_call.return_value = {"output": {"embedding": []}}
        ingestor = VectorDBIngestor()
        with pytest.raises(RuntimeError, match="DashScope返回的embedding为空"):
            ingestor._get_embeddings("hello")

    # -- _create_vector_db --

    def test_create_vector_db(self):
        ingestor = VectorDBIngestor()
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        index = ingestor._create_vector_db(embeddings)
        assert index.ntotal == 3
        assert index.d == 3

    # -- _process_report --

    @patch.object(VectorDBIngestor, "_get_embeddings")
    @patch.object(VectorDBIngestor, "_create_vector_db")
    def test_process_report_success(self, mock_create, mock_get_emb, sample_report):
        mock_get_emb.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_index = MagicMock()
        mock_create.return_value = mock_index

        ingestor = VectorDBIngestor()
        result = ingestor._process_report(sample_report)

        assert result is mock_index
        mock_get_emb.assert_called_once()
        # Verify chunks were passed (3 chunks)
        passed_chunks = mock_get_emb.call_args[0][0]
        assert len(passed_chunks) == 3
        assert passed_chunks[0] == "first chunk text"

    @patch.object(VectorDBIngestor, "_get_embeddings")
    @patch.object(VectorDBIngestor, "_create_vector_db")
    def test_process_report_truncates_to_2048(self, mock_create, mock_get_emb):
        long_text = "a" * 3000
        report = {
            "metainfo": {"sha1": "test"},
            "content": {"chunks": [{"text": long_text}]},
        }
        mock_get_emb.return_value = [[0.1, 0.2]]
        mock_create.return_value = MagicMock()

        ingestor = VectorDBIngestor()
        ingestor._process_report(report)

        passed_chunks = mock_get_emb.call_args[0][0]
        assert len(passed_chunks[0]) == 2048

    @patch.object(VectorDBIngestor, "_get_embeddings")
    @patch.object(VectorDBIngestor, "_create_vector_db")
    def test_process_report_filters_empty_chunks(self, mock_create, mock_get_emb):
        report = {
            "metainfo": {"sha1": "test"},
            "content": {"chunks": [{"text": ""}, {"text": "valid text"}, {"text": ""}]},
        }
        mock_get_emb.return_value = [[0.1, 0.2]]
        mock_create.return_value = MagicMock()

        ingestor = VectorDBIngestor()
        ingestor._process_report(report)

        passed_chunks = mock_get_emb.call_args[0][0]
        assert len(passed_chunks) == 1
        assert passed_chunks[0] == "valid text"

    # -- process_reports --

    @patch.object(VectorDBIngestor, "_process_report")
    @patch("src.ingestion.faiss.write_index")
    def test_process_reports_success(self, mock_write, mock_process, tmp_path, sample_report):
        mock_index = MagicMock()
        mock_process.return_value = mock_index

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        report_path.write_text(json.dumps(sample_report), encoding="utf-8")

        output_dir = tmp_path / "vector_output"
        ingestor = VectorDBIngestor()
        ingestor.process_reports(reports_dir, output_dir)

        expected_path = output_dir / f"{sample_report['metainfo']['sha1']}.faiss"
        mock_write.assert_called_once_with(mock_index, str(expected_path))

    @patch.object(VectorDBIngestor, "_process_report")
    @patch("src.ingestion.faiss.write_index")
    def test_process_reports_missing_sha1(self, mock_write, mock_process, tmp_path):
        bad_report = {
            "metainfo": {"company_name": "no sha1"},
            "content": {"chunks": [{"text": "test"}]},
        }
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "bad.json"
        report_path.write_text(json.dumps(bad_report), encoding="utf-8")

        output_dir = tmp_path / "vector_output"
        ingestor = VectorDBIngestor()
        with pytest.raises(ValueError, match="缺少 sha1 字段"):
            ingestor.process_reports(reports_dir, output_dir)

    @patch.object(VectorDBIngestor, "_process_report")
    @patch("src.ingestion.faiss.write_index")
    def test_process_reports_empty_dir(self, mock_write, mock_process, tmp_path, capsys):
        reports_dir = tmp_path / "empty_reports"
        reports_dir.mkdir()
        output_dir = tmp_path / "vector_output"

        ingestor = VectorDBIngestor()
        ingestor.process_reports(reports_dir, output_dir)

        captured = capsys.readouterr()
        assert "Processed 0 reports" in captured.out
        mock_write.assert_not_called()
