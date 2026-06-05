"""
Tests for src/markdown_reports_merging.py

Follows TDD principles: tests are written against the spec in spec/markdown_reports_merging_spec.md.
Run with: pytest tests/test_markdown_reports_merging.py -v
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.markdown_reports_merging import MinerUReportMerger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def merger():
    """Return a fresh MinerUReportMerger instance."""
    return MinerUReportMerger()


@pytest.fixture
def mineru_report_data():
    """Return a sample MinerU-format report dict."""
    return {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [595, 841],
                "para_blocks": [
                    {
                        "type": "title",
                        "index": 0,
                        "lines": [
                            {"spans": [{"type": "text", "content": "年度报告"}]}
                        ],
                    },
                    {
                        "type": "text",
                        "index": 1,
                        "lines": [
                            {
                                "spans": [
                                    {
                                        "type": "text",
                                        "content": "本公司2024年度实现营业收入100亿元。",
                                    }
                                ]
                            }
                        ],
                    },
                    {
                        "type": "table",
                        "index": 2,
                        "lines": [
                            {
                                "spans": [
                                    {
                                        "html": "<table><tr><th>项目</th><th>金额</th></tr><tr><td>收入</td><td>100</td></tr></table>"
                                    }
                                ]
                            }
                        ],
                    },
                    {
                        "type": "image",
                        "index": 3,
                        "lines": [],
                    },
                ],
            },
            {
                "page_idx": 1,
                "page_size": [600, 850],
                "para_blocks": [
                    {
                        "type": "text",
                        "index": 0,
                        "lines": [
                            {"spans": [{"type": "text", "content": "Important Notice:"}]}
                        ],
                    },
                    {
                        "type": "list",
                        "index": 1,
                        "blocks": [
                            {
                                "type": "text",
                                "index": 0,
                                "lines": [
                                    {"spans": [{"type": "text", "content": " item 1"}]}
                                ],
                            },
                            {
                                "type": "text",
                                "index": 1,
                                "lines": [
                                    {"spans": [{"type": "text", "content": " item 2"}]}
                                ],
                            },
                        ],
                    },
                ],
            },
        ]
    }


@pytest.fixture
def parsed_report_data():
    """Return a sample already-parsed (Docling-style) report dict."""
    return {
        "metainfo": {
            "sha1_name": "doc123",
            "sha1": "doc123",
            "company_name": "示例科技",
            "file_name": "示例科技_2024年报.pdf",
            "pages_amount": 1,
            "text_blocks_amount": 2,
            "tables_amount": 0,
            "pictures_amount": 0,
            "equations_amount": 0,
            "footnotes_amount": 0,
        },
        "content": [
            {
                "page": 1,
                "content": [
                    {"type": "section_header", "text": "第一章"},
                    {"type": "text", "text": "正文内容。"},
                ],
                "page_dimensions": {"width": 595, "height": 841},
            }
        ],
        "tables": [],
        "pictures": [],
    }


@pytest.fixture
def mock_processed_report():
    """Return a mock result from PageTextPreparation.process_report."""
    return {
        "chunks": None,
        "pages": [
            {"page": 1, "text": "mocked page 1 text"},
            {"page": 2, "text": "mocked page 2 text"},
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5.1 Batch Processing Basics (T1~T5)
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    """Tests for batch report processing entry behavior."""

    def test_empty_directory_returns_empty_list(self, tmp_path, merger):
        """T1: 传入空目录返回空列表，不报错。"""
        reports_dir = tmp_path / "empty"
        reports_dir.mkdir()

        with patch.object(merger._page_preparator, "process_report", return_value={"chunks": None, "pages": []}):
            results = merger.process_reports(reports_dir=reports_dir)

        assert results == []

    def test_empty_paths_list_returns_empty_list(self, merger):
        """T2: 传入空路径列表返回空列表，不报错。"""
        with patch.object(merger._page_preparator, "process_report", return_value={"chunks": None, "pages": []}):
            results = merger.process_reports(reports_paths=[])

        assert results == []

    def test_single_mineru_file(self, tmp_path, merger, mineru_report_data, mock_processed_report):
        """T3: 传入单个 MinerU 格式 JSON，正确识别并完成转换。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        _write_json(report_path, mineru_report_data)

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(reports_paths=[report_path])

        assert len(results) == 1
        assert "metainfo" in results[0]
        assert "content" in results[0]
        # MinerU format -> metainfo assembled by _mineru_to_parsed_report
        assert results[0]["metainfo"]["pages_amount"] == 2

    def test_single_parsed_file(self, tmp_path, merger, parsed_report_data, mock_processed_report):
        """T4: 传入单个已规整格式 JSON，跳过转换直接规整。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "doc.json"
        _write_json(report_path, parsed_report_data)

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(reports_paths=[report_path])

        assert len(results) == 1
        # metainfo should be preserved from original parsed report
        assert results[0]["metainfo"]["company_name"] == "示例科技"
        assert results[0]["metainfo"]["sha1"] == "doc123"

    def test_mixed_formats(self, tmp_path, merger, mineru_report_data, parsed_report_data, mock_processed_report):
        """T5: 混合传入 MinerU 和已规整格式，分别正确处理。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mineru_path = reports_dir / "mineru.json"
        parsed_path = reports_dir / "parsed.json"
        _write_json(mineru_path, mineru_report_data)
        _write_json(parsed_path, parsed_report_data)

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(reports_paths=[mineru_path, parsed_path])

        assert len(results) == 2
        # First is MinerU -> pages_amount from conversion
        assert results[0]["metainfo"]["pages_amount"] == 2
        # Second is parsed -> preserve original metainfo
        assert results[1]["metainfo"]["company_name"] == "示例科技"


# ---------------------------------------------------------------------------
# 5.2 Format Conversion & Content Mapping (T6~T15)
# ---------------------------------------------------------------------------

class TestFormatConversion:
    """Tests for MinerU format conversion and block type mapping."""

    def test_title_mapped_to_section_header(self, merger):
        """T6: title 块映射为 section_header，文本正确提取。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "title",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "财务摘要"}]}
                            ],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section_header"
        assert blocks[0]["text"] == "财务摘要"

    def test_text_mapped_to_text(self, merger):
        """T7: text 块映射为 text，文本正确提取。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "正文段落。"}]}
                            ],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "正文段落。"

    def test_text_ending_with_colon_mapped_to_paragraph(self, merger):
        """T8: text 块文本以冒号结尾映射为 paragraph 类型。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "Important Notice:"}]}
                            ],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert blocks[0]["type"] == "paragraph"

    def test_table_extracts_html_and_converts_to_markdown(self, merger):
        """T9: table 块分配递增 table_id，HTML 提取并转为 Markdown，tables 数组正确填充。"""
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "table",
                            "index": 0,
                            "lines": [{"spans": [{"html": html}]}],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert blocks[0]["type"] == "table"
        assert blocks[0]["table_id"] == 0

        assert len(parsed["tables"]) == 1
        assert parsed["tables"][0]["table_id"] == 0
        assert parsed["tables"][0]["page"] == 1
        assert parsed["tables"][0]["html"] == html
        # Markdown should be non-empty (pandas conversion)
        assert isinstance(parsed["tables"][0]["markdown"], str)
        assert "A" in parsed["tables"][0]["markdown"]

    def test_table_html_parsing_failure_fallback(self, merger):
        """T10: table 块 HTML 解析失败时 Markdown 回退为原始 HTML 字符串。"""
        bad_html = "<not-a-table>broken"
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "table",
                            "index": 0,
                            "lines": [{"spans": [{"html": bad_html}]}],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        assert parsed["tables"][0]["markdown"] == bad_html

    def test_image_mapped_to_picture(self, merger):
        """T11: image 块映射为 picture，picture_id 为 0。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {"type": "image", "index": 0, "lines": []}
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert blocks[0]["type"] == "picture"
        assert blocks[0]["picture_id"] == 0

    def test_list_recursive_expansion(self, merger):
        """T12: list 块递归展开子块，内容正确合并，统计准确。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "list",
                            "index": 0,
                            "blocks": [
                                {
                                    "type": "text",
                                    "index": 0,
                                    "lines": [
                                        {"spans": [{"type": "text", "content": "a"}]}
                                    ],
                                },
                                {
                                    "type": "text",
                                    "index": 1,
                                    "lines": [
                                        {"spans": [{"type": "text", "content": "b"}]}
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["text"] == "a"
        assert blocks[1]["text"] == "b"
        assert parsed["metainfo"]["text_blocks_amount"] == 2

    def test_unknown_block_treated_as_text(self, merger):
        """T13: 未知类型的 block 按 text 处理，尝试提取文本。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "custom_unknown",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "fallback text"}]}
                            ],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "fallback text"

    def test_multi_page_ordered_by_page_idx(self, merger):
        """T14: 多页报告按 page_idx 升序处理，每页内容独立。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 1,
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "page2"}]}
                            ],
                        }
                    ],
                },
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "page1"}]}
                            ],
                        }
                    ],
                },
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        pages = parsed["content"]
        assert len(pages) == 2
        assert pages[0]["page"] == 1
        assert pages[0]["content"][0]["text"] == "page1"
        assert pages[1]["page"] == 2
        assert pages[1]["content"][0]["text"] == "page2"

    def test_empty_span_ignored(self, merger):
        """T15: 页面含空 text span 或缺失 content 时忽略空内容。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {
                                    "spans": [
                                        {"type": "text", "content": ""},
                                        {"type": "text", "content": "valid"},
                                    ]
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "index": 1,
                            "lines": [
                                {"spans": [{"type": "image"}]}  # no content key
                            ],
                        },
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        blocks = parsed["content"][0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["text"] == "valid"


# ---------------------------------------------------------------------------
# 5.3 Metainfo Parsing & Output Filename (T16~T24)
# ---------------------------------------------------------------------------

class TestMetainfoParsing:
    """Tests for subset.csv loading, filename resolution and output naming."""

    def test_subset_csv_loads_metainfo(self, tmp_path, merger, mineru_report_data, mock_processed_report):
        """T16: 提供有效的 subset.csv，按文件名匹配正确填充元信息。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        _write_json(report_path, mineru_report_data)

        subset_csv = tmp_path / "subset.csv"
        subset_csv.write_text(
            "file_name,sha1,company_name\n"
            "report.json,abc123,测试公司\n",
            encoding="utf-8",
        )

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(
                reports_paths=[report_path], subset_csv=subset_csv
            )

        assert results[0]["metainfo"]["company_name"] == "测试公司"
        assert results[0]["metainfo"]["sha1"] == "abc123"

    def test_subset_csv_gbk_encoding(self, tmp_path, merger, mineru_report_data, mock_processed_report):
        """T17: subset.csv 编码为 gbk 时正确解码并读取。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        _write_json(report_path, mineru_report_data)

        subset_csv = tmp_path / "subset.csv"
        subset_csv.write_text(
            "file_name,sha1,company_name\n"
            "report.json,abc123,测试公司\n",
            encoding="gbk",
        )

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(
                reports_paths=[report_path], subset_csv=subset_csv
            )

        assert results[0]["metainfo"]["company_name"] == "测试公司"

    def test_subset_csv_missing_filename_column(self, tmp_path, merger, mineru_report_data, mock_processed_report):
        """T18: subset.csv 缺少 file_name 列时忽略 CSV，metainfo 使用默认值。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        _write_json(report_path, mineru_report_data)

        subset_csv = tmp_path / "subset.csv"
        subset_csv.write_text(
            "sha1,company_name\n" "abc123,测试公司\n", encoding="utf-8"
        )

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(
                reports_paths=[report_path], subset_csv=subset_csv
            )

        # Without file_name column, no mapping is built; company_name defaults to ""
        assert results[0]["metainfo"]["company_name"] == ""

    def test_subset_csv_not_exists(self, tmp_path, merger, mineru_report_data, mock_processed_report):
        """T19: subset.csv 文件不存在时返回空映射，metainfo 使用默认值。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "report.json"
        _write_json(report_path, mineru_report_data)

        subset_csv = tmp_path / "nonexistent.csv"

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            results = merger.process_reports(
                reports_paths=[report_path], subset_csv=subset_csv
            )

        assert results[0]["metainfo"]["company_name"] == ""

    def test_mineru_filename_prefix_stripping(self, merger):
        """T20: MinerU 文件名以 MinerU_ 开头并含 __ 时正确提取报告主名。"""
        path = Path("MinerU_【财报】中芯国际：中芯国际2024年年度报告__20260520083937.json")
        stem = merger._resolve_report_stem(path)
        assert stem == "【财报】中芯国际：中芯国际2024年年度报告"

    def test_mineru_filename_no_prefix(self, merger):
        """T21: MinerU 文件名无特殊前缀时直接使用文件 stem 作为主名。"""
        path = Path("普通报告.json")
        stem = merger._resolve_report_stem(path)
        assert stem == "普通报告"

    def test_output_filename_from_file_name(self, merger):
        """T22: metainfo 中 file_name 非空时输出文件名为其去扩展名 + .json。"""
        report_data = {
            "metainfo": {
                "file_name": "原始文件.pdf",
                "sha1_name": "abc",
            }
        }
        name = merger._resolve_output_name(Path("dummy.json"), report_data)
        assert name == "原始文件.json"

    def test_output_filename_from_sha1_name(self, merger):
        """T23: metainfo 中 file_name 为空但 sha1_name 非空时输出文件名为 sha1_name + .json。"""
        report_data = {
            "metainfo": {
                "file_name": "",
                "sha1_name": "abc",
            }
        }
        name = merger._resolve_output_name(Path("dummy.json"), report_data)
        assert name == "abc.json"

    def test_output_filename_from_stem_fallback(self, merger):
        """T24: metainfo 中 file_name 和 sha1_name 均为空时输出文件名为解析的报告主名 + .json。"""
        report_data = {"metainfo": {"file_name": "", "sha1_name": ""}}
        path = Path("fallback.json")
        name = merger._resolve_output_name(path, report_data)
        assert name == "fallback.json"


# ---------------------------------------------------------------------------
# 5.4 Stats & Boundary Cases (T25~T30)
# ---------------------------------------------------------------------------

class TestStatsAndBoundaries:
    """Tests for statistics, page dimensions and output directory handling."""

    def test_empty_page(self, merger):
        """T25: 空页面（无 para_blocks）时页面内容为空块列表，pages_amount 正确。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [595, 841],
                    "para_blocks": [],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        assert len(parsed["content"]) == 1
        assert parsed["content"][0]["content"] == []
        assert parsed["metainfo"]["pages_amount"] == 1
        assert parsed["metainfo"]["text_blocks_amount"] == 0

    def test_missing_page_size_fallback(self, merger):
        """T26: 页面缺少 page_size 时宽高回退为默认值 595×841。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {"spans": [{"type": "text", "content": "x"}]}
                            ],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        dims = parsed["content"][0]["page_dimensions"]
        assert dims["width"] == 595
        assert dims["height"] == 841

    def test_multiple_tables_incrementing_ids(self, merger):
        """T27: 多表格场景下 table_id 全局递增不重复，各表格 Markdown 独立。"""
        html1 = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        html2 = "<table><tr><th>B</th></tr><tr><td>2</td></tr></table>"
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "table",
                            "index": 0,
                            "lines": [{"spans": [{"html": html1}]}],
                        },
                        {
                            "type": "table",
                            "index": 1,
                            "lines": [{"spans": [{"html": html2}]}],
                        },
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        assert len(parsed["tables"]) == 2
        assert parsed["tables"][0]["table_id"] == 0
        assert parsed["tables"][1]["table_id"] == 1
        assert parsed["tables"][0]["markdown"] != parsed["tables"][1]["markdown"]

    def test_empty_table_html_returns_empty_markdown(self, merger):
        """T28: 表格 HTML 为空字符串时 Markdown 返回空字符串。"""
        mineru_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "table",
                            "index": 0,
                            "lines": [{"spans": [{"html": ""}]}],
                        }
                    ],
                }
            ]
        }
        parsed = merger._mineru_to_parsed_report(mineru_data, {}, "test")
        assert parsed["tables"][0]["markdown"] == ""

    def test_output_dir_exists(self, tmp_path, merger, parsed_report_data, mock_processed_report):
        """T29: 输出目录已存在时不报错，正常写入。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "doc.json"
        _write_json(report_path, parsed_report_data)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            merger.process_reports(reports_paths=[report_path], output_dir=output_dir)

        assert len(list(output_dir.glob("*.json"))) == 1

    def test_output_dir_auto_created(self, tmp_path, merger, parsed_report_data, mock_processed_report):
        """T30: 输出目录不存在时自动递归创建。"""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_path = reports_dir / "doc.json"
        _write_json(report_path, parsed_report_data)

        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        with patch.object(merger._page_preparator, "process_report", return_value=mock_processed_report):
            merger.process_reports(reports_paths=[report_path], output_dir=output_dir)

        assert output_dir.exists()
        assert len(list(output_dir.glob("*.json"))) == 1
