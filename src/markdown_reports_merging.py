import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.parsed_reports_merging import PageTextPreparation


class MinerUReportMerger:
    """
    将 JSON 报告规整为 02_merged_reports 格式（metainfo + content.pages）。
    接口模仿 PageTextPreparation.process_reports。

    输入支持：
    - MinerU 解析 JSON（pdf_info）-> 先转为 01_parsed_reports 结构，再经 PageTextPreparation 规整
    - Docling 解析 JSON（content 为页面块列表）-> 直接经 PageTextPreparation 规整
    """

    def __init__(self):
        self._page_preparator = PageTextPreparation()
        self._table_id_counter = 0

    def process_reports(
        self,
        reports_dir: Path = None,
        reports_paths: List[Path] = None,
        output_dir: Path = None,
        subset_csv: Path = None,
    ) -> List[Dict]:
        """
        批量处理 JSON 报告，返回规整后报告列表，可选保存到输出目录。
        """
        all_reports = []

        if reports_dir:
            reports_paths = list(reports_dir.glob("*.json"))

        file2meta = self._load_metainfo_from_subset(subset_csv)

        for report_path in reports_paths:
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)

            if "pdf_info" in report_data:
                stem = self._resolve_report_stem(report_path)
                metainfo = file2meta.get(stem, {})
                report_data = self._mineru_to_parsed_report(report_data, metainfo, stem)

            full_report_text = self._page_preparator.process_report(report_data)
            report = {"metainfo": report_data["metainfo"], "content": full_report_text}
            all_reports.append(report)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_name = self._resolve_output_name(report_path, report_data)
                with open(output_dir / output_name, "w", encoding="utf-8") as file:
                    json.dump(report, file, indent=2, ensure_ascii=False)

        return all_reports

    def _mineru_to_parsed_report(
        self, mineru_data: Dict, base_metainfo: Dict, report_stem: str
    ) -> Dict:
        """MinerU JSON 转为 01_parsed_reports 结构，供 PageTextPreparation 使用。"""
        self._table_id_counter = 0
        pages_content = []
        tables = []
        text_blocks_amount = 0
        tables_amount = 0
        pictures_amount = 0

        pdf_pages = sorted(mineru_data["pdf_info"], key=lambda p: p.get("page_idx", 0))

        for page_info in pdf_pages:
            page_num = page_info.get("page_idx", 0) + 1
            page_blocks = []
            para_blocks = sorted(
                page_info.get("para_blocks", []), key=lambda b: b.get("index", 0)
            )

            for block in para_blocks:
                converted_blocks, stats = self._convert_mineru_block(block, page_num, tables)
                page_blocks.extend(converted_blocks)
                text_blocks_amount += stats["text"]
                tables_amount += stats["table"]
                pictures_amount += stats["picture"]

            page_size = page_info.get("page_size", [595, 841])
            pages_content.append(
                {
                    "page": page_num,
                    "content": page_blocks,
                    "page_dimensions": {
                        "width": page_size[0] if len(page_size) > 0 else 595,
                        "height": page_size[1] if len(page_size) > 1 else 841,
                    },
                }
            )

        metainfo = {
            "sha1_name": base_metainfo.get("sha1", base_metainfo.get("sha1_name", report_stem)),
            "sha1": base_metainfo.get("sha1", ""),
            "company_name": base_metainfo.get("company_name", ""),
            "file_name": base_metainfo.get("file_name", report_stem),
            "pages_amount": len(pages_content),
            "text_blocks_amount": text_blocks_amount,
            "tables_amount": tables_amount,
            "pictures_amount": pictures_amount,
            "equations_amount": 0,
            "footnotes_amount": 0,
        }

        return {
            "metainfo": metainfo,
            "content": pages_content,
            "tables": tables,
            "pictures": [],
        }

    def _convert_mineru_block(
        self, block: Dict, page_num: int, tables: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """将 MinerU 块映射为 Docling 风格 content 块。"""
        block_type = block.get("type")
        stats = {"text": 0, "table": 0, "picture": 0}
        result = []

        if block_type == "title":
            text = self._extract_mineru_text(block)
            if text:
                result.append({"type": "section_header", "text": text})
                stats["text"] += 1
        elif block_type == "text":
            text = self._extract_mineru_text(block)
            if text:
                block_item = {"type": "text", "text": text}
                if text.rstrip().endswith(":"):
                    block_item["type"] = "paragraph"
                result.append(block_item)
                stats["text"] += 1
        elif block_type == "table":
            html = self._extract_mineru_table_html(block)
            markdown = self._html_to_markdown(html)
            table_id = self._table_id_counter
            self._table_id_counter += 1
            tables.append(
                {
                    "table_id": table_id,
                    "page": page_num,
                    "markdown": markdown,
                    "html": html,
                }
            )
            result.append({"type": "table", "table_id": table_id})
            stats["table"] += 1
        elif block_type == "image":
            result.append({"type": "picture", "picture_id": 0})
            stats["picture"] += 1
        elif block_type == "list":
            sub_blocks = sorted(block.get("blocks", []), key=lambda b: b.get("index", 0))
            for sub_block in sub_blocks:
                sub_result, sub_stats = self._convert_mineru_block(sub_block, page_num, tables)
                result.extend(sub_result)
                for key in stats:
                    stats[key] += sub_stats[key]
        else:
            text = self._extract_mineru_text(block)
            if text:
                result.append({"type": "text", "text": text})
                stats["text"] += 1

        return result, stats

    def _extract_mineru_text(self, block: Dict) -> str:
        """从 MinerU 块提取纯文本。"""
        line_texts = []
        for line in block.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                if span.get("type") == "text" and span.get("content"):
                    parts.append(span["content"])
            if parts:
                line_texts.append("".join(parts))
        return "".join(line_texts)

    def _extract_mineru_table_html(self, block: Dict) -> str:
        """从 MinerU 表格块提取 HTML。"""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("html"):
                    return span["html"]

        for sub_block in block.get("blocks", []):
            html = self._extract_mineru_table_html(sub_block)
            if html:
                return html
        return ""

    def _html_to_markdown(self, html: str) -> str:
        """HTML 表格转 Markdown。"""
        if not html:
            return ""
        try:
            dfs = pd.read_html(StringIO(html))
            if dfs:
                return dfs[0].to_markdown(index=False)
        except Exception:
            pass
        return html

    def _resolve_report_stem(self, report_path: Path) -> str:
        """从 MinerU 文件名解析报告主名。"""
        stem = report_path.stem
        if stem.startswith("MinerU_"):
            stem = stem[len("MinerU_") :].rsplit("__", 1)[0]
        return stem

    def _resolve_output_name(self, report_path: Path, report_data: Dict) -> str:
        """确定输出 JSON 文件名（优先使用 file_name 主名）。"""
        metainfo = report_data.get("metainfo", {})
        if metainfo.get("file_name"):
            return os.path.splitext(metainfo["file_name"])[0] + ".json"
        if metainfo.get("sha1_name"):
            return f"{metainfo['sha1_name']}.json"
        return self._resolve_report_stem(report_path) + ".json"

    def _load_metainfo_from_subset(self, subset_csv: Optional[Path]) -> Dict[str, Dict]:
        """从 subset.csv 建立文件名到 metainfo 的映射。"""
        file2meta = {}
        if subset_csv is None or not subset_csv.exists():
            return file2meta

        try:
            df = pd.read_csv(subset_csv, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(subset_csv, encoding="gbk")

        if "file_name" not in df.columns:
            return file2meta

        for _, row in df.iterrows():
            file_no_ext = os.path.splitext(str(row["file_name"]))[0]
            file2meta[file_no_ext] = {
                "sha1": str(row.get("sha1", "")),
                "company_name": str(row.get("company_name", "")),
                "file_name": str(row["file_name"]),
            }

        return file2meta


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    json_path = (
        root
        / "data/stock_data/debug_data/MinerU_【财报】中芯国际：中芯国际2024年年度报告__20260520083937.json"
    )
    output_dir = root / "data/stock_data/debug_data/02_merged_reports"
    subset_csv = root / "data/stock_data/subset.csv"

    merger = MinerUReportMerger()
    reports = merger.process_reports(
        reports_paths=[json_path],
        output_dir=output_dir,
        subset_csv=subset_csv,
    )
    report = reports[0]
    print(f"公司: {report['metainfo'].get('company_name')}")
    print(f"页数: {len(report['content']['pages'])}")
    print(f"首段预览:\n{report['content']['pages'][0]['text'][:400]}...")
