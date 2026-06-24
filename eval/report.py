"""
RAG 离线评估报告生成。

将 RAGEvaluator.run_batch 返回的 DataFrame 导出为明细 CSV 与 Markdown 汇总报告。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import settings

# 数值型评估指标列
METRIC_COLUMNS: List[str] = [
    "page_recall@k",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]

# 低分阈值
LOW_SCORE_THRESHOLD = 0.5

# generation 摘要截断长度
GENERATION_SUMMARY_MAX_LEN = 200


def _truncate_text(text: str, max_len: int = GENERATION_SUMMARY_MAX_LEN) -> str:
    """截断文本并追加省略号。"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def save_detail_csv(df: pd.DataFrame, path: Path) -> None:
    """将评估明细 DataFrame 写入 CSV。

    参数：
        df: run_batch 返回的 DataFrame
        path: 输出 CSV 路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    if "expected_source_pages" in export_df.columns:
        export_df["expected_source_pages"] = export_df["expected_source_pages"].apply(
            lambda pages: json.dumps(pages, ensure_ascii=False)
            if isinstance(pages, list)
            else pages
        )
    export_df.to_csv(path, index=False, encoding="utf-8")


def _compute_summary(df: pd.DataFrame) -> dict:
    """计算各指标的均值与 NaN 样本数。"""
    summary: dict = {"sample_count": len(df), "failed_count": 0}
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        summary[f"{col}_mean"] = series.mean(skipna=True)
        summary[f"{col}_nan_count"] = int(series.isna().sum())
        summary["failed_count"] = max(summary["failed_count"], int(series.isna().sum()))
    return summary


def _find_low_score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """找出任一指标低于阈值或含 NaN 的样本。"""
    mask = pd.Series(False, index=df.index)
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        col_mask = df[col].isna() | (df[col] < LOW_SCORE_THRESHOLD)
        mask = mask | col_mask
    return df[mask]


def generate_markdown_report(
    df: pd.DataFrame,
    *,
    dataset_path: Path,
    top_k: int,
    output_path: Path,
    run_timestamp: Optional[datetime] = None,
) -> str:
    """生成 Markdown 评估报告并写入文件。

    参数：
        df: run_batch 返回的 DataFrame
        dataset_path: 评估集路径
        top_k: 页面召回率评估截止位置
        output_path: 输出 Markdown 路径
        run_timestamp: 运行时间戳；为 None 时使用当前时间

    返回：
        生成的 Markdown 文本
    """
    run_timestamp = run_timestamp or datetime.now()
    summary = _compute_summary(df)
    low_score_df = _find_low_score_rows(df)

    lines: List[str] = [
        "# RAG 离线评估报告",
        "",
        "## 运行信息",
        "",
        f"- 运行时间: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评估集: `{dataset_path}`",
        f"- 样本数: {summary['sample_count']}",
        f"- top_k: {top_k}",
        f"- 评估模型: {settings.chat_model or '（未配置，使用 generage_graph 默认）'}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 均值 | NaN 样本数 |",
        "|------|------|------------|",
    ]

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        mean_val = summary.get(f"{col}_mean")
        nan_count = summary.get(f"{col}_nan_count", 0)
        mean_str = f"{mean_val:.4f}" if pd.notna(mean_val) else "N/A"
        lines.append(f"| {col} | {mean_str} | {nan_count} |")

    lines.extend(["", "## 分项明细", ""])
    for idx, row in df.iterrows():
        lines.append(f"### 样本 {idx + 1}")
        lines.append("")
        lines.append(f"- **问题**: {row.get('question', '')}")
        lines.append(f"- **预期答案**: {row.get('expected_answer', '')}")
        lines.append(f"- **生成答案**: {_truncate_text(str(row.get('generation', '')))}")
        lines.append(f"- **预期页面**: {row.get('expected_source_pages', [])}")
        lines.append("")
        lines.append("| 指标 | 得分 |")
        lines.append("|------|------|")
        for col in METRIC_COLUMNS:
            if col not in df.columns:
                continue
            val = row[col]
            val_str = f"{val:.4f}" if pd.notna(val) else "N/A"
            lines.append(f"| {col} | {val_str} |")
        lines.append("")

    lines.extend(["## 低分样本", ""])
    if len(low_score_df) == 0:
        lines.append("无低分或失败样本。")
    else:
        lines.append(
            f"共 {len(low_score_df)} 条样本任一指标 < {LOW_SCORE_THRESHOLD} 或评估失败（NaN）："
        )
        lines.append("")
        for idx, row in low_score_df.iterrows():
            low_metrics = []
            for col in METRIC_COLUMNS:
                if col not in df.columns:
                    continue
                val = row[col]
                if pd.isna(val):
                    low_metrics.append(f"{col}=N/A")
                elif val < LOW_SCORE_THRESHOLD:
                    low_metrics.append(f"{col}={val:.4f}")
            lines.append(
                f"- 样本 {idx + 1}: {row.get('question', '')} "
                f"（{', '.join(low_metrics)}）"
            )

    content = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content
