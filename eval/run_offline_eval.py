#!/usr/bin/env python3
"""
RAG 离线批量评估 CLI。

用法：
    python eval/run_offline_eval.py
    python eval/run_offline_eval.py --dataset eval/data/eval_dataset.csv --top-k 6
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.evaluation import EvalDataset, RAGEvaluator
from eval.report import METRIC_COLUMNS, generate_markdown_report, save_detail_csv

DEFAULT_DATASET = ROOT_DIR / "eval" / "data" / "eval_dataset.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval" / "reports"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="RAG 离线批量评估")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"评估集路径（.csv 或 .json），默认 {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="page_recall@k 的评估截止位置，默认 6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"报告输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="输出文件名前缀；默认使用时间戳 YYYYMMDD_HHMMSS",
    )
    return parser.parse_args()


def main() -> int:
    """执行离线评估并生成报告。"""
    args = parse_args()
    dataset_path = args.dataset.resolve()

    if not dataset_path.exists():
        print(f"[错误] 评估集不存在: {dataset_path}", file=sys.stderr)
        return 1

    run_timestamp = datetime.now()
    prefix = args.prefix or run_timestamp.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve()
    detail_path = output_dir / f"{prefix}_detail.csv"
    report_path = output_dir / f"{prefix}_report.md"

    print(f"[信息] 加载评估集: {dataset_path}")
    dataset = EvalDataset.from_path(dataset_path)
    print(f"[信息] 样本数: {len(dataset)}")

    print(f"[信息] 开始批量评估 (top_k={args.top_k})...")
    evaluator = RAGEvaluator()
    df = evaluator.run_batch(dataset, top_k=args.top_k)

    print(f"[信息] 写入明细 CSV: {detail_path}")
    save_detail_csv(df, detail_path)

    print(f"[信息] 生成 Markdown 报告: {report_path}")
    generate_markdown_report(
        df,
        dataset_path=dataset_path,
        top_k=args.top_k,
        output_path=report_path,
        run_timestamp=run_timestamp,
    )

    print("\n[汇总]")
    for col in METRIC_COLUMNS:
        if col in df.columns:
            mean_val = df[col].mean(skipna=True)
            mean_str = f"{mean_val:.4f}" if pd.notna(mean_val) else "N/A"
            print(f"  {col}: {mean_str}")

    print(f"\n[完成] 报告已保存至 {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
