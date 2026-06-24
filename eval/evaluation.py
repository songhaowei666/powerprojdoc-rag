"""
RAG 评估模块。

提供离线批量评估与单条实时评估两种能力：
1. RAGEvaluator：在评测集上批量执行 RAG，计算 page_recall@k 与 ragas 标准指标
2. SingleTurnEvaluator：对单次 RAG 调用进行轻量打分，可用于线上监控

用法：
    from eval.evaluation import EvalDataset, RAGEvaluator

    dataset = EvalDataset.from_json(Path("eval/eval_dataset.json"))
    evaluator = RAGEvaluator()
    df = evaluator.run_batch(dataset, top_k=6)
    print(df[["question", "page_recall@k", "faithfulness"]])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics.collections import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import settings
from src.generage_graph import build_llm

if TYPE_CHECKING:
    from src.rag_app import RAGApp


# ---------------------------------------------------------------------------
# EvalDataset
# ---------------------------------------------------------------------------

class EvalDataset:
    """评估集加载与管理。"""

    def __init__(self, samples: List[dict]):
        self._samples = list(samples)

    @classmethod
    def from_json(cls, path: Path) -> "EvalDataset":
        """从 JSON 文件加载评估集。

        参数：
            path: JSON 文件路径，顶层必须为对象数组

        返回：
            EvalDataset 实例

        异常：
            ValueError: 文件内容不是列表时抛出
        """
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"评估集 JSON 顶层必须是数组， got {type(data).__name__}")
        return cls(data)

    @classmethod
    def from_csv(cls, path: Path) -> "EvalDataset":
        """从 CSV 文件加载评估集。

        参数：
            path: CSV 文件路径

        返回：
            EvalDataset 实例

        说明：
            - expected_source_pages 列支持 JSON 数组字符串（如 "[3,4]"）
            - 若存在 company_name 列且无 company_code 列，自动映射为 company_code
        """
        import json

        header = pd.read_csv(path, nrows=0, encoding="utf-8")
        dtype: dict = {}
        for col in ("company_code", "company_name"):
            if col in header.columns:
                dtype[col] = str
        df = pd.read_csv(path, encoding="utf-8", dtype=dtype or None)
        if "company_name" in df.columns and "company_code" not in df.columns:
            df = df.rename(columns={"company_name": "company_code"})

        samples: List[dict] = []
        for _, row in df.iterrows():
            sample = row.to_dict()
            pages = sample.get("expected_source_pages")
            if isinstance(pages, str):
                sample["expected_source_pages"] = json.loads(pages)
            elif isinstance(pages, float) and pd.isna(pages):
                sample["expected_source_pages"] = []
            samples.append(sample)

        return cls(samples)

    @classmethod
    def from_path(cls, path: Path) -> "EvalDataset":
        """按文件后缀自动选择加载方式。"""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls.from_csv(path)
        if suffix == ".json":
            return cls.from_json(path)
        raise ValueError(f"不支持的评估集格式: {suffix}，仅支持 .csv 与 .json")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    def to_list(self) -> List[dict]:
        """返回样本列表的浅拷贝。"""
        return list(self._samples)


# ---------------------------------------------------------------------------
# 自定义检索指标
# ---------------------------------------------------------------------------

def compute_page_recall_at_k(
    documents: List[Document],
    expected_pages: List[int],
    top_k: int = 6,
) -> Optional[float]:
    """计算页面召回率 Page Recall@K。

    取前 top_k 个检索结果，统计其中命中的唯一页面号占预期页面集合的比例。
    同一页面在结果中多次出现只计一次命中。

    参数：
        documents: 检索到的文档列表
        expected_pages: 预期页面号列表
        top_k: 评估截止位置

    返回：
        float（0.0 ~ 1.0）或 None（expected_pages 为空时无法评估）
    """
    if not expected_pages:
        return None
    if top_k <= 0:
        return 0.0

    expected_set = set(expected_pages)
    hit_pages: set = set()
    for i in range(min(top_k, len(documents))):
        page = documents[i].metadata.get("page")
        if page is not None and page in expected_set:
            hit_pages.add(page)

    return len(hit_pages) / len(expected_set)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """RAG 批量离线评估器。"""

    def __init__(
        self,
        rag_app: Optional[RAGApp] = None,
        evaluator_llm: Optional[BaseChatModel] = None,
    ):
        """初始化评估器。

        参数：
            rag_app: 注入的 RAG 应用实例；为 None 时使用默认 RAGApp()
            evaluator_llm: 传给 ragas 的评估 LLM；为 None 时通过 build_llm(temperature=0) 构造
        """
        if rag_app is not None:
            self.rag_app = rag_app
        else:
            from src.rag_app import RAGApp

            self.rag_app = RAGApp()
        self.evaluator_llm = evaluator_llm if evaluator_llm is not None else build_llm(temperature=0)

    def run_batch(self, dataset: EvalDataset, top_k: int = 6) -> pd.DataFrame:
        """对评估集执行批量评估。

        参数：
            dataset: 评估集
            top_k: 页面召回率的评估截止位置

        返回：
            包含所有评估指标的 DataFrame

        异常：
            RuntimeError: ragas 评估阶段整体失败时抛出
        """
        samples = dataset.to_list()
        rag_results: List[dict] = []
        failed_indices: List[int] = []

        # 1. 执行 RAG
        for idx, sample in enumerate(samples):
            question = sample.get("question", "")
            company_code = sample.get("company_code", "")
            try:
                result = self.rag_app.run(question, company_code)
                rag_results.append(result)
            except Exception as exc:
                print(f"[警告] 样本评估失败: question={question}, error={exc}")
                failed_indices.append(idx)
                rag_results.append(
                    {
                        "question": question,
                        "generation": "",
                        "documents": [],
                    }
                )

        # 2. 计算 page_recall@k
        page_recalls: List[Optional[float]] = []
        for sample, result in zip(samples, rag_results):
            docs = result.get("documents", [])
            expected_pages = sample.get("expected_source_pages", [])
            pr = compute_page_recall_at_k(docs, expected_pages, top_k)
            page_recalls.append(pr)

        # 3. 构建 ragas 数据集
        ragas_samples = []
        for sample, result in zip(samples, rag_results):
            ragas_samples.append(
                {
                    "user_input": sample.get("question", ""),
                    "response": result.get("generation", ""),
                    "retrieved_contexts": [
                        d.page_content for d in result.get("documents", [])
                    ],
                    "reference": sample.get("expected_answer", ""),
                }
            )

        ragas_dataset = EvaluationDataset.from_list(ragas_samples)

        # 4. ragas 批量评估
        try:
            ragas_result = evaluate(
                ragas_dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                    answer_correctness,
                ],
                llm=self.evaluator_llm,
            )
            ragas_df = ragas_result.to_pandas()
        except Exception as exc:
            raise RuntimeError(f"ragas 评估失败: {exc}") from exc

        # 5. 组装结果
        df = pd.DataFrame(
            {
                "question": [s.get("question", "") for s in samples],
                "expected_answer": [s.get("expected_answer", "") for s in samples],
                "generation": [r.get("generation", "") for r in rag_results],
                "expected_source_pages": [
                    s.get("expected_source_pages", []) for s in samples
                ],
                "page_recall@k": page_recalls,
            }
        )

        ragas_cols = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ]
        for col in ragas_cols:
            if col in ragas_df.columns:
                df[col] = ragas_df[col].values

        # RAG 失败的样本，ragas 指标置为 NaN
        for idx in failed_indices:
            for col in ragas_cols:
                if col in df.columns:
                    df.loc[idx, col] = float("nan")

        return df


class SingleTurnEvaluator:
    """单次 RAG 调用评估器，适用于实时/在线评估。"""

    def __init__(
        self,
        evaluator_llm: Optional[BaseChatModel] = None,
    ):
        """初始化评估器。

        参数：
            evaluator_llm: 传给 ragas 的评估 LLM；为 None 时通过 build_llm(temperature=0) 构造
        """
        self.evaluator_llm = evaluator_llm if evaluator_llm is not None else build_llm(temperature=0)

    def evaluate(
        self,
        question: str,
        generation: str,
        documents: List[Document],
        expected_answer: str = "",
        expected_pages: Optional[List[int]] = None,
        top_k: int = 6,
    ) -> dict:
        """对单次 RAG 调用进行完整评估。

        参数：
            question: 用户查询
            generation: RAG 生成的答案
            documents: 检索到的文档列表
            expected_answer: 预期回答；为空时不计算 `context_recall` 和
                `answer_correctness`
            expected_pages: 预期页面号列表；为 `None` 或空列表时不计算
                `page_recall@k`
            top_k: 页面召回率的评估截止位置

        返回：
            包含各指标的字典，键包括 page_recall@k、context_precision、
            context_recall、faithfulness、answer_relevancy、answer_correctness。
            当 `expected_answer` 为空时，`context_recall` 和
            `answer_correctness` 固定为 `None`。

        异常：
            RuntimeError: ragas 评估失败时抛出
        """
        expected_pages = expected_pages or []
        page_recall = compute_page_recall_at_k(documents, expected_pages, top_k)

        has_reference = bool(expected_answer and expected_answer.strip())
        sample: dict = {
            "user_input": question,
            "response": generation,
            "retrieved_contexts": [d.page_content for d in documents],
        }
        if has_reference:
            sample["reference"] = expected_answer

        dataset = EvaluationDataset.from_list([sample])

        metrics = [
            context_precision,
            faithfulness,
            answer_relevancy,
        ]
        if has_reference:
            metrics.extend([context_recall, answer_correctness])

        try:
            ragas_result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
            )
            ragas_df = ragas_result.to_pandas()
        except Exception as exc:
            raise RuntimeError(f"ragas 评估失败: {exc}") from exc

        result: dict = {
            "page_recall@k": page_recall,
        }
        for col in [
            "context_precision",
            "faithfulness",
            "answer_relevancy",
        ]:
            result[col] = (
                ragas_df[col].iloc[0] if col in ragas_df.columns else None
            )

        if has_reference:
            for col in ["context_recall", "answer_correctness"]:
                result[col] = (
                    ragas_df[col].iloc[0] if col in ragas_df.columns else None
                )
        else:
            result["context_recall"] = None
            result["answer_correctness"] = None

        return result
