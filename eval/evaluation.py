"""
RAG 评估模块。

提供离线批量评估与单条实时评估两种能力：
1. RAGEvaluator：在评测集上批量执行 RAG，计算 page_precision@k 与 ragas 标准指标
2. SingleTurnEvaluator：对单次 RAG 调用进行轻量打分，可用于线上监控

用法：
    from eval.evaluation import EvalDataset, RAGEvaluator

    dataset = EvalDataset.from_json(Path("eval/eval_dataset.json"))
    evaluator = RAGEvaluator()
    df = evaluator.run_batch(dataset, top_k=6)
    print(df[["question", "page_precision@k", "faithfulness"]])
"""

import sys
from pathlib import Path
from typing import List, Optional

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

def compute_page_precision_at_k(
    documents: List[Document],
    expected_pages: List[int],
    top_k: int = 6,
) -> Optional[float]:
    """计算页面精确率 Page Precision@K。

    取前 top_k 个检索结果，统计其中页面号命中预期页面集合的比例。
    不足 top_k 个结果时，空缺位置视为未命中，分母固定为 top_k。

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
    hits = 0
    for i in range(top_k):
        if i < len(documents):
            page = documents[i].metadata.get("page")
            if page is not None and page in expected_set:
                hits += 1

    return hits / top_k


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
        self.rag_app = rag_app if rag_app is not None else RAGApp()
        self.evaluator_llm = evaluator_llm if evaluator_llm is not None else build_llm(temperature=0)

    def run_batch(self, dataset: EvalDataset, top_k: int = 6) -> pd.DataFrame:
        """对评估集执行批量评估。

        参数：
            dataset: 评估集
            top_k: 页面精确率的评估截止位置

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
            company_name = sample.get("company_name", "")
            try:
                result = self.rag_app.run(question, company_name)
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

        # 2. 计算 page_precision@k
        page_precisions: List[Optional[float]] = []
        for sample, result in zip(samples, rag_results):
            docs = result.get("documents", [])
            expected_pages = sample.get("expected_source_pages", [])
            pp = compute_page_precision_at_k(docs, expected_pages, top_k)
            page_precisions.append(pp)

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
                "page_precision@k": page_precisions,
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
            expected_answer: 预期回答
            expected_pages: 预期页面号列表
            top_k: 页面精确率的评估截止位置

        返回：
            包含各指标的字典，键包括 page_precision@k、context_precision、
            context_recall、faithfulness、answer_relevancy、answer_correctness

        异常：
            RuntimeError: ragas 评估失败时抛出
        """
        expected_pages = expected_pages or []
        page_precision = compute_page_precision_at_k(documents, expected_pages, top_k)

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": question,
                    "response": generation,
                    "retrieved_contexts": [d.page_content for d in documents],
                    "reference": expected_answer,
                }
            ]
        )

        try:
            ragas_result = evaluate(
                dataset,
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

        result = {
            "page_precision@k": page_precision,
        }
        for col in [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ]:
            result[col] = (
                ragas_df[col].iloc[0] if col in ragas_df.columns else None
            )

        return result
