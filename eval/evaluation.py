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
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from src.config import settings
from src.generage_graph import build_llm

if TYPE_CHECKING:
    from src.rag_app import RAGApp

RAGAS_COLS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def build_evaluator_embeddings() -> OpenAIEmbeddings:
    """基于项目配置构建 ragas 评估用 Embedding 实例。"""
    return OpenAIEmbeddings(
        model=settings.embedding_model or "text-embedding-3-large",
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base or None,
    )


def build_ragas_metrics(*, has_reference: bool) -> list:
    """构建 ragas 已实例化的标准指标列表（兼容 evaluate 的 llm 注入）。"""
    metrics = [
        context_precision,
        faithfulness,
        answer_relevancy,
    ]
    if has_reference:
        metrics.extend([context_recall, answer_correctness])
    return metrics


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
        evaluator_embeddings: Optional[Embeddings] = None,
    ):
        """初始化评估器。

        参数：
            rag_app: 注入的 RAG 应用实例；为 None 时使用默认 RAGApp()
            evaluator_llm: 传给 ragas 的评估 LLM；为 None 时通过 build_llm(temperature=0) 构造
            evaluator_embeddings: 传给 ragas 的 Embedding；为 None 时通过 build_evaluator_embeddings() 构造
        """
        if rag_app is not None:
            self.rag_app = rag_app
        else:
            from src.rag_app import RAGApp

            self.rag_app = RAGApp()
        self.evaluator_llm = evaluator_llm if evaluator_llm is not None else build_llm(temperature=0)
        self.evaluator_embeddings = (
            evaluator_embeddings
            if evaluator_embeddings is not None
            else build_evaluator_embeddings()
        )

    def run_batch(
        self,
        dataset: EvalDataset,
        top_k: int = 6,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """对评估集执行批量评估（逐条 RAG + ragas，节省 token 且可增量输出）。

        参数：
            dataset: 评估集
            top_k: 页面召回率的评估截止位置
            limit: 最多评估条数；为 None 时评估 offset 之后的全部样本
            offset: 起始样本索引（0-based）

        返回：
            包含所有评估指标的 DataFrame

        异常：
            RuntimeError: 单条 ragas 评估失败时抛出
        """
        all_samples = dataset.to_list()
        if offset < 0:
            raise ValueError("offset 不能小于 0")
        if offset >= len(all_samples):
            raise ValueError(f"offset={offset} 超出评估集范围（共 {len(all_samples)} 条）")

        end = offset + limit if limit is not None else len(all_samples)
        samples = all_samples[offset:end]
        single_evaluator = SingleTurnEvaluator(
            evaluator_llm=self.evaluator_llm,
            evaluator_embeddings=self.evaluator_embeddings,
        )
        rows: List[dict] = []

        for local_idx, sample in enumerate(samples):
            global_idx = offset + local_idx
            question = sample.get("question", "")
            company_code = sample.get("company_code", "")
            expected_answer = sample.get("expected_answer", "")
            expected_pages = sample.get("expected_source_pages", [])

            print(
                f"[信息] 评估样本 {local_idx + 1}/{len(samples)} "
                f"(索引 {global_idx}): {question[:40]}..."
            )

            try:
                rag_result = self.rag_app.run(question, company_code)
                generation = rag_result.get("generation", "")
                documents = rag_result.get("documents", [])
            except Exception as exc:
                print(f"[警告] 样本 RAG 失败: question={question}, error={exc}")
                row = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generation": "",
                    "expected_source_pages": expected_pages,
                    "page_recall@k": 0.0 if expected_pages else None,
                }
                for col in RAGAS_COLS:
                    row[col] = float("nan")
                rows.append(row)
                continue

            try:
                metrics = single_evaluator.evaluate(
                    question=question,
                    generation=generation,
                    documents=documents,
                    expected_answer=expected_answer,
                    expected_pages=expected_pages,
                    top_k=top_k,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ragas 评估失败 (样本索引 {global_idx}): {exc}"
                ) from exc

            row = {
                "question": question,
                "expected_answer": expected_answer,
                "generation": generation,
                "expected_source_pages": expected_pages,
                **metrics,
            }
            rows.append(row)
            print(
                f"  page_recall@k={metrics.get('page_recall@k')}, "
                f"faithfulness={metrics.get('faithfulness')}"
            )

        return pd.DataFrame(rows)


class SingleTurnEvaluator:
    """单次 RAG 调用评估器，适用于实时/在线评估。"""

    def __init__(
        self,
        evaluator_llm: Optional[BaseChatModel] = None,
        evaluator_embeddings: Optional[Embeddings] = None,
    ):
        """初始化评估器。

        参数：
            evaluator_llm: 传给 ragas 的评估 LLM；为 None 时通过 build_llm(temperature=0) 构造
            evaluator_embeddings: 传给 ragas 的 Embedding；为 None 时通过 build_evaluator_embeddings() 构造
        """
        self.evaluator_llm = evaluator_llm if evaluator_llm is not None else build_llm(temperature=0)
        self.evaluator_embeddings = (
            evaluator_embeddings
            if evaluator_embeddings is not None
            else build_evaluator_embeddings()
        )

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

        metrics = build_ragas_metrics(has_reference=has_reference)

        try:
            ragas_result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings,
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
