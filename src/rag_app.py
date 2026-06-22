"""
最终 RAG 应用入口。

将 `src.retrieval_graph` 与 `src.generage_graph` 串联：
1. 根据用户问题与公司编码，通过 `retrieval_graph` 执行混合检索，召回相关文档；
2. 将检索到的文档与用户问题传入 `generage_graph`，生成带自检的答案；
3. 若生成图返回 `should_retry_retrieval=True`，使用改写后的问题重新检索并生成（受 max_rag_rounds 限制）。

用法：
    from src.rag_app import RAGApp, run_rag

    result = run_rag(
        question="工程总投资是多少？",
        company_code="001",
    )
    print(result["generation"])
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from typing import List

from langchain_core.documents import Document

from src.generage_graph import app as generation_app
from src.retrieval_graph import app as retrieval_app


class RAGApp:
    """最终 RAG 应用：检索 + 生成。"""

    def __init__(
        self,
        retrieval_app=retrieval_app,
        generation_app=generation_app,
    ):
        """
        初始化 RAG 应用。

        参数：
            retrieval_app: 检索图实例，默认使用 src.retrieval_graph.app
            generation_app: 生成图实例，默认使用 src.generage_graph.app
        """
        self.retrieval_app = retrieval_app
        self.generation_app = generation_app

    def run(
        self,
        question: str,
        company_code: str = "001",
        is_direct_retrieve: bool = False,
        is_direct_generate: bool = False,
        max_rag_rounds: int = 2,
    ) -> dict:
        """
        执行完整 RAG 流程。

        参数：
            question: 用户问题
            company_code: 目标公司编码，用于检索过滤；为空时不按公司过滤
            is_direct_retrieve: 为 True 时跳过检索相关性评估，直接返回混合检索结果
            is_direct_generate: 为 True 时跳过生成质量评估，直接基于文档返回答案
            max_rag_rounds: 外层「检索+生成」最大轮数，默认 2

        返回：
            包含以下键的字典：
                - question: 原始问题
                - working_question: 最后一轮使用的工作问题（可能已改写）
                - documents: 检索到的文档列表（List[Document]）
                - generation: 最终生成的答案
                - has_relevant_docs: 检索文档是否相关（直接检索模式下未评估）
                - is_direct_retrieve: 是否使用了直接检索模式
                - is_grounded_in_docs: 生成是否基于检索文档（直接生成模式下未评估）
                - is_question_answered: 生成是否回答了用户问题（直接生成模式下未评估）
                - is_direct_generate: 是否使用了直接生成模式
                - should_retry_retrieval: 最后一轮生成图是否仍建议重检索
                - failure_reason: 最后一轮失败原因（ok | hallucination | not_answered | skipped）
                - rag_rounds: 实际执行的外层轮数

        异常：
            ValueError: question 为空或 max_rag_rounds < 1 时抛出
            RuntimeError: 检索或生成阶段发生非预期错误时抛出
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")
        if max_rag_rounds < 1:
            raise ValueError("max_rag_rounds 必须大于等于 1")

        original_question = question.strip()
        working_question = original_question
        rag_round = 0
        retrieval_state: dict = {}
        generation_state: dict = {}

        while rag_round < max_rag_rounds:
            rag_round += 1

            retrieval_inputs = {
                "question": working_question,
                "company_code": company_code,
                "documents": [],
                "retrieval_attempts": 0,
                "has_relevant_docs": False,
                "is_direct_retrieve": is_direct_retrieve,
            }

            try:
                print("=" * 60)
                print(f"开始检索阶段（第 {rag_round} 轮）")
                print("=" * 60)
                retrieval_state = self.retrieval_app.invoke(retrieval_inputs)
            except Exception as exc:
                raise RuntimeError(f"检索阶段执行失败: {exc}") from exc

            documents: List[Document] = retrieval_state.get("documents", [])
            if not documents:
                print("警告：未检索到任何文档，将基于空上下文生成答案")

            generation_inputs = {
                "question": working_question,
                "documents": documents,
                "generation": "",
                "generation_attempts": 0,
                "is_grounded_in_docs": False,
                "is_question_answered": False,
                "is_direct_generate": is_direct_generate,
                "should_retry_retrieval": False,
                "failure_reason": "skipped",
            }

            try:
                print("=" * 60)
                print(f"开始生成阶段（第 {rag_round} 轮）")
                print("=" * 60)
                generation_state = self.generation_app.invoke(generation_inputs)
            except Exception as exc:
                raise RuntimeError(f"生成阶段执行失败: {exc}") from exc

            if not generation_state.get("should_retry_retrieval", False):
                break

            working_question = generation_state.get("question", working_question)

        if (
            generation_state.get("should_retry_retrieval", False)
            and rag_round >= max_rag_rounds
        ):
            print(
                "警告：已达 max_rag_rounds 上限，生成图仍建议重检索，"
                "返回当前轮结果"
            )

        documents = generation_state.get(
            "documents", retrieval_state.get("documents", [])
        )

        return {
            "question": original_question,
            "working_question": generation_state.get("question", working_question),
            "documents": documents,
            "generation": generation_state.get("generation", ""),
            "has_relevant_docs": retrieval_state.get("has_relevant_docs", False),
            "is_direct_retrieve": retrieval_state.get(
                "is_direct_retrieve", is_direct_retrieve
            ),
            "is_grounded_in_docs": generation_state.get("is_grounded_in_docs", False),
            "is_question_answered": generation_state.get("is_question_answered", False),
            "is_direct_generate": generation_state.get(
                "is_direct_generate", is_direct_generate
            ),
            "should_retry_retrieval": generation_state.get(
                "should_retry_retrieval", False
            ),
            "failure_reason": generation_state.get("failure_reason", "skipped"),
            "rag_rounds": rag_round,
        }


# 默认实例，供便捷函数使用
default_rag_app = RAGApp()


def run_rag(
    question: str,
    company_code: str = "001",
    is_direct_retrieve: bool = False,
    is_direct_generate: bool = False,
    max_rag_rounds: int = 2,
) -> dict:
    """
    便捷函数：使用默认 RAG 应用执行查询。

    参数：
        question: 用户问题
        company_code: 目标公司编码
        is_direct_retrieve: 为 True 时跳过检索相关性评估
        is_direct_generate: 为 True 时跳过生成质量评估
        max_rag_rounds: 外层「检索+生成」最大轮数

    返回：
        包含 question、documents、generation 等的字典
    """
    return default_rag_app.run(
        question,
        company_code,
        is_direct_retrieve=is_direct_retrieve,
        is_direct_generate=is_direct_generate,
        max_rag_rounds=max_rag_rounds,
    )


if __name__ == "__main__":
    from pprint import pprint

    result = run_rag(
        question="工程总投资是多少？",
        company_code="001",
        is_direct_retrieve=True,
        is_direct_generate=True,
    )
    print("\n最终答案：")
    print(result["generation"])
    print(f"直接检索: {result.get('is_direct_retrieve')}")
    print(f"直接生成: {result.get('is_direct_generate')}")
    print(f"外层轮数: {result.get('rag_rounds')}")
