"""
最终 RAG 应用入口。

将 `src.retrieval_graph` 与 `src.generage_graph` 串联：
1. 根据用户问题与公司名，通过 `retrieval_graph` 执行混合检索，召回相关文档；
2. 将检索到的文档与用户问题传入 `generage_graph`，生成带自检的答案。

用法：
    from src.rag_app import RAGApp, run_rag

    result = run_rag(
        question="中芯国际2024年营业收入是多少？",
        company_name="中芯国际集成电路制造有限公司",
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

    def run(self, question: str, company_name: str = "") -> dict:
        """
        执行完整 RAG 流程。

        参数：
            question: 用户问题
            company_name: 目标公司名称，用于检索过滤；为空时不按公司过滤

        返回：
            包含以下键的字典：
                - question: 原始问题
                - documents: 检索到的文档列表（List[Document]）
                - generation: 最终生成的答案

        异常：
            ValueError: question 为空时抛出
            RuntimeError: 检索或生成阶段发生非预期错误时抛出
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        # 1. 检索阶段
        retrieval_inputs = {
            "question": question,
            "company_name": company_name,
            "documents": [],
            "retrieval_attempts": 0,
            "has_relevant_docs": False,
        }

        try:
            print("=" * 60)
            print("开始检索阶段")
            print("=" * 60)
            retrieval_state = self.retrieval_app.invoke(retrieval_inputs)
        except Exception as exc:
            raise RuntimeError(f"检索阶段执行失败: {exc}") from exc

        documents: List[Document] = retrieval_state.get("documents", [])
        if not documents:
            print("警告：未检索到任何文档，将基于空上下文生成答案")

        # 2. 生成阶段
        generation_inputs = {
            "question": question,
            "documents": documents,
            "generation": "",
            "generation_attempts": 0,
        }

        try:
            print("=" * 60)
            print("开始生成阶段")
            print("=" * 60)
            generation_state = self.generation_app.invoke(generation_inputs)
        except Exception as exc:
            raise RuntimeError(f"生成阶段执行失败: {exc}") from exc

        return {
            "question": question,
            "documents": generation_state.get("documents", documents),
            "generation": generation_state.get("generation", ""),
        }


# 默认实例，供便捷函数使用
default_rag_app = RAGApp()


def run_rag(question: str, company_name: str = "") -> dict:
    """
    便捷函数：使用默认 RAG 应用执行查询。

    参数：
        question: 用户问题
        company_name: 目标公司名称

    返回：
        包含 question、documents、generation 的字典
    """
    return default_rag_app.run(question, company_name)


if __name__ == "__main__":
    from pprint import pprint

    result = run_rag(
        question="中芯国际2024年营业收入是多少？",
        company_name="中芯国际集成电路制造有限公司",
    )
    print("\n最终答案：")
    print(result["generation"])
