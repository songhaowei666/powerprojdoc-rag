"""
基于 LangGraph 的自适应生成工作流（Self-RAG 生成侧）。

职责：给定问题与检索到的文档列表，生成答案并对生成质量进行自检。
若出现幻觉或答案未回答问题，则重写问题后重新生成，最多重试固定次数。

输入：
    {
        "question": str,
        "documents": List[Document],
        "generation": "",
        "generation_attempts": 0,
        "is_grounded_in_docs": False,
        "is_question_answered": False,
        "is_direct_generate": False,
    }

输出：
    {
        "question": str,
        "documents": List[Document],
        "generation": str,
        "generation_attempts": int,
        "is_grounded_in_docs": bool,
        "is_question_answered": bool,
        "is_direct_generate": bool,
    }

注意：本模块不执行检索，文档列表由上游模块（如 retrieval_graph）提供。
当 `is_direct_generate=True` 时，仅基于文档生成一次并返回，跳过质量评估与重试。
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

from src.config import settings


# ---------------------------------------------------------------------------
# 图状态定义
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    """
    表示生成工作流的状态。

    属性：
        question: 用户问题
        documents: 检索到的文档列表
        generation: LLM 生成内容
        generation_attempts: 已执行的生成尝试次数
        is_grounded_in_docs: 最终生成是否基于检索文档（直接生成模式下未评估，恒为 False）
        is_question_answered: 最终生成是否回答了用户问题（直接生成模式下未评估，恒为 False）
        is_direct_generate: 是否跳过质量评估，直接基于文档生成并返回
    """

    question: str
    documents: List[Document]
    generation: str
    generation_attempts: int
    is_grounded_in_docs: bool
    is_question_answered: bool
    is_direct_generate: bool


# ---------------------------------------------------------------------------
# 结构化输出模型
# ---------------------------------------------------------------------------
class GradeHallucinations(BaseModel):
    """判断生成内容是否基于检索文档的事实。"""

    binary_score: str = Field(
        description="答案是否基于给定文档，'yes' 或 'no'"
    )


class GradeAnswer(BaseModel):
    """判断生成内容是否回答了用户问题。"""

    binary_score: str = Field(
        description="答案是否解决了用户问题，'yes' 或 'no'"
    )


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------
def build_llm(temperature: float = 0) -> ChatOpenAI:
    """基于项目配置构建 ChatOpenAI 实例。"""
    return ChatOpenAI(
        model=settings.chat_model or "gpt-4o",
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    """将 Document 列表拼接为单一上下文字符串。"""
    return "\n\n".join(
        f"[文档 {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------
class RAGGenerator:
    """基于检索上下文的问题生成器。"""

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or build_llm(temperature=0)
        self.system_content = (
            "你是一个基于检索文档回答问题的助手。"
            "请严格根据下面提供的上下文回答问题，不要引入外部知识。"
            "如果上下文中没有足够信息，请明确说明无法回答。"
            "回答应简洁、准确、直接针对问题。"
        )
        self.user_template = (
            "以下是检索到的上下文：\n\n{context}\n\n"
            "---\n\n"
            "请根据上述上下文回答以下问题：\n{question}"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_content),
                ("human", self.user_template),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: dict) -> str:
        """
        根据上下文生成答案。

        参数：
            inputs: 包含 "context" 和 "question" 的字典

        返回：
            生成的答案字符串
        """
        return self.chain.invoke(inputs).strip()


_default_generator = RAGGenerator()


# ---------------------------------------------------------------------------
# 问题重写器
# ---------------------------------------------------------------------------
class QuestionRewriter:
    """查询重写器：基于 LLM 重写用户问题以改进生成效果。"""

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or build_llm(temperature=0)
        self.system_content = (
            "你是一个查询重写专家。给定一个用户问题，请生成一个改进的查询版本，"
            "使其更适合用于基于检索文档的问答。保持问题的核心意图，"
            "但使用更明确、更具体的表达方式。"
            "直接返回重写后的问题，不要添加任何解释或前缀。"
        )
        self.user_template = "原始问题：{question}\n\n请重写上述问题以改进检索与生成效果。"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_content),
                ("human", self.user_template),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: dict) -> str:
        """
        重写问题。

        参数：
            inputs: 包含 "question" 的字典

        返回：
            重写后的查询字符串
        """
        return self.chain.invoke(inputs).strip()


_default_question_rewriter = QuestionRewriter()


# ---------------------------------------------------------------------------
# 评分器
# ---------------------------------------------------------------------------
class GenerationGrader:
    """对生成内容进行幻觉检测与答案相关性评分。"""

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or build_llm(temperature=0)

    def check_hallucination(self, documents: List[Document], generation: str) -> str:
        """返回 'yes' 或 'no'，表示生成是否基于文档。"""
        system = (
            "你是一个评估 LLM 生成内容是否基于检索事实的评分器。\n"
            "给出 'yes' 或 'no' 的二元评分。'yes' 表示答案是基于/由给定文档支持。"
        )
        human = (
            f"事实集合：\n\n{format_docs(documents)}\n\n"
            f"LLM 生成内容：\n{generation}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", human)]
        )
        grader = prompt | self.llm.with_structured_output(GradeHallucinations)
        result = grader.invoke({})
        return result.binary_score

    def check_answer(self, question: str, generation: str) -> str:
        """返回 'yes' 或 'no'，表示生成是否回答问题。"""
        system = (
            "你是一个评估答案是否解决/回答问题的评分器。\n"
            "给出 'yes' 或 'no' 的二元评分。'yes' 表示答案解决了问题。"
        )
        human = f"用户问题：\n\n{question}\n\nLLM 生成内容：\n{generation}"
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", human)]
        )
        grader = prompt | self.llm.with_structured_output(GradeAnswer)
        result = grader.invoke({})
        return result.binary_score


_default_grader = GenerationGrader()


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------
def generate(state: GraphState) -> dict:
    """
    基于检索文档生成答案。

    参数：
        state: 当前图状态

    返回：
        更新后的 generation 与 generation_attempts
    """
    print("---生成答案---")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("generation_attempts", 0)

    context = format_docs(documents)
    generation = _default_generator.invoke({"context": context, "question": question})

    print(f"---生成完成（第 {attempts + 1} 次尝试）---")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_attempts": attempts + 1,
    }


def transform_query(state: GraphState) -> dict:
    """
    重写问题以改进后续生成效果。

    参数：
        state: 当前图状态

    返回：
        更新后的 question
    """
    print("---转换查询---")
    question = state["question"]
    better_question = _default_question_rewriter.invoke({"question": question})
    print(f"---重写后的问题：{better_question}---")
    return {"question": better_question}


# ---------------------------------------------------------------------------
# 条件边函数
# ---------------------------------------------------------------------------
def grade_generation(state: GraphState) -> dict:
    """
    评估生成质量，并将结果写入 state 标识字段。

    参数：
        state: 当前图状态

    返回：
        is_grounded_in_docs、is_question_answered 的更新
    """
    print("---评估生成质量---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    is_grounded = _default_grader.check_hallucination(documents, generation) == "yes"
    is_answered = False

    if is_grounded:
        print("---决策：生成内容基于文档---")
        is_answered = _default_grader.check_answer(question, generation) == "yes"
        if is_answered:
            print("---决策：生成内容回答了问题---")
        else:
            print("---决策：生成内容未回答问题---")
    else:
        print("---决策：生成内容未基于文档---")

    return {
        "is_grounded_in_docs": is_grounded,
        "is_question_answered": is_answered,
    }


def route_after_grade(state: GraphState) -> str:
    """
    根据评分标识与尝试次数决定下一步。

    参数：
        state: 当前图状态

    返回：
        下一个节点的决策："useful" / "not useful" / "not supported"
    """
    is_grounded = state["is_grounded_in_docs"]
    is_answered = state["is_question_answered"]
    attempts = state.get("generation_attempts", 0)

    if is_grounded and is_answered:
        return "useful"

    if not is_grounded:
        print("---决策：生成内容未基于文档，重试---")
        if attempts >= 2:
            return "not useful"
        return "not supported"

    print("---决策：生成内容未回答问题，改写查询---")
    return "not useful"


def route_after_generate(state: GraphState) -> str:
    """
    生成后决定进入质量评估还是直接结束。

    参数：
        state: 当前图状态

    返回：
        "grade" 或 "end"
    """
    if state.get("is_direct_generate", False):
        print("---直接生成模式：跳过质量评估---")
        return "end"
    return "grade"


def should_continue(state: GraphState) -> str:
    """
    在重新生成前检查是否超过最大尝试次数，防止无限循环。

    参数：
        state: 当前图状态

    返回：
        下一节点名称："generate" 或 "end"
    """
    attempts = state.get("generation_attempts", 0)
    if attempts >= 3:
        print("---已达到最大生成尝试次数，结束工作流---")
        return "end"
    return "generate"


# ---------------------------------------------------------------------------
# 构建工作流
# ---------------------------------------------------------------------------
workflow = StateGraph(GraphState)

workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "generate")
workflow.add_conditional_edges(
    "generate",
    route_after_generate,
    {
        "end": END,
        "grade": "grade_generation",
    },
)
workflow.add_conditional_edges(
    "grade_generation",
    route_after_grade,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)
workflow.add_conditional_edges(
    "transform_query",
    should_continue,
    {
        "generate": "generate",
        "end": END,
    },
)

app = workflow.compile()


# ---------------------------------------------------------------------------
# 本地调试入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成工作流本地调试")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="直接基于文档生成，跳过质量评估与重试",
    )
    args = parser.parse_args()

    docs = [
        Document(
            page_content="中芯国际 2024 年营业收入为 577.96 亿元，同比增长 27.0%。",
            metadata={"page": 1, "file_name": "demo.pdf"},
        )
    ]

    inputs = {
        "question": "中芯国际2024年营业收入是多少？",
        "documents": docs,
        "generation": "",
        "generation_attempts": 0,
        "is_grounded_in_docs": False,
        "is_question_answered": False,
        "is_direct_generate": True,
    }

    final_state = app.invoke(inputs)

    print("\n最终生成答案：")
    print(final_state.get("generation", "N/A"))
    print(f"直接生成模式: {final_state.get('is_direct_generate')}")
    if not final_state.get("is_direct_generate"):
        print(f"基于文档: {final_state.get('is_grounded_in_docs')}")
        print(f"回答问题: {final_state.get('is_question_answered')}")
