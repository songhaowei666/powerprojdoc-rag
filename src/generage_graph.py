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
    }

输出：
    {
        "question": str,
        "documents": List[Document],
        "generation": str,
        "generation_attempts": int,
    }

注意：本模块不执行检索，文档列表由上游模块（如 retrieval_graph）提供。
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
    """

    question: str
    documents: List[Document]
    generation: str
    generation_attempts: int


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
def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    判断生成内容是否基于文档并回答问题。

    参数：
        state: 当前图状态

    返回：
        下一个节点的决策："useful" / "not useful" / "not supported"
    """
    print("---评估生成质量---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempts = state.get("generation_attempts", 0)

    # 1. 幻觉检测
    grade = _default_grader.check_hallucination(documents, generation)

    if grade == "yes":
        print("---决策：生成内容基于文档---")

        # 2. 答案相关性检测
        grade = _default_grader.check_answer(question, generation)

        if grade == "yes":
            print("---决策：生成内容回答了问题---")
            return "useful"
        print("---决策：生成内容未回答问题---")
        return "not useful"

    print("---决策：生成内容未基于文档，重试---")

    # 如果已经连续多次生成均出现幻觉，改为走查询改写分支，避免原地死循环
    if attempts >= 2:
        return "not useful"
    return "not supported"


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
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
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
    from pprint import pprint

    # 示例：使用空/占位文档列表演示图结构
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
    }

    final_state = None
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"节点 '{key}' 完成")
            final_state = value
        pprint("---")

    if final_state:
        print("\n最终生成答案：")
        print(final_state.get("generation", "N/A"))
