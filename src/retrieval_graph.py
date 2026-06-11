import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from typing import List
from typing_extensions import TypedDict
from pathlib import Path

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START

from src.retrieval import HybridRetriever
from src.post_retrieval_correction import retrieval_grader
from src.config import settings
from src.api_requests import APIProcessor


class GraphState(TypedDict):
    """
    表示图的状态
    属性：
        question：用户的问题
        company_name：目标公司名称
        documents：文档列表
        retrieval_attempts：检索尝试次数
        has_relevant_docs：是否有相关文档
    """
    question: str
    company_name: str
    documents: List[Document]
    retrieval_attempts: int
    has_relevant_docs: bool


# 初始化混合检索器
_vector_db_dir = Path(settings.chroma_persist_dir) if settings.chroma_persist_dir else Path("data/stock_data/databases/vector_dbs")
_documents_dir = Path(settings.reports_input_dir) if settings.reports_input_dir else Path("data/stock_data/databases/chunked_reports")

_hybrid_retriever = HybridRetriever(
    vector_db_dir=_vector_db_dir,
    documents_dir=_documents_dir,
)


class QuestionRewriter:
    """查询重写器：基于LLM重写用户问题以改进检索效果。"""

    def __init__(self, provider: str = "openai"):
        self.processor = APIProcessor(provider=provider)
        self.system_content = (
            "你是一个查询重写专家。给定一个用户问题，请生成一个改进的查询版本，"
            "使其更适合用于文档检索。保持问题的核心意图，但使用更明确、更具体的表达方式。"
            "直接返回重写后的问题，不要添加任何解释或前缀。"
        )

    def invoke(self, inputs: dict) -> str:
        """兼容 LangChain / LangGraph 的 Runnable 接口。

        参数：
            inputs: 字典，必须包含 "question" 键
        返回：
            重写后的查询字符串
        """
        question = inputs["question"]
        result = self.processor.send_message(
            system_content=self.system_content,
            human_content=f"原始问题：{question}\n\n请重写上述问题以改进检索效果。",
            is_structured=False,
        )
        if isinstance(result, dict):
            return result.get("final_answer", str(result)).strip()
        return str(result).strip()


# 默认实例，供工作流节点直接调用
question_rewriter = QuestionRewriter()


def retrieve(state: GraphState) -> dict:
    """使用混合检索模块检索与问题相关的文档。

    参数：
        state: 当前图状态
    返回：
        状态更新片段，包含 documents 和自增后的 retrieval_attempts
    """
    print("---检索---")
    question = state["question"]
    company_name = state["company_name"]
    attempts = state.get("retrieval_attempts", 0)

    results = _hybrid_retriever.retrieve(
        company_name=company_name,
        query=question,
        top_n=6,
        return_parent_pages=True,
    )

    documents: List[Document] = []
    for r in results:
        metadata = {k: v for k, v in r.items() if k != "text"}
        documents.append(Document(page_content=r["text"], metadata=metadata))

    print(f"---检索完成，获取 {len(documents)} 篇文档---")
    return {
        "documents": documents,
        "retrieval_attempts": attempts + 1,
    }


def grade_documents(state: GraphState) -> dict:
    """确定检索到的文档是否与问题相关，过滤掉不相关文档。

    参数：
        state: 当前图状态
    返回：
        状态更新片段，包含过滤后的 documents 和 has_relevant_docs
    """
    print("---检查文档与问题的相关性---")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("retrieval_attempts", 0)

    filtered_docs: List[Document] = []
    has_relevant = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---评分：文档相关---")
            filtered_docs.append(d)
            has_relevant = True
        else:
            print("---评分：文档不相关---")

    # 第二次检索后若仍无相关文档，保留原始检索结果（不过滤为空）
    if not has_relevant and attempts >= 2:
        print("---第二次检索仍无相关文档，保留原始检索结果---")
        return {"documents": documents, "has_relevant_docs": False}

    return {"documents": filtered_docs, "has_relevant_docs": has_relevant}


def transform_query(state: GraphState) -> dict:
    """基于当前状态重写问题以改进搜索效果。

    参数：
        state: 当前图状态
    返回：
        状态更新片段，包含重写后的 question
    """
    print("---转换查询---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    print(f"---重写后的问题：{better_question}---")
    return {"question": better_question}


def decide_next_step(state: GraphState) -> str:
    """基于当前状态决定下一步操作。

    参数：
        state: 当前图状态
    返回：
        下一个节点名称（"end" 或 "transform_query"）
    """
    print("---评估已评分文档---")
    has_relevant = state["has_relevant_docs"]
    attempts = state.get("retrieval_attempts", 0)

    if has_relevant:
        print("---决策：有相关文档，直接返回---")
        return "end"
    if attempts == 1:
        print("---决策：无相关文档，转换查询后重新检索---")
        return "transform_query"

    print("---决策：第二次检索仍无相关文档，返回当前检索结果---")
    return "end"


# 初始化工作流状态图
workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)

# 构建图的边（连接）
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {
        "end": END,
        "transform_query": "transform_query",
    },
)
workflow.add_edge("transform_query", "retrieve")

# 编译整个工作流
app = workflow.compile()

if __name__ == "__main__":
    from pprint import pprint

    # 设置输入问题
    inputs = {
        "question": "中芯国际集成电路制造有限公司",
        "company_name": "",
        "documents": [],
        "retrieval_attempts": 0,
        "has_relevant_docs": False,
    }

    # 运行程序并处理输出
    final_state = None
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"节点 '{key}':")
            final_state = value

        pprint("\n---\n")

    # 输出最终返回的文档
    pprint("最终返回的文档：")
    if final_state and final_state.get("documents"):
        for i, doc in enumerate(final_state["documents"], 1):
            pprint(
                f"\n[{i}] page={doc.metadata.get('page', 'N/A')} "
                f"distance={doc.metadata.get('distance', 'N/A')}"
            )
            pprint(doc.page_content[:300] + "...")
    else:
        pprint("无文档")
