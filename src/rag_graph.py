from typing import List
from typing_extensions import TypedDict
class GraphState(TypedDict):
       """
              表示图的状态
       属性：
              question：用户的问题
              generation：语言模型生成的答案
              web_search：是否需要进行网络搜索以补充信息
              documents：文档列表
       """
       question:str
       generation:str
       web_search:str
       documents:List[str]

from langchain.schema import Document
from src.post_retrieval_correction import RetrievalGrader, retrieval_grader
def retrieve(state):
       """
       检索与问题相关的文档
       参数：
              state (dict)：当前图状态
       返回：
              state (dict)：更新后的图状态
              documents：添加了检索到的相关文档
       """
       print("---检索---")
       question = state["question"]
       # 检索
       documents = retriever.get_relevant_documents(question)
       return {"documents": documents, "question": question}
def generate(state):
       """
       生成答案
       参数：
              state (dict)：当前图状态
       返回：
              state (dict)：更新后的图状态
              generation：包含语言模型生成的内容
       """
       print("---生成---")
       question = state["question"]
       documents = state["documents"]
       # RAG生成
       generation = rag_chain.invoke({"context": documents, "question": question})
       return {"documents": documents, "question": question, "generation": generation}
def grade_documents(state):
       """
       确定检索到的文档是否与问题相关
       参数：
                     state (dict)：当前图状态
       返回：
              state (dict)：更新documents键，只保留经过筛选的相关文档
       """
       print("---检查文档与问题的相关性---")
       question = state["question"]
       documents = state["documents"]
       # 对每个文档评分
       filtered_docs = []
       web_search = "No"
       has_relevant_docs=False
       for d in documents:
              score = retrieval_grader.invoke (
                     {"question": question, "document": d.page_content}
              )
              grade = score.binary_score
              if grade == "yes":
                     print("---评分：文档相关---")
                     filtered_docs.append(d)
                     has_relevant_docs=True
              else:
                     print("---评分：文档不相关---")
                     if not has_relevant_docs:
                        web_search = "Yes"
              continue
       return {"documents": filtered_docs, "question": question, "web_search": web_search}
def transform_query(state):
       """
       基于当前状态重写问题以改进搜索效果
       参数：
              state (dict)：当前图状态
       返回：
              state (dict)：更新后的图状态，包含了重写的问题
       """
       print("---转换查询---")
       question = state["question"]
       documents = state["documents"]
       # 重写问题
       better_question = question_rewriter.invoke({"question": question})
       return {"documents": documents, "question": better_question}
def web_search(state):
       """
       使用网络搜索工具获取额外信息
       参数：
              state (dict)：包含当前状态
                     - question：问题
                                          - documents：文档列表
       返回：
              state (dict)：用追加的网络搜索结果更新documents键
       """
       print("---网络搜索---")
       question = state["question"]
       documents = state["documents"]
       # 网络搜索
       search_results = web_search_tool.invoke(question)
       # 将搜索结果列表转换为字符串
       search_results_str = "\n".join([str(result) for result in search_results])
       web_results = Document(page_content=search_results_str)
       documents.append(web_results)
       return {"documents": documents, "question": question}
# 边缘情况处理
def decide_to_generate(state):
       """
       基于当前状态决定下一步操作：是生成答案还是重写问题
       参数：
              state (dict)：当前图状态
       返回：
              str：下一个要调用的操作名称
       """
       print("---评估已评分文档---")
       state["question"]
       web_search = state["web_search"]
       state["documents"]
       if web_search == "Yes":
              # 所有文档都已被check_relevance过滤
              # 我们将重新生成一个新的查询
              print(
                     "---决策：所有文档与问题都不相关，转换查询---"
              )
              return "transform_query"
       else:
              # 因为我们有了相关文档，所以可以生成答案
              print("---决策：生成---")
              return "generate"
       

from langgraph.graph import END, StateGraph, START
# 初始化工作流状态图
workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("retrieve", retrieve)  # 检索文档
workflow.add_node("grade_documents", grade_documents)  # 给文档评分
workflow.add_node("generate", generate)  # 生成答案
workflow.add_node("transform_query", transform_query)  # 转换查询
workflow.add_node("web_search_node", web_search)  # 网络搜索
# 构建图的边（连接）
workflow.add_edge(START, "retrieve")  # 从开始到检索文档
workflow.add_edge("retrieve", "grade_documents")  # 从检索文档到给文档评分
workflow.add_conditional_edges(
       "grade_documents",
       decide_to_generate,
       {
              "transform_query": "transform_query",
              "generate": "generate",
       },
)
workflow.add_edge("transform_query", "web_search_node")  # 从转换查询到网络搜索
workflow.add_edge("web_search_node", "generate")  # 从网络搜索到生成答案
workflow.add_edge("generate", END)  # 从生成答案到结束
# 编译整个工作流
app = workflow.compile()

from pprint import pprint
# 设置输入问题
inputs = {"question": "What are the types of agent memory?"}
# 运行程序并处理输出
for output in app.stream(inputs):
       for key, value in output.items():
              # 打印当前节点名称
              pprint(f"节点 '{key}':")
              # 可选：在每个节点输出完整状态
              # pprint(value["keys"], indent=2, width=80, depth=None)
       pprint("\n---\n")
# 输出最终生成的答案
pprint(value["generation"])