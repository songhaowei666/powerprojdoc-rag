from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
# 数据模型
class GradeDocuments(BaseModel):
       """对检索文档相关性的二元评分"""
       binary_score: str = Field(
              description="文档与问题相关为'yes'，不相关为'no'"
       )
# 需要支持工具调用的语言模型
llm = ChatDeepSeek(model="deepseek-chat")
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# 提示模板
system = """你是一个评估检索到的文档与用户问题相关性的评分员。\n
       如果文档包含与问题相关的关键词或语义，则将其评为相关。\n
       给出一个二元评分'yes'或'no'来表示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
       [
              ("system", system)@@@,
              ("human", "检索到的文档：\n\n {document} \n\n 用户问题：{question}")@@@,
       ]
)
retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))