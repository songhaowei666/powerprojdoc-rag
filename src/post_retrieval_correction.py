from pydantic import BaseModel, Field
from src.api_requests import APIProcessor


class GradeDocuments(BaseModel):
    """对检索文档相关性的二元评分"""
    binary_score: str = Field(
        description="文档与问题相关为'yes'，不相关为'no'"
    )


class RetrievalGrader:
    """检索后文档相关性评分器（后处理验证）"""

    def __init__(self, provider: str = "openai"):
        self.processor = APIProcessor(provider=provider)
        self.system_content = (
            "你是一个评估检索到的文档与用户问题相关性的评分员。\n"
            "如果文档包含与问题相关的关键词或语义，则将其评为相关。\n"
            "给出一个二元评分'yes'或'no'来表示文档是否与问题相关。"
        )

    def grade(self, question: str, document: str) -> GradeDocuments:
        """对单篇文档与用户问题的相关性进行评分。"""
        human_content = f"检索到的文档：\n\n{document}\n\n用户问题：{question}"
        result = self.processor.send_message(
            system_content=self.system_content,
            human_content=human_content,
            is_structured=True,
            response_format=GradeDocuments
        )
        return GradeDocuments(**result)

    def invoke(self, inputs: dict) -> GradeDocuments:
        """兼容 LangChain / LangGraph 的 Runnable 接口。

        参数：
            inputs: 字典，必须包含 "question" 和 "document" 键
        返回：
            GradeDocuments 实例
        """
        question = inputs["question"]
        document = inputs["document"]
        return self.grade(question, document)


# 默认实例，供其他模块直接导入使用
retrieval_grader = RetrievalGrader()


# 示例用法
if __name__ == "__main__":
    question = "agent memory"
    doc_txt = "这是一篇讲 agent memory 的文档"
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
