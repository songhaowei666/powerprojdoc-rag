import json
import sys
from pathlib import Path

# 导入所需的库
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

# 将项目根目录加入 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.config import settings

doc_str = '''
{'source': 'zDvnAY0zH7U',
'title': '……山西大山中……的唐代木构建筑……一生必去的一个地方',
'description': '唐代是中国封建社会的鼎盛时期，唐代石刻建筑在全国有很多……在山西五台山保留了一座气势恢宏的唐代木构建筑，是中国现存规模最大，保存最完全的唐代木构建筑。',
'view_count': 81598,
'publish_date': '2024-04-08 00:00:00',
'length': 1394,
'author': '行迹旅途中'}'''

# 解析为 dict
data = json.loads(doc_str.replace("'", '"'))

# 转成 LangChain Document 对象
doc = Document(
    page_content=data["description"],
    metadata={
        "source": data["source"],
        "title": data["title"],
        "view_count": data["view_count"],
        "publish_date": data["publish_date"],
        "length": data["length"],
        "author": data["author"],
    },
)

print("LangChain Document:")
print(f"  page_content: {doc.page_content[:30]}...")
print(f"  metadata: {doc.metadata}")



# 创建向量存储



# 使用通过 URL 访问的 OpenAI Embedding API
embed_model = OpenAIEmbeddings(
    model=settings.embedding_model or "text-embedding-3-large",
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_api_base or None,
)
vectorstore = Chroma.from_documents([doc], embed_model)
# 配置检索器的元数据字段
metadata_field_info = [
       AttributeInfo(
              name="title",
              description="视频标题（@@@字符串）@@@",
              type="string",
       ),
       AttributeInfo(
              name="author",
              description="视频作者（@@@字符串）@@@",
              type="string",
       ),
       AttributeInfo(
              name="view_count",
              description="视频观看次数（@@@整数）@@@",
              type="integer",
       ),
       AttributeInfo(
              name="publish_date",
              description="视频发布日期，格式为YYYY-MM-DD的字符串",
              type="string",
       ),
       AttributeInfo(
              name="length",
              description="视频长度，以秒为单位的整数",
              type="integer",
       ),
]
# 创建自查询检索器SelfQueryRetriever
# 使用 OpenAI 语言模型
llm = ChatOpenAI(
    model=settings.chat_model or "gpt-4o",
    temperature=0,
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_api_base or None,
)
retriever = SelfQueryRetriever.from_llm(
       llm=llm,
       vectorstore=vectorstore,
       document_contents="包含视频标题、作者、观看次数、发布日期等信息的视频元数据",
       metadata_field_info=metadata_field_info,
       enable_limit=True,
       verbose=True
)
# 执行示例查询
queries = [
       "找出两个观看次数超过100000的视频",
       "显示最新发布的视频"
]
# 执行查询并输出结果
for query in queries:
       print(f"\n查询：{query}")
       try:
              results = retriever.invoke(query)
              if not results:
                     print("未找到匹配的视频")
                     continue
              for doc in results:
                     print(f"标题：{doc.metadata['title']}")
                     print(f"观看次数：{doc.metadata['view_count']}")
                     print(f"发布日期：{doc.metadata['publish_date']}")
       except Exception as e:
              print(f"查询出错：{str(e)}")
              continue