"""
OpenAI Embedding 向量库构建脚本。

读取分块后的 JSON 报告，使用 OpenAI Embedding API 为每个 chunk 生成向量，
并保存为 FAISS 索引文件。

用法:
    python -m src.openai_embedding
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，解决直接运行时的模块导入问题
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from src.config import settings


class OpenAIEmbedder:
    """封装 OpenAI Embedding API 调用（基于 LangChain OpenAIEmbeddings）。"""

    def __init__(self, model: str = None):
        self.model = model or settings.embedding_model or "text-embedding-3-large"
        self.client = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base or None,
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的 embedding 向量。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            与输入一一对应的 embedding 列表
        """
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return []

        return self.client.embed_documents(texts)


# 类外部默认实例化，供其他模块直接复用

def get_openai_embedding():
    return OpenAIEmbeddings(
                model= settings.embedding_model,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_api_base or None,
            )



class OpenAIVectorDBIngestor:
    """基于 OpenAI Embedding 的 ChromaDB 向量库构建器。"""

    def __init__(self):
        self.embedder = default_embedder

    def _build_docs(self, report: dict) -> List[Document]:
        """处理单份报告：提取 chunks → 构建 Document 列表。"""
        chunks = report.get("content", {}).get("chunks", [])
        metainfo = report.get("metainfo", {})

        docs = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            # 截断超长文本（OpenAI text-embedding-3 系列支持 8k tokens，
            # 这里保留 2048 字符作为安全冗余）
            text = text[:2048]

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": chunk.get("id", 0),
                        "chunk_type": chunk.get("type", ""),
                        "page": chunk.get("page", 0),
                        "length_tokens": chunk.get("length_tokens", 0),
                        "sha1": metainfo.get("sha1", ""),
                        "sha1_name": metainfo.get("sha1_name", ""),
                        "company_code": metainfo.get("company_code", ""),
                        "file_name": metainfo.get("file_name", ""),
                        "pages_amount": metainfo.get("pages_amount", 0),
                    }
                )
            )

        return docs

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """批量处理目录下所有 JSON 报告，生成并保存 ChromaDB 向量库。

        Args:
            all_reports_dir: 存放分块后 JSON 报告的目录
            output_dir: 保存 ChromaDB 向量库的目录
        """
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = None

        for report_path in tqdm(
            all_report_paths, desc="OpenAI embedding & ChromaDB indexing"
        ):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            docs = self._build_docs(report_data)
            if not docs:
                continue

            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embedder.client,
                    persist_directory=str(output_dir),
                )
            else:
                vectorstore.add_documents(docs)

        print(f"Processed {len(all_report_paths)} reports → {output_dir}")


if __name__ == "__main__":
    """本地调试入口：为 chunked_reports 下的 JSON 报告构建 OpenAI 向量索引。"""
    # root = Path(__file__).resolve().parent.parent

    # # 默认路径与 PipelineConfig 保持一致
    # input_dir = root / "data" / "stock_data" / "databases" / "chunked_reports"
    # output_dir = root / "data" / "stock_data" / "databases" / "vector_dbs_openai"

    # ingestor = OpenAIVectorDBIngestor()
    # ingestor.process_reports(input_dir, output_dir)
    embeding = OpenAIEmbedder()
    em_arr = embeding.get_embeddings(["aaa"])
    print(em_arr)