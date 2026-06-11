import os
import json
import pickle
import sys
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import hashlib

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# 将项目根目录加入 sys.path，支持直接运行本文件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.openai_embedding import OpenAIEmbedder, default_embedder
from src.config import settings
from tenacity import retry, wait_fixed, stop_after_attempt

# BM25Ingestor：BM25索引构建与保存工具
class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """从文本块列表创建BM25索引"""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """
        批量处理所有报告，生成并保存BM25索引。
        参数：
            all_reports_dir (Path): 存放JSON报告的目录
            output_dir (Path): 保存BM25索引的目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # 提取文本块并创建BM25索引
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # 保存BM25索引，文件名用sha1_name
            sha1_name = report_data["metainfo"]["sha1"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

# VectorDBIngestor：向量库构建与保存工具
class VectorDBIngestor:
    def __init__(self):
        # 复用 openai_embedding 中的默认 embedder 实例
        self.embedder = default_embedder

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[List[float]]:
        """使用 OpenAI Embedding API 获取文本块的嵌入向量，支持重试。"""
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        # 统一为字符串列表
        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        # 类型与空值检查
        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError("所有待嵌入文本必须为字符串类型！实际类型: {}".format([type(x) for x in text_chunks]))

        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")

        print("start embedding ================================")
        # 复用 OpenAIEmbedder 实例；若指定了不同 model，则临时创建新实例
        embedder = self.embedder
        if model != embedder.model:
            embedder = OpenAIEmbedder(model=model)
        return embedder.get_embeddings(text_chunks)

    def _build_docs(self, report: dict, index_name: str = "default") -> List[Document]:
        """针对单份报告，提取文本块并构建 Document 列表。"""
        chunks = report.get("content", {}).get("chunks", [])
        metainfo = report.get("metainfo", {})

        docs = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            # 截断超长文本
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
                        "company_name": metainfo.get("company_name", ""),
                        "file_name": metainfo.get("file_name", ""),
                        "pages_amount": metainfo.get("pages_amount", 0),
                        "index_name": index_name,
                    }
                )
            )

        return docs

    def process_reports(
        self,
        all_reports_dir: Path,
        output_dir: Path | None = None,
        index_name: str | None = None,
    ):
        """批量处理所有报告，生成并保存 ChromaDB 向量库。

        Args:
            all_reports_dir: 存放 JSON 报告的目录。
            output_dir: ChromaDB 持久化目录；为空时取 .env 中的 CHROMA_PERSIST_DIR。
            index_name: 索引标识，写入 metadata；为空时默认为 "default"。
        """
        if output_dir is None:
            output_dir = Path(settings.chroma_persist_dir)
        if index_name is None:
            index_name = "default"

        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = None

        for report_path in tqdm(all_report_paths, desc="Processing reports for ChromaDB"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            docs = self._build_docs(report_data, index_name=index_name)
            if not docs:
                continue

            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embedder.client,
                    persist_directory=str(output_dir),
                    collection_name=index_name,
                )
            else:
                vectorstore.add_documents(docs)

        print(f"Processed {len(all_report_paths)} reports")


if __name__ == "__main__":
    """
    本地调试入口：读取分块后的 JSON 报告，为每个 chunk 生成 embedding 并保存为 ChromaDB 向量库。
    """
    root = Path(__file__).resolve().parent.parent

    # 默认路径与 PipelineConfig 保持一致
    input_dir = root / "data" / "stock_data" / "databases" / "chunked_reports"

    vdb_ingestor = VectorDBIngestor()
    vdb_ingestor.process_reports(input_dir)
    print(f"Vector databases created in {settings.chroma_persist_dir}")