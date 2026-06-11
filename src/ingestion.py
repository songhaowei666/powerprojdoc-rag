import os
import json
import pickle
import sys
from typing import List, Union
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import hashlib
import jieba
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.openai_embedding import get_openai_embedding

# 将项目根目录加入 sys.path，支持直接运行本文件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
# from src.openai_embedding import default_embedder
from src.config import settings
from tenacity import retry, wait_fixed, stop_after_attempt

# BM25Ingestor：BM25索引构建与保存工具
class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """从文本块列表创建BM25索引，使用 jieba 进行中文分词。"""
        tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    @staticmethod
    def _build_chunk_metadata(chunk: dict, metainfo: dict) -> dict:
        """为单个 chunk 构建元数据字典，保留页码、文档来源等信息。"""
        return {
            "chunk_id": chunk.get("id", 0),
            "chunk_type": chunk.get("type", ""),
            "page": chunk.get("page", 0),
            "length_tokens": chunk.get("length_tokens", 0),
            "sha1": metainfo.get("sha1", ""),
            "sha1_name": metainfo.get("sha1_name", ""),
            "company_name": metainfo.get("company_name", ""),
            "file_name": metainfo.get("file_name", ""),
            "pages_amount": metainfo.get("pages_amount", 0),
        }

    def process_reports(
        self,
        all_reports_dir: Path | None = None,
        output_dir: Path | None = None,
        index_name: str = "default",
    ):
        """
        批量处理所有报告，生成并保存BM25索引。
        参数：
            all_reports_dir: 存放JSON报告的目录；为空时取 .env 中的 REPORTS_INPUT_DIR
            output_dir: 保存BM25索引的目录；为空时取 .env 中的 BM25_OUTPUT_DIR
            index_name: 索引标识，默认为 "default"
        """
        if all_reports_dir is None:
            all_reports_dir = Path(settings.reports_input_dir)
        if output_dir is None:
            output_dir = Path(settings.bm25_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        all_chunks: List[str] = []
        all_metadatas: List[dict] = []
        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            metainfo = report_data.get("metainfo", {})
            for chunk in report_data['content']['chunks']:
                all_chunks.append(chunk['text'])
                all_metadatas.append(self._build_chunk_metadata(chunk, metainfo))

        if all_chunks:
            bm25_index = self.create_bm25_index(all_chunks)
            output_file = output_dir / f"{index_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump({"index": bm25_index, "metadatas": all_metadatas, "texts": all_chunks}, f)

        print(f"Processed {len(all_report_paths)} reports")

    @staticmethod
    def load_bm25_index(index_path: Path) -> tuple[BM25Okapi, List[dict], List[str]]:
        """加载BM25索引及其对应的元数据、文本列表。

        兼容旧格式（纯 BM25Okapi 对象）与新版格式（dict 含 index + metadatas + texts）。

        返回：
            (bm25_index, metadatas, texts)
        """
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data["index"], data["metadatas"], data.get("texts", [])
        # 兼容旧格式：纯 BM25Okapi 对象
        return data, [], []

    def search(
        self,
        query: str,
        index_name: str = "default",
        output_dir: Path | None = None,
    ) -> tuple[np.ndarray, List[dict], List[str]]:
        """根据 query 查询指定 BM25 索引，返回所有 chunk 的 relevance scores、元数据及文本。

        参数：
            query: 查询文本
            index_name: 索引标识，用于定位 `{index_name}.pkl` 文件
            output_dir: 索引文件所在目录；为空时取 `settings.bm25_output_dir`

        返回：
            (scores, metadatas, texts) — 三者长度一致

        异常：
            FileNotFoundError: 索引文件不存在时抛出
        """
        if output_dir is None:
            output_dir = Path(settings.bm25_output_dir)
        index_path = output_dir / f"{index_name}.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        bm25_index, metadatas, texts = self.load_bm25_index(index_path)
        tokenized_query = list(jieba.cut(query))
        scores = bm25_index.get_scores(tokenized_query)
        return scores, metadatas, texts

# VectorDBIngestor：向量库构建与保存工具
class VectorDBIngestor:
    def __init__(self, embedder:OpenAIEmbeddings):
        self.embedder = embedder

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]]) -> List[List[float]]:
        """使用构造方法传入的 embedder 获取文本块的嵌入向量，支持重试。"""
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
        return self.embedder.embed_documents(text_chunks)

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
        all_reports_dir: Path | None = None,
        output_dir: Path | None = None,
        index_name: str | None = None,
    ):
        """批量处理所有报告，生成并保存 ChromaDB 向量库。

        Args:
            all_reports_dir: 存放 JSON 报告的目录；为空时取 .env 中的 REPORTS_INPUT_DIR。
            output_dir: ChromaDB 持久化目录；为空时取 .env 中的 CHROMA_PERSIST_DIR。
            index_name: 索引标识，写入 metadata；为空时默认为 "default"。
        """
        if all_reports_dir is None:
            all_reports_dir = Path(settings.reports_input_dir)
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
                    embedding=self.embedder,
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
    vdb_ingestor = VectorDBIngestor(get_openai_embedding())
    vdb_ingestor.process_reports()
    print(f"Vector databases created in {settings.chroma_persist_dir}")