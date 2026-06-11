import json
import logging
import sys
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import time

# 将项目根目录加入 sys.path，支持直接运行本文件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import BM25Ingestor
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path, index_name: str = "default"):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        self.index_name = index_name
        self._pages_by_sha1 = self._load_pages_mapping()

    def _load_pages_mapping(self) -> dict[str, dict[int, str]]:
        """遍历 documents_dir 下所有 JSON，建立 sha1 -> {page_num: page_text} 映射。"""
        mapping: dict[str, dict[int, str]] = {}
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception:
                continue
            sha1 = doc.get("metainfo", {}).get("sha1", "")
            if not sha1:
                continue
            pages = doc.get("content", {}).get("pages", [])
            mapping[sha1] = {p["page"]: p["text"] for p in pages if "page" in p}
        return mapping

    def _get_page_text(self, sha1: str, page: int, fallback: str = "") -> str:
        """根据 sha1 和 page 号获取整页内容；找不到时回退到 fallback。"""
        return self._pages_by_sha1.get(sha1, {}).get(page, fallback)

    def retrieve(
        self,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        """在全局BM25索引中检索与query最相关的文本块。

        参数：
            query: 查询文本
            top_n: 返回结果数量上限
            return_parent_pages: 为True时按(文档,页码)去重，返回整页内容

        返回：
            包含distance、page、text的字典列表
        """
        ingestor = BM25Ingestor()
        scores, metadatas, texts = ingestor.search(
            query=query,
            index_name=self.index_name,
            output_dir=self.bm25_db_dir,
        )

        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        retrieval_results = []
        seen_keys = set()

        for idx in top_indices:
            score = round(float(scores[idx]), 4)
            meta = metadatas[idx]
            chunk_text = texts[idx] if idx < len(texts) else ""
            page = meta.get("page", 0)
            sha1 = meta.get("sha1", "")

            if return_parent_pages:
                key = (sha1, page)
                if key not in seen_keys:
                    seen_keys.add(key)
                    page_text = self._get_page_text(sha1, page, fallback=chunk_text)
                    retrieval_results.append({
                        "distance": score,
                        "page": page,
                        "text": page_text,
                    })
            else:
                retrieval_results.append({
                    "distance": score,
                    "page": page,
                    "text": chunk_text,
                })

        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, index_name: str = "default"):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.index_name = index_name
        self._vectorstore = self._load_vectorstore()
        self._pages_by_sha1 = self._load_pages_mapping()

    def _load_vectorstore(self):
        from langchain_chroma import Chroma
        from src.openai_embedding import get_openai_embedding
        return Chroma(
            persist_directory=str(self.vector_db_dir),
            embedding_function=get_openai_embedding(),
            collection_name=self.index_name,
        )

    def _load_pages_mapping(self) -> dict[str, dict[int, str]]:
        mapping: dict[str, dict[int, str]] = {}
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception:
                continue
            sha1 = doc.get("metainfo", {}).get("sha1", "")
            if not sha1:
                continue
            pages = doc.get("content", {}).get("pages", [])
            mapping[sha1] = {p["page"]: p["text"] for p in pages if "page" in p}
        return mapping

    def _get_page_text(self, sha1: str, page: int, fallback: str = "") -> str:
        return self._pages_by_sha1.get(sha1, {}).get(page, fallback)

    def _find_report_by_company(self, company_name: str) -> dict:
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception:
                continue
            metainfo = doc.get("metainfo", {})
            if metainfo.get("company_name") == company_name:
                return doc
            elif company_name in metainfo.get("file_name", ""):
                return doc
        raise ValueError(f"No report found with '{company_name}' company name.")

    def retrieve(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = None,  # 占位，兼容 HybridRetriever 调用
        top_n: int = 3,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        """在全局 ChromaDB 向量库中按公司名过滤检索与 query 最相关的文本块。

        参数：
            company_name: 目标公司名称，用于 ChromaDB metadata 过滤
            query: 查询文本
            llm_reranking_sample_size: 占位参数，当前未使用
            top_n: 返回结果数量上限
            return_parent_pages: 为True时按(文档,页码)去重，返回整页内容

        返回：
            包含distance、page、text的字典列表
        """
        search_kwargs = {"k": top_n}
        if company_name:
            search_kwargs["filter"] = {"company_name": company_name}

        docs_with_scores = self._vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )

        retrieval_results = []
        seen_keys = set()

        for doc, score in docs_with_scores:
            score = round(float(score), 4)
            meta = doc.metadata
            chunk_text = doc.page_content
            page = meta.get("page", 0)
            sha1 = meta.get("sha1", "")

            if return_parent_pages:
                key = (sha1, page)
                if key not in seen_keys:
                    seen_keys.add(key)
                    page_text = self._get_page_text(sha1, page, fallback=chunk_text)
                    retrieval_results.append({
                        "distance": score,
                        "page": page,
                        "text": page_text,
                    })
            else:
                retrieval_results.append({
                    "distance": score,
                    "page": page,
                    "text": chunk_text,
                })

        return retrieval_results

    @staticmethod
    def set_up_llm():
        # 静态方法，初始化OpenAI LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()
        
    def retrieve(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用混合检索方法进行检索和重排。
        
        参数：
            company_name: 需要检索的公司名称
            query: 检索查询语句
            llm_reranking_sample_size: 首轮向量检索返回的候选数量
            documents_batch_size: 每次送入LLM重排的文档数
            top_n: 最终返回的重排结果数量
            llm_weight: LLM分数权重（0-1）
            return_parent_pages: 是否返回完整页面（而非分块）
        
        返回：
            经过重排的文档字典列表，包含分数
        """
        t0 = time.time()
        # 首先用向量检索器获取初步结果
        print("[计时] [HybridRetriever] 开始向量检索 ...")
        vector_results = self.vector_retriever.retrieve(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        # 使用LLM对结果进行重排
        print("[计时] [HybridRetriever] 开始LLM重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        t2 = time.time()
        print(f"[计时] [HybridRetriever] LLM重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        return reranked_results[:top_n]


if __name__ == "__main__":
    """
    本地调试入口：对现有索引进行检索并打印结果。
    用法：python src/retrieval.py
    """
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    DB_ROOT = ROOT / "data" / "stock_data" / "databases"
    REPORTS_DIR = DB_ROOT / "chunked_reports"
    BM25_DIR = DB_ROOT / "bm25_dbs"
    VECTOR_DIR = DB_ROOT / "vector_dbs"

    TEST_COMPANY = "中芯国际集成电路制造有限公司"
    TEST_QUERY = "营业收入"

    print("=" * 60)
    print("BM25 检索测试")
    print("=" * 60)
    if (BM25_DIR / "default.pkl").exists():
        bm25_retriever = BM25Retriever(
            bm25_db_dir=BM25_DIR,
            documents_dir=REPORTS_DIR,
            index_name="default",
        )
        bm25_results = bm25_retriever.retrieve(
            query=TEST_QUERY,
            top_n=3,
            return_parent_pages=True,
        )
        for i, r in enumerate(bm25_results, 1):
            print(f"\n[{i}] distance={r['distance']}  page={r['page']}")
            print(r["text"][:300] + "...")
    else:
        print(f"BM25 索引不存在: {BM25_DIR / 'default.pkl'}")

    print("\n" + "=" * 60)
    print("向量检索测试")
    print("=" * 60)
    try:
        if (VECTOR_DIR / "chroma.sqlite3").exists():
            vector_retriever = VectorRetriever(
                vector_db_dir=VECTOR_DIR,
                documents_dir=REPORTS_DIR,
                index_name="default",
            )
            vector_results = vector_retriever.retrieve(
                company_name=TEST_COMPANY,
                query=TEST_QUERY,
                top_n=3,
                return_parent_pages=True,
            )
            for i, r in enumerate(vector_results, 1):
                print(f"\n[{i}] distance={r['distance']}  page={r['page']}")
                print(r["text"][:300] + "...")
        else:
            print(f"ChromaDB 不存在: {VECTOR_DIR}")
    except Exception as e:
        print(f"向量检索失败: {e}")

    print("\n" + "=" * 60)
    print("混合检索测试")
    print("=" * 60)
    try:
        if (VECTOR_DIR / "chroma.sqlite3").exists():
            hybrid_retriever = HybridRetriever(
                vector_db_dir=VECTOR_DIR,
                documents_dir=REPORTS_DIR,
            )
            hybrid_results = hybrid_retriever.retrieve(
                company_name=TEST_COMPANY,
                query=TEST_QUERY,
                top_n=3,
                return_parent_pages=True,
            )
            for i, r in enumerate(hybrid_results, 1):
                print(f"\n[{i}] distance={r.get('distance', 'N/A')}  page={r['page']}")
                print(r["text"][:300] + "...")
        else:
            print(f"ChromaDB 不存在: {VECTOR_DIR}")
    except Exception as e:
        print(f"混合检索失败: {e}")
    