"""
OpenAI Embedding 向量库构建脚本。

读取分块后的 JSON 报告，使用 OpenAI Embedding API 为每个 chunk 生成向量，
并保存为 FAISS 索引文件。

用法:
    python -m src.openai_embedding
"""

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.config import settings


class OpenAIEmbedder:
    """封装 OpenAI Embedding API 调用。"""

    def __init__(self, model: str = None):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
        )
        self.model = model or settings.embedding_model or "text-embedding-3-large"

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

        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in resp.data]


class OpenAIVectorDBIngestor:
    """基于 OpenAI Embedding 的 FAISS 向量库构建器。"""

    def __init__(self):
        self.embedder = OpenAIEmbedder()

    @staticmethod
    def _create_vector_db(embeddings: List[List[float]]):
        """用 FAISS 构建向量索引（归一化后使用内积近似余弦相似度）。"""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        """处理单份报告：提取 chunks → 生成 embedding → 构建 FAISS 索引。"""
        text_chunks = [
            chunk["text"]
            for chunk in report["content"]["chunks"]
            if chunk.get("text")
        ]
        # 截断超长文本（OpenAI text-embedding-3 系列支持 8k tokens，
        # 这里保留 2048 字符作为安全冗余）
        text_chunks = [t[:2048] for t in text_chunks]

        embeddings = self.embedder.get_embeddings(text_chunks)
        if not embeddings:
            raise ValueError("没有有效的文本块可供嵌入")

        return self._create_vector_db(embeddings)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """批量处理目录下所有 JSON 报告，生成并保存 FAISS 向量库。

        Args:
            all_reports_dir: 存放分块后 JSON 报告的目录
            output_dir: 保存 .faiss 索引文件的目录
        """
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(
            all_report_paths, desc="OpenAI embedding & FAISS indexing"
        ):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            index = self._process_report(report_data)

            sha1 = report_data["metainfo"].get("sha1", "")
            if not sha1:
                raise ValueError(f"分块报告 {report_path} 缺少 sha1 字段，无法保存 faiss 文件！")

            faiss.write_index(index, str(output_dir / f"{sha1}.faiss"))

        print(f"Processed {len(all_report_paths)} reports → {output_dir}")


if __name__ == "__main__":
    """本地调试入口：为 chunked_reports 下的 JSON 报告构建 OpenAI 向量索引。"""
    root = Path(__file__).resolve().parent.parent

    # 默认路径与 PipelineConfig 保持一致
    input_dir = root / "data" / "stock_data" / "databases" / "chunked_reports"
    output_dir = root / "data" / "stock_data" / "databases" / "vector_dbs_openai"

    ingestor = OpenAIVectorDBIngestor()
    ingestor.process_reports(input_dir, output_dir)
