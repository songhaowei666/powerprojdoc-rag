# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
import sys
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import shutil

# 将项目根目录加入 sys.path，支持直接运行本文件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from src.pdf_parsing import PDFParser
from src import pdf_mineru
from src.parsed_reports_merging import PageTextPreparation
from src.markdown_reports_merging import MinerUReportMerger
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from src.config import settings
from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    """流水线路径与目录配置，根据 root_path 派生所有流程路径。"""

    root_path: Path
    subset_name: str = "subset.csv"
    questions_file_name: str = "questions.json"
    pdf_reports_dir_name: str = "pdf_reports"
    use_serialized_tables: bool = False
    config_suffix: str = ""
    company_code: str = "001"

    def __post_init__(self) -> None:
        suffix = "_ser_tab" if self.use_serialized_tables else ""

        self.subset_path = self.root_path / self.subset_name
        self.questions_file_path = self.root_path / self.questions_file_name
        self.pdf_reports_dir = self.root_path / self.pdf_reports_dir_name

        self.answers_file_path = self.root_path / f"answers{self.config_suffix}.json"
        self.debug_data_path = self.root_path / "debug_data"
        self.databases_path = self.root_path / f"databases{suffix}"

        self.vector_db_dir = self.databases_path / settings.vector_db_subdir
        self.documents_dir = self.databases_path / settings.chunked_reports_subdir
        self.bm25_db_path = self.databases_path / settings.bm25_dbs_subdir

        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

    @classmethod
    def from_root(cls, root_path: Path, **kwargs: object) -> "PipelineConfig":
        """从数据集根目录快速构建配置。"""
        return cls(root_path=root_path, **kwargs)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config




    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.config.parsed_reports_path,
            csv_metadata_path=self.config.subset_path
        )
        pdf_parser.debug_data_path = self.config.parsed_reports_debug_path

        input_doc_paths = list(self.config.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.config.parsed_reports_path}")

    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py，将指定 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        # 调用 pdf_mineru 获取 task_id 并下载、解压
        print(f"开始处理: {file_name}")
        task_id = pdf_mineru.get_task_id(file_name)
        print(f"task_id: {task_id}")
        pdf_mineru.get_result(task_id)

        # 解压后目录名与 task_id 相同
        extract_dir = f"{task_id}"
        md_path = os.path.join(extract_dir, "full.md")
        if not os.path.exists(md_path):
            print(f"未找到 markdown 文件: {md_path}")
            return
        # 目标目录
        os.makedirs(self.config.reports_markdown_path, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.config.reports_markdown_path, f"{base_name}.md")
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")

    def merge_mineru_reports(
        self,
        reports_dir: Path = None,
        reports_paths: list[Path] = None,
    ) -> list[dict]:
        """
        将 MinerU 解析后的 JSON 报告批量规整为标准报告结构（metainfo + content.pages）。

        参数：
            reports_dir: 输入 JSON 文件目录，自动收集该目录下所有 *.json
            reports_paths: 输入 JSON 文件路径列表
        返回：
            规整后的报告对象列表，每个元素为 {"metainfo": ..., "content": ...}
        """
        merger = MinerUReportMerger()
        reports = merger.process_reports(
            reports_dir=reports_dir,
            reports_paths=reports_paths,
            output_dir=self.config.merged_reports_path,
            company_code=self.config.company_code,
        )
        print(f"Merged {len(reports)} reports into {self.config.merged_reports_path}")
        return reports
    
    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.config.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.config.merged_reports_path,
            output_dir=self.config.documents_dir,
            subset_csv=self.config.subset_path
        )
        print(f"分割完成，结果已保存到 {self.config.documents_dir}")
    def chunk_reports2(self, include_serialized_tables: bool = False):
        """将规整后报告分块，便于后续向量化和检索"""
        text_splitter = TextSplitter()
        
        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.config.parsed_reports_path
        
        text_splitter.split_all_reports(
            self.config.merged_reports_path,
            self.config.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.config.documents_dir}")
    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.config.documents_dir
        output_dir = self.config.vector_db_dir
        
        from src.openai_embedding import get_openai_embedding
        vdb_ingestor = VectorDBIngestor(embedder=get_openai_embedding())
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.config.documents_dir
        output_file = self.config.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 解析PDF报告，支持并行处理
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        

    def vector_retrieve(
        self,
        query: str,
        company_code: str = "",
        top_n: int = 3,
        return_parent_pages: bool = False,
        index_name: str = "default",
    ) -> list[dict]:
        """
        向量检索：在 ChromaDB 中按语义相似度检索文本块。

        参数：
            query: 查询文本
            company_code: 公司编码，用于 metadata 过滤；为空时使用 PipelineConfig.company_code
            top_n: 返回结果数量上限
            return_parent_pages: 为 True 时返回整页内容
            index_name: ChromaDB collection 名称
        """
        retriever = VectorRetriever(
            vector_db_dir=self.config.vector_db_dir,
            documents_dir=self.config.documents_dir,
            index_name=index_name,
        )
        code = company_code or self.config.company_code
        return retriever.retrieve(
            company_code=code,
            query=query,
            top_n=top_n,
            return_parent_pages=return_parent_pages,
        )

    def bm25_retrieve(
        self,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False,
        index_name: str = "default",
    ) -> list[dict]:
        """
        关键词检索：在 BM25 索引中检索与 query 最相关的文本块。

        参数：
            query: 查询文本
            top_n: 返回结果数量上限
            return_parent_pages: 为 True 时返回整页内容
            index_name: BM25 索引名称，对应 `{index_name}.pkl`
        """
        retriever = BM25Retriever(
            bm25_db_dir=self.config.bm25_db_path,
            documents_dir=self.config.documents_dir,
            index_name=index_name,
        )
        return retriever.retrieve(
            query=query,
            top_n=top_n,
            return_parent_pages=return_parent_pages,
        )

    def hybrid_retrieve(
        self,
        query: str,
        company_code: str = "",
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
    ) -> list[dict]:
        """
        混合检索：向量召回 + LLM 重排。

        参数：
            query: 查询文本
            company_code: 公司编码，用于向量召回阶段的 metadata 过滤；为空时使用 PipelineConfig.company_code
            llm_reranking_sample_size: 首轮向量召回候选数
            documents_batch_size: 每批送入 LLM 重排的文档数
            top_n: 最终返回结果数
            llm_weight: LLM 分数权重（0-1）
            return_parent_pages: 为 True 时返回整页内容
        """
        retriever = HybridRetriever(
            vector_db_dir=self.config.vector_db_dir,
            documents_dir=self.config.documents_dir,
        )
        code = company_code or self.config.company_code
        return retriever.retrieve(
            company_code=code,
            query=query,
            llm_reranking_sample_size=llm_reranking_sample_size,
            documents_batch_size=documents_batch_size,
            top_n=top_n,
            llm_weight=llm_weight,
            return_parent_pages=return_parent_pages,
        )


if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程
    pipeline = Pipeline(PipelineConfig.from_root(root_path))
    
    # print('将pdf转化为纯markdown文本')
    # pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf') 

    # print('将MinerU解析后的JSON报告规整为标准报告结构')
    # pipeline.merge_mineru_reports(reports_dir=root_path / "debug_data")

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    # print('将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    # pipeline.chunk_reports2() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    # print('从分块报告创建向量数据库，输出到 databases/vector_dbs')
    # pipeline.create_vector_dbs()     

    # print('向量检索')
    # print(pipeline.vector_retrieve(query="工程总投资",return_parent_pages=True))

    # print("bm25关键词构建-------")
    # pipeline.create_bm25_db()


    print(pipeline.bm25_retrieve(query="工程总投资", top_n=5, return_parent_pages=True))


    # print('向量检索')
    # pipeline.vector_retrieve(query="工程总投资")

    # print('关键词检索')
    # pipeline.bm25_retrieve(query="工程总投资")

    # print('混合检索')
    # pipeline.hybrid_retrieve(query="工程总投资")
    
    print('完成')
