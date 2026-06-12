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
import time

# 将项目根目录加入 sys.path，支持直接运行本文件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from src.pdf_parsing import PDFParser
from src import pdf_mineru
from src.parsed_reports_merging import PageTextPreparation
from src.markdown_reports_merging import MinerUReportMerger
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.config import settings
from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / settings.vector_db_subdir
        self.documents_dir = self.databases_path / settings.chunked_reports_subdir
        self.bm25_db_path = self.databases_path / settings.bm25_dbs_subdir

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

class Pipeline:
    def __init__(
        self,
        root_path: Path,
        subset_name: str = "subset.csv",
        questions_file_name: str = "questions.json",
        pdf_reports_dir_name: str = "pdf_reports",
        use_serialized_tables: bool = False,
        config_suffix: str = "",
    ):
        # 初始化主流程，加载路径和配置
        self.use_serialized_tables = use_serialized_tables
        self.config_suffix = config_suffix
        self.paths = PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=use_serialized_tables,
            config_suffix=config_suffix
        )
        self._convert_json_to_csv_if_needed()

    def _convert_json_to_csv_if_needed(self):



    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

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
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
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
            output_dir=self.paths.merged_reports_path,
            subset_csv=self.paths.subset_path,
        )
        print(f"Merged {len(reports)} reports into {self.paths.merged_reports_path}")
        return reports
    
    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.merged_reports_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")
    def chunk_reports2(self, include_serialized_tables: bool = False):
        """将规整后报告分块，便于后续向量化和检索"""
        text_splitter = TextSplitter()
        
        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path
        
        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")
    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        from openai_embedding import get_openai_embedding
        vdb_ingestor = VectorDBIngestor(embedder=get_openai_embedding())
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
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
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(
        self,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 30,
        top_n_retrieval: int = 10,
        parallel_requests: int = 1,
        pipeline_details: str = "",
        submission_file: bool = True,
        full_context: bool = False,
        api_provider: str = "openai",
        answering_model: str = "gpt-4-turbo",
    ):
        """
        处理所有问题，生成答案文件。
        问题处理相关参数均在调用时显式传入。
        """
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=parent_document_retrieval,
            llm_reranking=llm_reranking,
            llm_reranking_sample_size=llm_reranking_sample_size,
            top_n_retrieval=top_n_retrieval,
            parallel_requests=parallel_requests,
            api_provider=api_provider,
            answering_model=answering_model,
            full_context=full_context
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=submission_file,
            pipeline_details=pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(
        self,
        question: str,
        kind: str = "string",
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 30,
        top_n_retrieval: int = 10,
        parallel_requests: int = 1,
        api_provider: str = "openai",
        answering_model: str = "gpt-4-turbo",
        full_context: bool = False,
    ):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=parent_document_retrieval,
            llm_reranking=llm_reranking,
            llm_reranking_sample_size=llm_reranking_sample_size,
            top_n_retrieval=top_n_retrieval,
            parallel_requests=parallel_requests,
            api_provider=api_provider,
            answering_model=answering_model,
            full_context=full_context
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        return answer


if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程
    pipeline = Pipeline(root_path)
    
    # print('4. 将pdf转化为纯markdown文本')
    # pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf') 

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    # print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    # pipeline.chunk_reports2() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    # print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    # pipeline.create_vector_dbs()     

    # print("bm25关键词构建-------")
    # pipeline.create_bm25_db()
    
    # 7. 处理问题并生成答案，具体逻辑由 process_questions 参数决定
    # 默认questions.json
    # print('7. 处理问题并生成答案，具体逻辑由 process_questions 参数决定')
    # pipeline.process_questions() 
    
    print('完成')
