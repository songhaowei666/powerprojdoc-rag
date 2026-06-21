"""
项目全局配置，基于 pydantic-settings 从 .env 文件加载。

用法:
    from src.config import settings
    print(settings.openai_api_key)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用配置类，自动读取项目根目录下的 .env 文件。
    字段名采用小写 + 下划线，映射到环境变量时大小写不敏感。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_api_base: str = ""

    # --- 模型选择 ---
    chat_model: str = ""
    embedding_model: str = ""

    # --- Gemini ---
    gemini_api_key: str = ""

    # --- Jina Reranker ---
    jina_api_key: str = ""

    # --- MinerU ---
    mineru_api_key: str = ""

    # --- DashScope (代码中使用，.env 中可补充) ---
    dashscope_api_key: str = ""

    # --- IBM (代码中使用，.env 中可补充) ---
    ibm_api_key: str = ""

    # --- ChromaDB 向量库持久化目录 ---
    chroma_persist_dir: str = "data/projdoc_data/databases/vector_dbs"

    # --- BM25 索引输出目录 ---
    bm25_output_dir: str = "data/projdoc_data/databases/bm25_index"

    # --- 报告 JSON 输入目录 ---
    reports_input_dir: str = "data/projdoc_data/databases/chunked_reports"

    # --- Pipeline 子目录名 ---
    vector_db_subdir: str = "vector_dbs"
    chunked_reports_subdir: str = "chunked_reports"
    bm25_dbs_subdir: str = "bm25_dbs"


# 全局单例，导入即用
settings = Settings()
