# scripts/import_knowledge_doc.py

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 允许从项目根目录直接执行 python scripts/import_knowledge_doc.py。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

from src.agents.knowledge.doc_ingestion import ingest_file
from src.core.config import get_settings


DATA_DIR = Path(__file__).resolve().parent / "data"


def resolve_document_path(file_name: str) -> Path:
    """返回位于 scripts/data 目录下的待导入文档路径。"""
    return DATA_DIR / file_name


async def import_document(
    file_name: str,
    doc_type: str,
    category: str,
) -> int:
    """将 scripts/data 中的指定文档解析、向量化并写入知识库。"""
    file_path = resolve_document_path(file_name)
    if not file_path.is_file():
        raise FileNotFoundError(f"未找到待导入文档：{file_path}")

    settings = get_settings()
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )

    return await ingest_file(
        file_path=str(file_path),
        doc_name=file_path.name,
        doc_type=doc_type,
        category=category,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
    )


def parse_args() -> argparse.Namespace:
    """解析文档导入命令行参数。"""
    parser = argparse.ArgumentParser(description="导入 scripts/data 中的知识文档")
    parser.add_argument("--file", required=True, help="scripts/data 下的文件名")
    parser.add_argument(
        "--doc-type",
        default="guideline",
        choices=["guideline", "drug_instruction", "sop", "literature"],
        help="文档类型",
    )
    parser.add_argument("--category", default="未分类", help="文档分类")
    return parser.parse_args()


def main() -> None:
    """执行知识文档离线导入命令。"""
    args = parse_args()
    chunk_count = asyncio.run(
        import_document(
            file_name=args.file,
            doc_type=args.doc_type,
            category=args.category,
        )
    )
    print(f"导入完成：{args.file}，共写入 {chunk_count} 个文档分块。")


if __name__ == "__main__":
    main()
