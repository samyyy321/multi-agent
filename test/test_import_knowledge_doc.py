import subprocess
import sys
from pathlib import Path

import pytest

import scripts.import_knowledge_doc as importer


def test_resolve_document_path_uses_scripts_data_directory():
    assert importer.resolve_document_path("guide.pdf") == importer.DATA_DIR / "guide.pdf"


@pytest.mark.asyncio
async def test_import_document_passes_scripts_data_file_to_ingestion(monkeypatch):
    document = importer.DATA_DIR / "_test_import_knowledge_doc.md"
    importer.DATA_DIR.mkdir(exist_ok=True)
    document.write_text("test document", encoding="utf-8")
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured["embedding_kwargs"] = kwargs

    class FakeMilvusClient:
        def __init__(self, **kwargs):
            captured["milvus_kwargs"] = kwargs

    async def fake_ingest_file(**kwargs):
        captured["ingest_kwargs"] = kwargs
        return 3

    monkeypatch.setattr(importer, "DashScopeEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(importer, "MilvusClient", FakeMilvusClient)
    monkeypatch.setattr(importer, "ingest_file", fake_ingest_file)
    monkeypatch.setattr(
        importer,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "EMBEDDING_MODEL": "embedding-model",
                "DASHSCOPE_API_KEY": "test-key",
                "MILVUS_HOST": "localhost",
                "MILVUS_PORT": 19530,
            },
        )(),
    )

    try:
        count = await importer.import_document(
            file_name=document.name,
            doc_type="guideline",
            category="cardiology",
        )

        assert count == 3
        assert captured["ingest_kwargs"]["file_path"] == str(document)
        assert captured["ingest_kwargs"]["doc_name"] == document.name
        assert captured["ingest_kwargs"]["doc_type"] == "guideline"
        assert captured["ingest_kwargs"]["category"] == "cardiology"
    finally:
        document.unlink(missing_ok=True)


def test_cli_can_run_from_project_root():
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/import_knowledge_doc.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--file" in result.stdout
    assert "\u5bfc\u5165 scripts/data \u4e2d\u7684\u77e5\u8bc6\u6587\u6863" in result.stdout
