import pytest

import src.agents.knowledge.doc_rag as doc_rag
import src.agents.knowledge.graph_rag as graph_rag
import src.agents.knowledge.mineru_client as mineru_client
import src.agents.knowledge.tools as knowledge_tools
import src.agents.knowledge.doc_ingestion as doc_ingestion
from src.agents.knowledge.doc_ingestion import _split_markdown
from src.agents.knowledge.nl2sql import _validate_sql, search_sql
from src.agents.knowledge.tools import KnowledgeDeps, build_knowledge_tools
from src.core.config import Settings


def _deps():
    return KnowledgeDeps(
        llm=None,
        embedding_model=None,
        milvus_client=None,
        neo4j_driver=None,
    )


def test_build_knowledge_tools_returns_all_registered_tools():
    tools = build_knowledge_tools(_deps())

    assert [tool.name for tool in tools] == [
        "search_knowledge_docs",
        "search_knowledge_graph",
        "search_knowledge_sql",
        "search_knowledge_multi",
    ]


async def test_document_tool_returns_search_result_without_audit_dependencies(monkeypatch):
    received = {}

    async def fake_rewrite(question, deps):
        received["original_question"] = question
        return "rewritten question"

    async def fake_search_docs(**kwargs):
        received["search_question"] = kwargs["question"]
        return "document answer"

    monkeypatch.setattr(knowledge_tools, "_rewrite", fake_rewrite)
    monkeypatch.setattr(doc_rag, "search_docs", fake_search_docs)
    document_tool = next(
        tool
        for tool in build_knowledge_tools(_deps())
        if tool.name == "search_knowledge_docs"
    )

    result = await document_tool.ainvoke({"question": "original question"})

    assert result == "document answer"
    assert received == {
        "original_question": "original question",
        "search_question": "rewritten question",
    }


class _FakeLlm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        return type("Response", (), {"content": next(self.responses)})()


class _FakeQueryResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _FakeDatabase:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    async def execute(self, statement):
        self.statements.append(str(statement))
        return _FakeQueryResult(self.rows)


async def test_search_sql_uses_validated_sql_to_generate_answer():
    llm = _FakeLlm(["SELECT total FROM visits", "answer"])
    database = _FakeDatabase([{"total": 3}])

    answer = await search_sql("record count", llm, database)

    assert answer == "answer"
    assert database.statements == ["SELECT total FROM visits LIMIT 100"]
    assert "SELECT total FROM visits LIMIT 100" in llm.prompts[1]


def test_validate_sql_rejects_limit_above_maximum():
    valid, _ = _validate_sql("SELECT * FROM visits LIMIT 101")

    assert not valid


def test_validate_sql_rejects_multiple_statements():
    valid, _ = _validate_sql("SELECT * FROM visits; SELECT * FROM drugs")

    assert not valid


def test_validate_sql_rejects_sensitive_patient_field():
    valid, _ = _validate_sql("SELECT p.phone FROM patients AS p")

    assert not valid


def test_validate_cypher_rejects_mutating_statement():
    valid, _ = graph_rag._validate_cypher("CREATE (:Disease {name: 'test'})")

    assert not valid


def test_validate_cypher_adds_result_limit_to_read_query():
    valid, cypher = graph_rag._validate_cypher("MATCH (d:Disease) RETURN d.name")

    assert valid
    assert cypher == "MATCH (d:Disease) RETURN d.name LIMIT 20"


def test_split_markdown_keeps_overlap_when_a_long_paragraph_is_split():
    chunks = _split_markdown("abcdefghij", chunk_size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_settings_disables_mineru_without_api_url():
    settings = Settings(_env_file=None)

    assert settings.MINERU_API_URL == ""
    assert settings.MINERU_BACKEND
    assert settings.MINERU_TIMEOUT > 0


async def test_parse_document_fails_fast_when_mineru_is_not_configured(monkeypatch):
    monkeypatch.setattr(mineru_client.settings, "MINERU_API_URL", "")

    with pytest.raises(RuntimeError, match="MINERU_API_URL"):
        await mineru_client.parse_document("nonexistent.pdf")


class _RecordingMilvusClient:
    def __init__(self):
        self.deleted = []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


class _FailingEmbeddingModel:
    async def aembed_documents(self, texts):
        raise RuntimeError("embedding failed")


async def test_ingest_file_preserves_existing_document_when_embedding_fails(monkeypatch):
    async def fake_mineru_parser(file_path, file_name):
        return "new content"

    monkeypatch.setattr(doc_ingestion, "ensure_knowledge_collection", lambda client: None)
    monkeypatch.setattr(doc_ingestion, "_parse_with_mineru", fake_mineru_parser)
    client = _RecordingMilvusClient()

    with pytest.raises(RuntimeError, match="embedding failed"):
        await doc_ingestion.ingest_file(
            file_path="document.pdf",
            doc_name="document.pdf",
            doc_type="guideline",
            category="medical",
            embedding_model=_FailingEmbeddingModel(),
            milvus_client=client,
        )

    assert client.deleted == []
