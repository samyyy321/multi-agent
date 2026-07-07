import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import PrivateAttr

import src.agents.inquiry.db_queries as db_queries
import src.agents.inquiry.graph as inquiry_graph
import src.api.routers.chat as chat_router
from src.agents.inquiry.state import CandidateDisease, InquiryPhase, InquiryState


class _GatedChatModel(FakeListChatModel):
    """控制正文生成进度，验证 SSE 在模型完成之前就发送首个内容块。"""

    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _finished: bool = PrivateAttr(default=False)

    async def _astream(self, messages, **kwargs):
        is_reply = self.responses[self.i] == "请继续描述"
        async for chunk in super()._astream(messages, **kwargs):
            yield chunk
            if is_reply:
                await self._release.wait()
        if is_reply:
            self._finished = True


class _MemoryRedis:
    """仅在内存保存测试会话，避免访问或污染真实 Redis。"""

    def __init__(self, state):
        self.values = {
            "inquiry_active:user:session": "1",
            "inquiry_state:user:session": state.model_dump_json(),
        }
        self.expiry = {}

    async def exists(self, key):
        return key in self.values

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex):
        self.values[key] = value
        self.expiry[key] = ex

    async def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)


def _configure_flow(monkeypatch, mode, llm):
    """保留真实问诊图和模型流，仅替换数据库、图谱及症状检索依赖。"""
    state = InquiryState(
        session_id="user:session", round=1,
        messages=[AIMessage(content="previous reply")],
    )
    redis = _MemoryRedis(state)
    deps = SimpleNamespace(llm=llm, symptom_llm=llm, neo4j_driver=None,
                           embedding_model=None, milvus_client=None, db_session=None)
    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: redis)
    monkeypatch.setattr(chat_router, "build_inquiry_deps", lambda db_session: deps)
    monkeypatch.setattr(inquiry_graph, "normalize_symptoms", AsyncMock(return_value={
        "all_standard": [] if mode == "clarify" else ["fever"], "unmatched": [],
    }))
    candidates = [CandidateDisease(name="cold", confidence=0.3, department="general")]
    monkeypatch.setattr(inquiry_graph, "query_candidate_diseases", AsyncMock(return_value=candidates))
    monkeypatch.setattr(inquiry_graph, "enrich_candidate_details", AsyncMock(return_value=candidates))
    monkeypatch.setattr(inquiry_graph, "check_convergence", lambda *args: (mode == "conclude", False))
    monkeypatch.setattr(inquiry_graph, "get_pending_symptoms", AsyncMock(return_value=[("cough", 1)]))
    monkeypatch.setattr(db_queries, "save_consultation_record", AsyncMock(return_value=1))
    return redis


def _event(raw):
    return json.loads(raw.removeprefix("data: ").strip())


@pytest.mark.parametrize("mode,internal", [
    ("clarify", []), ("ask", ['["cough"]']), ("conclude", ['{"decision":"accept"}']),
])
async def test_active_inquiry_streams_only_reply_before_completion(monkeypatch, mode, internal):
    llm = _GatedChatModel(responses=[*internal, "请继续描述"])
    redis = _configure_flow(monkeypatch, mode, llm)
    original_state = redis.values["inquiry_state:user:session"]
    request = chat_router.ChatRequest(user_id="user", session_id="session", message="fever")
    response = await chat_router.chat_stream(request, db=object())
    events = []
    try:
        first = _event(await asyncio.wait_for(anext(response.body_iterator), timeout=5))
        assert first == {"type": "token", "content": "请"}
        assert not llm._finished
        assert redis.values["inquiry_state:user:session"] == original_state
        events.append(first)
        llm._release.set()
        async for raw in response.body_iterator:
            event = _event(raw)
            if event["type"] == "done":
                if mode == "conclude":
                    assert redis.values == {}
                else:
                    saved = InquiryState.model_validate_json(redis.values["inquiry_state:user:session"])
                    assert saved.messages[-1].content == "请继续描述"
                    assert saved.round == 2
                    assert redis.expiry["inquiry_active:user:session"] == 3600
            events.append(event)
    finally:
        llm._release.set()
        await response.body_iterator.aclose()

    assert "".join(e["content"] for e in events if e["type"] == "token") == "请继续描述"
    assert events[-1] == {"type": "done", "session_id": "session"}
    assert all(e["type"] != "error" for e in events)


async def test_active_inquiry_stream_reports_error_without_saving_partial_state(monkeypatch):
    llm = FakeListChatModel(responses=["请继续描述"], error_on_chunk_number=1)
    redis = _configure_flow(monkeypatch, "clarify", llm)
    original = dict(redis.values)
    request = chat_router.ChatRequest(user_id="user", session_id="session", message="fever")

    response = await chat_router.chat_stream(request, db=object())
    events = [_event(raw) async for raw in response.body_iterator]

    assert events[0] == {"type": "token", "content": "请"}
    assert events[-1]["type"] == "error"
    assert not any(event["type"] == "done" for event in events)
    assert redis.values == original


async def test_active_inquiry_stream_can_close_before_generation_finishes(monkeypatch):
    llm = _GatedChatModel(responses=["请继续描述"])
    redis = _configure_flow(monkeypatch, "clarify", llm)
    original = dict(redis.values)
    request = chat_router.ChatRequest(user_id="user", session_id="session", message="fever")
    response = await chat_router.chat_stream(request, db=object())

    try:
        assert _event(await asyncio.wait_for(anext(response.body_iterator), 5))["content"] == "请"
        await asyncio.wait_for(response.body_iterator.aclose(), timeout=5)
        assert not llm._finished
        assert redis.values == original
    finally:
        llm._release.set()
        await response.body_iterator.aclose()
