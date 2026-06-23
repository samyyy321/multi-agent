from types import SimpleNamespace

import pytest
from langgraph.prebuilt import ToolRuntime


import src.agents.inquiry.graph as inquiry_graph
import src.agents.supervisor_agent as supervisor_agent
import src.agents.worker_tools as worker_tools
import src.api.routers.chat as chat_router
from src.agents.inquiry.state import InquiryPhase
from src.agents.worker_tools import UserContext


class _FakeRedisSaver:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    async def asetup(self):
        return None


@pytest.mark.asyncio
async def test_supervisor_declares_user_context_for_runtime_tools(monkeypatch):
    captured = {}

    monkeypatch.setattr(supervisor_agent, "get_checkpointer_redis", lambda: object())
    monkeypatch.setattr(supervisor_agent, "AsyncRedisSaver", _FakeRedisSaver)
    monkeypatch.setattr(supervisor_agent, "get_milvus_client_alias", lambda: "test")
    monkeypatch.setattr(supervisor_agent, "_get_embedding_model", lambda: object())
    monkeypatch.setattr(supervisor_agent, "MilvusStore", lambda **kwargs: object())
    monkeypatch.setattr(supervisor_agent, "ChatDeepSeek", lambda **kwargs: object())

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(supervisor_agent, "create_agent", fake_create_agent)

    await supervisor_agent.create_supervisor_agent()

    assert captured["context_schema"] is UserContext


class _FakeRedis:
    async def exists(self, key):
        return False


class _FakeStreamingAgent:
    def __init__(self):
        self.invocation_kwargs = None

    async def astream(self, inputs, **kwargs):
        self.invocation_kwargs = kwargs
        message = type("Message", (), {"content": "ok"})()
        yield message, {}


@pytest.mark.asyncio
async def test_streaming_chat_passes_user_context_to_supervisor(monkeypatch):
    agent = _FakeStreamingAgent()
    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: _FakeRedis())

    async def fake_get_supervisor_agent():
        return agent

    monkeypatch.setattr(chat_router, "get_supervisor_agent", fake_get_supervisor_agent)
    request = chat_router.ChatRequest(
        user_id="user-1",
        session_id="session-1",
        message="hello",
        patient_id=42,
    )

    response = await chat_router.chat_stream(request, db=object())
    async for _ in response.body_iterator:
        pass

    assert agent.invocation_kwargs["context"] == UserContext(
        user_id="user-1",
        session_id="session-1",
        patient_id=42,
    )


class _FakeInvocationAgent:
    def __init__(self):
        self.invocation_kwargs = None

    async def ainvoke(self, inputs, **kwargs):
        self.invocation_kwargs = kwargs
        message = type("Message", (), {"content": "ok"})()
        return {"messages": [message]}


@pytest.mark.asyncio
async def test_direct_supervisor_entrypoint_passes_user_context(monkeypatch):
    agent = _FakeInvocationAgent()

    async def fake_get_supervisor_agent():
        return agent

    monkeypatch.setattr(supervisor_agent, "get_supervisor_agent", fake_get_supervisor_agent)

    reply = await supervisor_agent.chat_endpoint("user-1", "session-1", "hello")

    assert reply == "ok"
    assert agent.invocation_kwargs["context"] == UserContext(
        user_id="user-1",
        session_id="session-1",
    )


@pytest.mark.asyncio
async def test_chat_passes_patient_id_to_supervisor_context(monkeypatch):
    agent = _FakeInvocationAgent()
    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: _FakeRedis())

    async def fake_get_supervisor_agent():
        return agent

    monkeypatch.setattr(chat_router, "get_supervisor_agent", fake_get_supervisor_agent)
    request = chat_router.ChatRequest(
        user_id="user-1",
        session_id="session-1",
        message="hello",
        patient_id=42,
    )

    await chat_router.chat(request, db=object())

    assert agent.invocation_kwargs["context"].patient_id == 42


@pytest.mark.asyncio
async def test_inquiry_tool_forwards_patient_id_to_inquiry_flow(monkeypatch):
    captured = {}

    async def fake_run_inquiry(**kwargs):
        captured.update(kwargs)
        return "answer", SimpleNamespace(
            phase=InquiryPhase.END,
            handoff_payload=None,
        )

    monkeypatch.setattr(worker_tools, "build_inquiry_deps", lambda: object())
    monkeypatch.setattr(worker_tools, "run_inquiry", fake_run_inquiry)
    runtime = ToolRuntime(
        state={},
        context=UserContext(user_id="user-1", session_id="session-1", patient_id=42),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call-1",
        store=None,
    )

    result = await worker_tools.call_inquiry_agent.coroutine(
        message="headache",
        runtime=runtime,
    )

    assert result == "answer"
    assert captured["patient_id"] == 42
    assert captured["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_inquiry_state_uses_patient_id_not_user_id_on_first_turn(monkeypatch):
    captured = {}

    class _FakeInquiryGraph:
        async def ainvoke(self, state, config):
            captured["state"] = state
            return state

    monkeypatch.setattr(
        inquiry_graph,
        "build_inquiry_graph",
        lambda deps: _FakeInquiryGraph(),
    )

    _, state = await inquiry_graph.run_inquiry(
        user_message="headache",
        thread_id="user-1:session-1",
        deps=object(),
        user_id="user-1",
        patient_id=42,
    )

    assert captured["state"].patient_context.patient_id == 42
    assert state.patient_context.patient_id == 42


class _NoActiveInquiryStateRedis:
    async def get(self, key):
        return None

    async def delete(self, *keys):
        return None

    async def set(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_active_inquiry_turn_passes_thread_id_to_inquiry_flow(monkeypatch):
    captured = {}

    async def fake_run_inquiry(**kwargs):
        captured.update(kwargs)
        return "answer", SimpleNamespace(
            phase=InquiryPhase.END,
            handoff_payload=None,
        )

    monkeypatch.setattr(chat_router, "build_inquiry_deps", lambda db_session: object())
    monkeypatch.setattr(chat_router, "run_inquiry", fake_run_inquiry)

    reply = await chat_router._run_inquiry_turn(
        message="headache",
        thread_id="user-1:session-1",
        redis=_NoActiveInquiryStateRedis(),
        db=object(),
    )

    assert reply == "answer"
    assert captured["thread_id"] == "user-1:session-1"
