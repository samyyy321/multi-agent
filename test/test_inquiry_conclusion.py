import json

import pytest

import src.agents.inquiry.graph as inquiry_graph
from src.agents.inquiry.state import CandidateDisease, InquiryState, PatientContext


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _SequencedLlm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def ainvoke(self, messages, config=None):
        self.prompts.append(messages[0].content)
        return _FakeResponse(next(self.responses))


@pytest.mark.asyncio
async def test_conclusion_uses_provisional_direction_when_llm_rejects_graph_candidate():
    llm = _SequencedLlm(
        [
            json.dumps(
                {
                    "decision": "provisional",
                    "disease": "acute respiratory infection tendency",
                    "department": "respiratory medicine",
                    "reason": "symptoms are nonspecific",
                }
            ),
            "final conclusion",
        ]
    )
    state = InquiryState(
        session_id="test-session",
        confirmed_symptoms=["fever", "headache"],
        candidate_diseases=[
            CandidateDisease(
                name="unreasonable graph candidate",
                confidence=1.0,
                base_confidence=1.0,
                matched_symptoms=["fever", "headache"],
                department="cardiology",
            )
        ],
        patient_context=PatientContext(patient_id=42),
    )
    deps = type("Deps", (), {"llm": llm})()

    result = await inquiry_graph.node_conclude(state, deps)

    handoff = result["handoff_payload"]
    assert handoff.primary_disease == "acute respiratory infection tendency\uff08\u5f85\u6392\uff09"
    assert handoff.primary_confidence == 0.0
    assert handoff.department == "respiratory medicine"
    assert "confidence" not in llm.prompts[1].lower()
    assert "1.0" not in llm.prompts[1]


@pytest.mark.asyncio
async def test_conclusion_falls_back_to_graph_candidate_when_review_is_not_json():
    llm = _SequencedLlm(["not json", "final conclusion"])
    state = InquiryState(
        session_id="test-session",
        confirmed_symptoms=["fever", "headache"],
        candidate_diseases=[
            CandidateDisease(
                name="graph candidate",
                confidence=0.5,
                base_confidence=0.5,
                matched_symptoms=["headache"],
                department="general medicine",
            )
        ],
    )
    deps = type("Deps", (), {"llm": llm})()

    result = await inquiry_graph.node_conclude(state, deps)

    assert result["handoff_payload"].primary_disease == "graph candidate"
    assert result["messages"][0].content == "final conclusion"


@pytest.mark.asyncio
async def test_conclusion_uses_llm_direction_when_graph_has_no_candidate():
    llm = _SequencedLlm(
        [
            json.dumps(
                {
                    "decision": "provisional",
                    "disease": "undifferentiated febrile illness",
                    "department": "general medicine",
                    "reason": "no graph candidate is available",
                }
            ),
            "final conclusion",
        ]
    )
    state = InquiryState(
        session_id="test-session",
        confirmed_symptoms=["fever"],
    )
    deps = type("Deps", (), {"llm": llm})()

    result = await inquiry_graph.node_conclude(state, deps)

    assert result["handoff_payload"].primary_disease == (
        "undifferentiated febrile illness\uff08\u5f85\u6392\uff09"
    )
    assert result["messages"][0].content == "final conclusion"


@pytest.mark.asyncio
async def test_provisional_review_retries_when_llm_returns_generic_direction():
    llm = _SequencedLlm(
        [
            json.dumps(
                {
                    "decision": "provisional",
                    "disease": "\u53d1\u70ed\u5934\u75db\u5f85\u67e5",
                    "department": "\u611f\u67d3\u79d1",
                    "reason": "symptoms are nonspecific",
                }
            ),
            json.dumps(
                {
                    "decision": "provisional",
                    "disease": "\u6025\u6027\u4e0a\u547c\u5438\u9053\u611f\u67d3",
                    "department": "\u547c\u5438\u5185\u79d1",
                    "reason": "symptoms fit an acute respiratory infection",
                }
            ),
            "final conclusion",
        ]
    )
    state = InquiryState(
        session_id="test-session",
        confirmed_symptoms=["\u53d1\u70e7", "\u5934\u75db"],
        candidate_diseases=[
            CandidateDisease(
                name="\u80a2\u7aef\u80a5\u5927\u75c7\u6027\u5fc3\u808c\u75c5",
                confidence=1.0,
                base_confidence=1.0,
                matched_symptoms=["\u5934\u75db"],
                department="\u5fc3\u5185\u79d1",
            )
        ],
    )
    deps = type("Deps", (), {"llm": llm})()

    result = await inquiry_graph.node_conclude(state, deps)

    assert result["handoff_payload"].primary_disease == (
        "\u6025\u6027\u4e0a\u547c\u5438\u9053\u611f\u67d3\uff08\u5f85\u6392\uff09"
    )
    assert len(llm.prompts) == 3
