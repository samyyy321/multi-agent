import pytest

from src.agents.inquiry.confidence import check_convergence
from src.agents.inquiry.neo4j_queries import query_candidate_diseases
from src.agents.inquiry.state import CandidateDisease


class _FakeResult:
    def __init__(self, records):
        self.records = records

    async def data(self):
        return self.records


class _FakeSession:
    def __init__(self, records):
        self.records = records
        self.parameters = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def run(self, query, **parameters):
        self.parameters = parameters
        return _FakeResult(self.records)


class _FakeDriver:
    def __init__(self, records):
        self.session_instance = _FakeSession(records)

    def session(self):
        return self.session_instance


@pytest.mark.asyncio
async def test_candidate_score_uses_confirmed_symptom_coverage():
    driver = _FakeDriver(
        [
            {
                "disease": "single-match disease",
                "matched_symptoms": ["headache"],
                "matched_count": 1,
                "total_symptoms": 1,
                "base_confidence": 0.5,
            }
        ]
    )

    candidates = await query_candidate_diseases(
        confirmed_symptoms=["fever", "headache"],
        neo4j_driver=driver,
    )

    assert driver.session_instance.parameters["confirmed_count"] == 2
    assert candidates[0].base_confidence == 0.5


@pytest.mark.asyncio
async def test_candidate_query_prefers_multi_symptom_matches_when_available():
    driver = _FakeDriver(
        [
            {
                "disease": "single-match disease",
                "matched_symptoms": ["headache"],
                "matched_count": 1,
                "total_symptoms": 1,
                "base_confidence": 0.5,
            },
            {
                "disease": "multi-match disease",
                "matched_symptoms": ["fever", "headache"],
                "matched_count": 2,
                "total_symptoms": 4,
                "base_confidence": 1.0,
            },
        ]
    )

    candidates = await query_candidate_diseases(
        confirmed_symptoms=["fever", "headache"],
        neo4j_driver=driver,
    )

    assert [candidate.name for candidate in candidates] == ["multi-match disease"]


def test_single_symptom_match_cannot_trigger_diagnosis_convergence():
    candidate = CandidateDisease(
        name="single-match disease",
        confidence=1.0,
        base_confidence=1.0,
        matched_symptoms=["headache"],
    )

    assert check_convergence([candidate], current_round=0) == (False, False)
