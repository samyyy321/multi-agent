import sys
from types import SimpleNamespace

import scripts.init_postgres as init_postgres

TEST_PATIENT_NAME = "\u6d4b\u8bd5\u60a3\u800542"
TEST_PATIENT_GENDER = "\u7537"
TEST_PATIENT_ALLERGY = "\u9752\u9709\u7d20"
TEST_PATIENT_HISTORY = "\u65e0"


class _RecordingCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))


class _FakeConnection:
    def __init__(self):
        self.cursor_instance = _RecordingCursor()
        self.commits = 0
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


class _ImportCursor(_RecordingCursor):
    def close(self):
        return None


class _ImportConnection(_FakeConnection):
    def __init__(self):
        super().__init__()
        self.cursor_instance = _ImportCursor()
        self.autocommit = None


def test_seed_test_patient_upserts_patient_42_and_updates_sequence():
    cursor = _RecordingCursor()

    patient_id = init_postgres.seed_test_patient(cursor)

    assert patient_id == 42
    insert_sql, insert_params = cursor.calls[0]
    assert "INSERT INTO patients" in insert_sql
    assert "ON CONFLICT (id) DO UPDATE" in insert_sql
    assert insert_params[0] == 42
    assert insert_params[1] == TEST_PATIENT_NAME
    assert insert_params[2] == TEST_PATIENT_GENDER
    assert insert_params[4] == TEST_PATIENT_ALLERGY
    assert insert_params[5] == TEST_PATIENT_HISTORY
    assert "setval" in cursor.calls[1][0]


def test_import_data_runs_test_patient_seed_before_medical_import(monkeypatch):
    connection = _ImportConnection()
    seeded = []

    monkeypatch.setitem(
        sys.modules,
        "psycopg2",
        SimpleNamespace(connect=lambda **kwargs: connection),
    )
    monkeypatch.setattr(init_postgres, "load_medical_data", lambda path: [])

    def fake_seed(cursor):
        seeded.append(cursor)
        return 42

    monkeypatch.setattr(init_postgres, "seed_test_patient", fake_seed)

    init_postgres.import_data()

    assert seeded == [connection.cursor_instance]
    assert connection.commits >= 1
    assert connection.closed
