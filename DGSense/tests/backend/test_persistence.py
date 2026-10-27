"""Tests for DGSense V3 SQLite persistence endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend import database
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the persistence layer at a throwaway database for every test."""
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_dgsense.db")
    database.init_db()
    yield
    for clearer in (
        database.clear_recognition_events,
        database.clear_conversation_messages,
        database.clear_practice_attempts,
        database.clear_recorded_samples,
    ):
        clearer()


# --- recognition history ---

def test_history_starts_empty_and_clears():
    res = client.get("/api/history")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 0
    assert data["items"] == []

    res = client.delete("/api/history")
    assert res.status_code == 200
    assert res.json()["deleted"] == 0


def test_history_roundtrip_via_database():
    database.insert_recognition_event(
        canonical_label="RAIN", german_text="Regen", english_text="Rain",
        confidence=0.91, status="ok", request_id="req-1",
    )
    res = client.get("/api/history?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    item = data["items"][0]
    assert item["canonical_label"] == "RAIN"
    assert item["german_text"] == "Regen"
    assert item["confidence"] == pytest.approx(0.91)

    res = client.delete("/api/history")
    assert res.json()["deleted"] == 1
    assert client.get("/api/history").json()["total"] == 0


# --- conversation ---

def test_conversation_crud_roundtrip():
    payload = {
        "speaker": "dgs_user",
        "speaker_name": "Deaf User (DGS Video)",
        "text": "Guten Morgen",
        "translation": "Good morning",
        "glosses": ["GUT", "MORGEN"],
        "match_type": "exact_phrase_match",
        "clips": [],
    }
    res = client.post("/api/conversations", json=payload)
    assert res.status_code == 200
    created = res.json()
    assert created["id"] > 0
    assert created["glosses"] == ["GUT", "MORGEN"]

    res = client.get("/api/conversations")
    assert res.json()["total"] == 1
    assert res.json()["messages"][0]["text"] == "Guten Morgen"

    res = client.delete("/api/conversations")
    assert res.json()["deleted"] == 1
    assert client.get("/api/conversations").json()["total"] == 0


def test_conversation_preserves_order():
    for i in range(3):
        client.post("/api/conversations", json={
            "speaker": "hearing_user" if i % 2 else "dgs_user",
            "speaker_name": "Speaker",
            "text": f"message-{i}",
            "glosses": [],
            "clips": [],
        })
    messages = client.get("/api/conversations").json()["messages"]
    assert [m["text"] for m in messages] == ["message-0", "message-1", "message-2"]


# --- practice attempts ---

def test_practice_attempts_and_accuracy():
    attempts = [
        ("RAIN", "RAIN", True, 0.9),
        ("RAIN", "SNOW", False, 0.6),
        ("SUN", "SUN", True, 0.8),
    ]
    for target, recognized, correct, conf in attempts:
        res = client.post("/api/practice/attempts", json={
            "target_sign": target, "recognized_sign": recognized,
            "correct": correct, "confidence": conf,
        })
        assert res.status_code == 200

    res = client.get("/api/practice/attempts")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert data["correct"] == 2
    assert data["accuracy"] == pytest.approx(0.6667, abs=1e-3)

    assert client.delete("/api/practice/attempts").json()["deleted"] == 3


def test_practice_rejects_out_of_range_confidence():
    res = client.post("/api/practice/attempts", json={
        "target_sign": "SUN", "correct": True, "confidence": 1.5,
    })
    assert res.status_code == 422


# --- recorded samples ---

def _frames():
    return [[0.1] * 144 for _ in range(30)]


def test_sample_upload_list_delete_roundtrip():
    res = client.post("/api/samples", json={
        "label": "RAIN", "frames": _frames(), "signer_id": "signer_test",
        "notes": "30/30 hands",
    })
    assert res.status_code == 200
    sample = res.json()
    assert sample["frame_count"] == 30
    sid = sample["id"]

    listed = client.get("/api/samples").json()
    assert listed["total"] == 1
    assert not listed["samples"][0].get("frames")  # summary excludes frames

    with_frames = client.get("/api/samples?include_frames=true").json()
    assert len(with_frames["samples"][0]["frames"]) == 30

    detail = client.get(f"/api/samples/{sid}").json()
    assert detail["frame_count"] == 30
    assert len(detail["frames"][0]) == 144

    assert client.delete(f"/api/samples/{sid}").json()["deleted"] == 1
    assert client.get("/api/samples").json()["total"] == 0


def test_sample_validation_rejects_wrong_shape():
    res = client.post("/api/samples", json={"label": "X", "frames": [[0.1] * 144] * 10})
    assert res.status_code == 422

    res = client.post("/api/samples", json={"label": "X", "frames": [[0.1] * 10] * 30})
    assert res.status_code == 422

    # NaN cannot travel through JSON (httpx), so validate at the schema layer
    from backend.schemas import RecordedSampleIn
    bad = _frames()
    bad[3][5] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        RecordedSampleIn(label="X", frames=bad)


def test_sample_missing_detail_returns_404():
    res = client.get("/api/samples/99999")
    assert res.status_code == 404