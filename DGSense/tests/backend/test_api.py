"""
Automated unit and integration tests for DGSense backend API — V2 Two-Way System.
"""

import math
import pytest
from fastapi.testclient import TestClient
import torch
from backend.main import app
from backend.model_service import ModelService, DGSenseTemporalGRU
from backend.config import settings

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "DGSense" in data["service"]
    assert len(data["supported_directions"]) == 2


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["feature_schema_version"] == settings.FEATURE_SCHEMA_VERSION
    assert "device" in data
    assert data["two_way_supported"] is True
    assert data["sign_language_code"] == "dgs-DE"


def test_list_signs_endpoint():
    response = client.get("/api/signs")
    assert response.status_code == 200
    data = response.json()
    assert data["total_signs"] > 0
    assert data["model_supported_count"] >= 10
    assert data["dictionary_only_count"] > 0

    # Ensure each item has required bilingual and support metadata
    for item in data["signs"]:
        assert "canonical_label" in item
        assert "german_text" in item
        assert "english_text" in item
        assert "is_model_supported" in item
        assert "is_dictionary_only" in item


def test_predict_endpoint_missing_model_returns_503():
    dummy_sequence = [[0.1] * 144 for _ in range(30)]
    payload = {
        "feature_schema_version": "dgsense-v1",
        "sequence_length": 30,
        "feature_dim": 144,
        "landmarks": dummy_sequence,
        "request_id": "test-req-1"
    }

    from backend.model_service import model_service
    original_state = model_service.is_model_loaded
    try:
        model_service.is_model_loaded = False
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 503
        assert "Model checkpoint is not loaded" in response.json()["detail"]
    finally:
        model_service.is_model_loaded = original_state


def test_predict_validation_rejects_nan_and_inf():
    nan_sequence = [[0.1] * 144 for _ in range(30)]
    nan_sequence[5][10] = float("nan")

    from backend.schemas import PredictRequest
    with pytest.raises(ValueError, match="NaN or Infinite"):
        PredictRequest(
            feature_schema_version="dgsense-v1",
            sequence_length=30,
            feature_dim=144,
            landmarks=nan_sequence
        )

    null_sequence = [[0.1] * 144 for _ in range(30)]
    null_sequence[5][10] = None
    payload = {
        "feature_schema_version": "dgsense-v1",
        "sequence_length": 30,
        "feature_dim": 144,
        "landmarks": null_sequence
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_rejects_wrong_dimensions():
    short_sequence = [[0.1] * 144 for _ in range(20)]
    payload = {
        "feature_schema_version": "dgsense-v1",
        "sequence_length": 20,
        "feature_dim": 144,
        "landmarks": short_sequence
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (422, 503)


def test_sentence_composer():
    # 1. Verified idiom template
    payload = {"sign_sequence": ["HELLO", "HOW_ARE_YOU"]}
    response = client.post("/api/compose-sentence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["composition_type"] == "verified_phrase"
    assert "Hallo! Wie geht es dir?" in data["german_sentence"]
    assert "Hello! How are you?" in data["english_sentence"]
    assert data["is_continuous_translation"] is False

    # 2. Approximate composition fallback
    payload = {"sign_sequence": ["HELLO", "THANK_YOU", "WATER"]}
    response = client.post("/api/compose-sentence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["composition_type"] == "approximate_composition"
    assert "Hallo" in data["german_sentence"]
    assert "Danke" in data["german_sentence"]
    assert "Wasser" in data["german_sentence"]
    assert data["is_continuous_translation"] is False

    # 3. Empty sequence
    payload = {"sign_sequence": []}
    response = client.post("/api/compose-sentence", json=payload)
    assert response.status_code == 200
    assert response.json()["composition_type"] == "empty"


# --- V2 Reverse Translation Tests (Direction B: German -> DGS) ---

def test_reverse_translate_exact_phrase():
    payload = {"text": "Guten Morgen", "source_lang": "de"}
    response = client.post("/api/reverse-translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["match_type"] == "exact_phrase_match"
    assert data["dgs_gloss_sequence"] == ["GUT", "MORGEN"]
    assert len(data["clips"]) > 0
    assert data["clips"][0]["canonical_meaning"] == "GOOD_MORNING"
    assert data["clips"][0]["is_verified"] is True
    assert "Meine DGS" in data["clips"][0]["source_attribution"] or "DGS Corpus" in data["clips"][0]["source_attribution"]


def test_reverse_translate_polite_phrases():
    for phrase, expected_canon in [
        ("Danke", "THANK_YOU"),
        ("Vielen Dank", "THANK_YOU_VERY_MUCH"),
        ("Bitte", "PLEASE"),
        ("Entschuldigung", "SORRY"),
        ("Auf Wiedersehen", "GOODBYE"),
    ]:
        resp = client.post("/api/reverse-translate", json={"text": phrase})
        assert resp.status_code == 200
        d = resp.json()
        assert d["match_type"] == "exact_phrase_match"
        assert len(d["clips"]) > 0
        assert d["clips"][0]["canonical_meaning"] == expected_canon


def test_reverse_translate_compositional():
    payload = {"text": "Hallo wie geht es dir"}
    response = client.post("/api/reverse-translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["match_type"] == "compositional_sequence"
    assert len(data["clips"]) >= 2
    assert "HALLO" in data["dgs_gloss_sequence"]


def test_reverse_translate_unsupported_phrase_no_fake_signs():
    # Complex sentence without verified DGS equivalent
    payload = {"text": "Die quantenmechanische Wellenfunktion kollabiert im Experiment."}
    response = client.post("/api/reverse-translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["match_type"] == "unsupported"
    assert len(data["clips"]) == 0
    assert data["unsupported_message"] is not None
    assert "keine verifizierte DGS-Gebärdenfolge" in data["unsupported_message"]


def test_reverse_phrases_catalog():
    response = client.get("/api/reverse-translate/phrases")
    assert response.status_code == 200
    data = response.json()
    assert data["total_phrases"] >= 10
    for p in data["phrases"]:
        assert "clip_id" in p
        assert "canonical_meaning" in p
        assert "german_phrase" in p
        assert p["is_verified"] is True


# --- V2 Dataset Manifest & Readiness Tests ---

def test_dataset_manifest_endpoint():
    response = client.get("/api/dataset/manifest")
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] >= 20
    assert data["language_code"] == "dgs-DE"
    assert data["verified_research_clips"] > 0
    assert data["data_required_count"] > 0  # Confirms unverified items are flagged DATA_REQUIRED

    # Check that required fields exist
    sample = data["items"][0]
    assert "phrase_id" in sample
    assert "canonical_meaning" in sample
    assert "german_text" in sample
    assert "verification_status" in sample
    assert "licence_notes" in sample


def test_dataset_readiness_endpoint():
    response = client.get("/api/dataset/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["total_target_concepts"] >= 20
    assert data["data_required_count"] > 0
    assert len(data["missing_concepts"]) == data["data_required_count"]
    assert data["download_limit_mb"] == 5120
    assert len(data["reference_sources"]) >= 3


def test_language_configuration_endpoint():
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert data["sign_language_code"] == "dgs-DE"
    assert data["spoken_language_code"] == "de-DE"
    assert data["feature_schema_version"] == "dgsense-v1"


def test_model_inference_with_synthetic_weights(tmp_path):
    import json as _json
    test_artifacts_dir = tmp_path / "artifacts"
    test_artifacts_dir.mkdir()

    import shutil
    shutil.copy(settings.ARTIFACTS_DIR / "labels.json", test_artifacts_dir / "labels.json")
    shutil.copy(settings.ARTIFACTS_DIR / "feature_schema.json", test_artifacts_dir / "feature_schema.json")
    shutil.copy(settings.ARTIFACTS_DIR / "model_config.json", test_artifacts_dir / "model_config.json")
    shutil.copy(settings.ARTIFACTS_DIR / "sign_clips.json", test_artifacts_dir / "sign_clips.json")
    shutil.copy(settings.ARTIFACTS_DIR / "everyday_manifest.json", test_artifacts_dir / "everyday_manifest.json")

    # Build a checkpoint that matches the architecture declared in model_config.json
    # (input_dim 144 raw, or 288 with velocity stacking; optional attention pooling).
    arch = _json.loads(
        (test_artifacts_dir / "model_config.json").read_text(encoding="utf-8")
    ).get("architecture", {})
    test_model = DGSenseTemporalGRU(
        input_dim=arch.get("input_dim", 144),
        hidden_dim=arch.get("hidden_dim", 128),
        num_layers=arch.get("num_layers", 2),
        num_classes=arch.get("num_classes", 10),
        dropout=0.1,
        bidirectional=arch.get("bidirectional", True),
        use_attention=arch.get("use_attention", False),
    )
    test_ckpt_path = test_artifacts_dir / "model.pt"
    torch.save(test_model.state_dict(), test_ckpt_path)

    service = ModelService(artifacts_dir=test_artifacts_dir)
    assert service.is_model_loaded is True

    valid_seq = [[0.05] * 144 for _ in range(30)]

    resp_uncertain = service.predict(
        landmarks=valid_seq,
        feature_schema_version="dgsense-v1",
        sequence_length=30,
        feature_dim=144,
        min_confidence=0.9999
    )
    assert resp_uncertain.status == "uncertain"
    assert resp_uncertain.prediction is None
    assert len(resp_uncertain.top_alternatives) > 0

    resp_confident = service.predict(
        landmarks=valid_seq,
        feature_schema_version="dgsense-v1",
        sequence_length=30,
        feature_dim=144,
        min_confidence=0.0001
    )
    assert resp_confident.status == "ok"
    assert resp_confident.prediction is not None
    assert resp_confident.german_text is not None
    assert resp_confident.english_text is not None
