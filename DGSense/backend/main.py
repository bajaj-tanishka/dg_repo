"""
DGSense FastAPI Backend Application — V2 Two-Way System
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure root workspace directory is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.config import settings
from backend import database
from backend.schemas import (
    HealthResponse,
    SignListResponse,
    PredictRequest,
    PredictResponse,
    ComposeSentenceRequest,
    ComposeSentenceResponse,
    ReverseTranslateRequest,
    ReverseTranslateResponse,
    ReversePhraseListResponse,
    DatasetManifestResponse,
    DatasetReadinessResponse,
    LanguageConfigResponse,
    RecognitionHistoryResponse,
    ConversationMessageIn,
    ConversationMessageOut,
    ConversationListResponse,
    PracticeAttemptIn,
    PracticeAttemptOut,
    PracticeHistoryResponse,
    RecordedSampleIn,
    RecordedSampleSummary,
    RecordedSampleListResponse,
    DeleteResponse,
)
from backend.model_service import model_service

# Create the SQLite database on startup (tables are idempotent).
database.init_db()

app = FastAPI(
    title="DGSense API — Two-Way German Sign Language Translation",
    description="Real-Time DGS Recognition (Webcam -> German) & Reverse Translation (German -> DGS Sign Clips)",
    version="2.0.0"
)

# Configure CORS for local development and LAN / Wi-Fi access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "service": "DGSense Two-Way German Sign Language System",
        "version": "2.0.0",
        "supported_directions": [
            "Direction A: DGS Video/Landmarks -> German Text -> English & Speech",
            "Direction B: German Text/Speech -> Verified DGS Sign Output"
        ],
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Returns service status, model checkpoint readiness, device, two-way readiness, and active language codes.
    """
    model_loaded = model_service.is_model_loaded
    supported_labels = len(model_service.label_map) if model_loaded else 0

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_service.model_version if model_loaded else None,
        num_supported_labels=supported_labels,
        device=str(model_service.device),
        feature_schema_version=settings.FEATURE_SCHEMA_VERSION,
        details=(
            "Model checkpoint loaded and active for inference; reverse translation ready."
            if model_loaded
            else "Backend operational in inspection mode; model checkpoint (model.pt) not loaded. Reverse translation and dictionary remain active."
        ),
        two_way_supported=True,
        sign_language_code=settings.SIGN_LANGUAGE_CODE,
        spoken_language_code=settings.SPOKEN_LANGUAGE_CODE
    )


# --- Direction A Endpoints: DGS -> German ---

@app.get("/api/signs", response_model=SignListResponse, tags=["Vocabulary"])
def list_signs():
    """
    Returns the supported DGS vocabulary, explicitly distinguishing model-supported
    signs from dictionary-only reference entries.
    """
    signs = model_service.get_supported_signs()
    model_supported_count = sum(1 for s in signs if s.is_model_supported)
    dictionary_only_count = sum(1 for s in signs if s.is_dictionary_only)

    return SignListResponse(
        total_signs=len(signs),
        model_supported_count=model_supported_count,
        dictionary_only_count=dictionary_only_count,
        signs=signs
    )


@app.post("/api/predict", response_model=PredictResponse, tags=["Direction A: Sign -> Text"])
def predict_sign(request: PredictRequest):
    """
    Accepts a 30-frame x 144-feature temporal sequence of MediaPipe landmarks.
    Validates dimensional schema, rejects invalid/NaN values, and runs PyTorch inference.
    """
    if not model_service.is_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model checkpoint is not loaded. Train the model using 'notebooks/DGSense_Training.ipynb' "
                "or place 'model.pt' into 'backend/artifacts/'. Camera preview and landmark tracking remain functional."
            )
        )

    try:
        response = model_service.predict(
            landmarks=request.landmarks,
            feature_schema_version=request.feature_schema_version,
            sequence_length=request.sequence_length,
            feature_dim=request.feature_dim,
            request_id=request.request_id
        )
        # Persist successful recognitions so history survives page refreshes.
        if response.status == "ok" and response.prediction:
            try:
                database.insert_recognition_event(
                    canonical_label=response.prediction,
                    german_text=response.german_text,
                    english_text=response.english_text,
                    confidence=response.confidence,
                    status=response.status,
                    request_id=response.request_id,
                )
            except Exception:
                pass  # never fail inference because of the history store
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/compose-sentence", response_model=ComposeSentenceResponse, tags=["Direction A: Sign -> Text"])
def compose_sentence(request: ComposeSentenceRequest):
    """
    Combines recognized sign sequences into German and English sentences.
    Applies verified phrase templates or transparent approximate composition.
    """
    return model_service.compose_sentence(request.sign_sequence)


# --- Direction B Endpoints: German Text/Speech -> DGS Sign Output ---

@app.post("/api/reverse-translate", response_model=ReverseTranslateResponse, tags=["Direction B: Text -> Sign"])
def reverse_translate(request: ReverseTranslateRequest):
    """
    Converts German text or speech transcript into verified DGS sign sequences or clips.
    Uses exact phrase matching, compositional sequence ordering, or returns an explicit unsupported state.
    """
    return model_service.reverse_translate(request.text)


@app.get("/api/reverse-translate/phrases", response_model=ReversePhraseListResponse, tags=["Direction B: Text -> Sign"])
def list_reverse_phrases():
    """
    Returns the complete catalog of verified DGS sign phrases and video references for reverse translation.
    """
    phrases = model_service.get_all_reverse_phrases()
    return ReversePhraseListResponse(
        total_phrases=len(phrases),
        phrases=phrases
    )


# --- Dataset, Diagnostics & Multi-Language Endpoints ---

@app.get("/api/dataset/manifest", response_model=DatasetManifestResponse, tags=["Dataset & Research"])
def get_dataset_manifest():
    """
    Returns the target everyday communication vocabulary manifest with verification status,
    research citations, and licensing notes.
    """
    return model_service.get_dataset_manifest()


@app.get("/api/dataset/readiness", response_model=DatasetReadinessResponse, tags=["Dataset & Research"])
def get_dataset_readiness():
    """
    Returns the dataset readiness audit report, identifying which concepts are ready vs marked DATA_REQUIRED.
    """
    return model_service.get_dataset_readiness()


@app.get("/api/languages", response_model=LanguageConfigResponse, tags=["Multi-Language"])
def get_language_configuration():
    """
    Provides sign language and spoken language extensibility configuration.
    """
    return model_service.get_language_config()


@app.get("/api/model/info", tags=["Diagnostics"])
def get_model_info():
    """
    Provides comprehensive model architecture, feature schema, and dataset status.
    """
    return {
        "model_loaded": model_service.is_model_loaded,
        "model_version": model_service.model_version,
        "model_config": model_service.model_config,
        "feature_schema": model_service.feature_schema,
        "supported_classes": list(model_service.label_map.values()),
        "total_vocabulary_in_dict": len(model_service.labels_data),
        "min_confidence_threshold": settings.MIN_CONFIDENCE,
        "device": str(model_service.device),
        "two_way_translation": True,
        "reverse_clips_available": len(model_service.sign_clips)
    }


# --- Persistence Endpoints: history, conversation, practice, recordings ---

@app.get("/api/history", response_model=RecognitionHistoryResponse, tags=["Persistence"])
def get_history(limit: int = 50):
    """Recognition history (Direction A), newest first. Stored in SQLite."""
    items = database.list_recognition_events(limit=max(1, min(limit, 500)))
    return RecognitionHistoryResponse(total=database.count_recognition_events(), items=items)


@app.delete("/api/history", response_model=DeleteResponse, tags=["Persistence"])
def clear_history():
    """Clears the stored recognition history."""
    return DeleteResponse(deleted=database.clear_recognition_events())


@app.get("/api/conversations", response_model=ConversationListResponse, tags=["Persistence"])
def get_conversations(limit: int = 500):
    """Two-way conversation transcript, oldest first."""
    messages = database.list_conversation_messages(limit=max(1, min(limit, 2000)))
    return ConversationListResponse(total=len(messages), messages=messages)


@app.post("/api/conversations", response_model=ConversationMessageOut, tags=["Persistence"])
def add_conversation_message(msg: ConversationMessageIn):
    """Appends a conversation message (both speakers)."""
    msg_id = database.insert_conversation_message(
        speaker=msg.speaker, speaker_name=msg.speaker_name, text=msg.text,
        translation=msg.translation, glosses=msg.glosses, match_type=msg.match_type,
        clips=msg.clips,
    )
    stored = next(m for m in database.list_conversation_messages() if m["id"] == msg_id)
    return ConversationMessageOut(**stored)


@app.delete("/api/conversations", response_model=DeleteResponse, tags=["Persistence"])
def clear_conversations():
    """Clears the conversation transcript."""
    return DeleteResponse(deleted=database.clear_conversation_messages())


@app.get("/api/practice/attempts", response_model=PracticeHistoryResponse, tags=["Persistence"])
def get_practice_attempts(limit: int = 100):
    """Practice Mode results with overall accuracy."""
    attempts = database.list_practice_attempts(limit=max(1, min(limit, 500)))
    correct = sum(1 for a in attempts if a["correct"])
    total = len(attempts)
    return PracticeHistoryResponse(
        total=total, correct=correct,
        accuracy=round(correct / total, 4) if total else 0.0,
        attempts=attempts,
    )


@app.post("/api/practice/attempts", response_model=PracticeAttemptOut, tags=["Persistence"])
def add_practice_attempt(attempt: PracticeAttemptIn):
    """Records one Practice Mode attempt (target vs recognized sign)."""
    att_id = database.insert_practice_attempt(
        target_sign=attempt.target_sign, recognized_sign=attempt.recognized_sign,
        correct=attempt.correct, confidence=attempt.confidence,
    )
    stored = next(a for a in database.list_practice_attempts() if a["id"] == att_id)
    return PracticeAttemptOut(**stored)


@app.delete("/api/practice/attempts", response_model=DeleteResponse, tags=["Persistence"])
def clear_practice_attempts():
    """Clears stored practice results."""
    return DeleteResponse(deleted=database.clear_practice_attempts())


@app.get("/api/samples", response_model=RecordedSampleListResponse, tags=["Persistence"])
def list_samples(include_frames: bool = False):
    """Sign Recorder samples stored for future retraining."""
    samples = database.list_recorded_samples(include_frames=include_frames)
    return RecordedSampleListResponse(total=len(samples), samples=samples)


@app.get("/api/samples/{sample_id}", response_model=RecordedSampleSummary, tags=["Persistence"])
def get_sample(sample_id: int):
    """Single recorded sample including its 30x144 landmark frames."""
    sample = database.get_recorded_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found.")
    return RecordedSampleSummary(**sample)


@app.post("/api/samples", response_model=RecordedSampleSummary, tags=["Persistence"])
def upload_sample(sample: RecordedSampleIn):
    """Uploads a 30-frame landmark clip recorded in the browser (Sign Recorder)."""
    sample_id = database.insert_recorded_sample(
        label=sample.label, frames=sample.frames,
        signer_id=sample.signer_id, notes=sample.notes,
    )
    return RecordedSampleSummary(**database.get_recorded_sample(sample_id))


@app.delete("/api/samples/{sample_id}", response_model=DeleteResponse, tags=["Persistence"])
def delete_sample(sample_id: int):
    """Deletes a single recorded sample."""
    return DeleteResponse(deleted=database.delete_recorded_sample(sample_id))


@app.delete("/api/samples", response_model=DeleteResponse, tags=["Persistence"])
def clear_samples():
    """Deletes all recorded samples."""
    return DeleteResponse(deleted=database.clear_recorded_samples())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
