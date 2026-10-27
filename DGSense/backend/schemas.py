"""
Pydantic schemas for DGSense API requests and responses — V2 Two-Way System.
"""

from typing import List, Optional
import math
from pydantic import BaseModel, Field, field_validator

from backend.config import settings


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    num_supported_labels: int = 0
    device: str
    feature_schema_version: str
    details: Optional[str] = None
    two_way_supported: bool = True
    sign_language_code: str = "dgs-DE"
    spoken_language_code: str = "de-DE"


class SignEntry(BaseModel):
    canonical_label: str
    german_text: str
    english_text: str
    is_model_supported: bool
    is_dictionary_only: bool
    description: Optional[str] = None
    cultural_notes: Optional[str] = None


class SignListResponse(BaseModel):
    total_signs: int
    model_supported_count: int
    dictionary_only_count: int
    signs: List[SignEntry]


class PredictionAlternative(BaseModel):
    canonical_label: str
    confidence: float
    german_text: str
    english_text: str


class PredictRequest(BaseModel):
    feature_schema_version: str = Field(..., description="Feature schema version, e.g. 'dgsense-v1'")
    sequence_length: int = Field(..., description="Number of temporal frames in the sequence, e.g. 30")
    feature_dim: int = Field(..., description="Number of landmark features per frame, e.g. 144")
    landmarks: List[List[float]] = Field(..., description="2D array of landmark coordinates [sequence_length, feature_dim]")
    request_id: Optional[str] = Field(None, description="Optional client-side request tracking ID")

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks(cls, v, info):
        if not v or not isinstance(v, list):
            raise ValueError("Landmarks must be a non-empty list of frames.")

        for frame_idx, frame in enumerate(v):
            if not isinstance(frame, list):
                raise ValueError(f"Frame {frame_idx} must be a list of float values.")
            for val_idx, val in enumerate(frame):
                if val is None or not isinstance(val, (int, float)):
                    raise ValueError(f"Frame {frame_idx}, value {val_idx} must be a number.")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Frame {frame_idx}, value {val_idx} contains NaN or Infinite value.")
        return v


class PredictResponse(BaseModel):
    status: str = Field(..., description="'ok' when confident, 'uncertain' when below threshold")
    prediction: Optional[str] = Field(None, description="Canonical sign label")
    confidence: float = Field(..., description="Model softmax confidence between 0.0 and 1.0")
    german_text: Optional[str] = Field(None, description="German translation")
    english_text: Optional[str] = Field(None, description="English translation")
    is_supported: bool = Field(True, description="Whether predicted sign is model-supported")
    model_version: str = Field(..., description="Version of the loaded model checkpoint")
    top_alternatives: List[PredictionAlternative] = Field(default_factory=list)
    request_id: Optional[str] = None
    note: Optional[str] = None


class ComposeSentenceRequest(BaseModel):
    sign_sequence: List[str] = Field(..., description="Ordered list of canonical sign labels")


class ComposeSentenceResponse(BaseModel):
    gloss_sequence: List[str]
    german_sentence: str
    english_sentence: str
    composition_type: str = Field(
        ...,
        description="'verified_phrase' for exact idiom/template match, or 'approximate_composition' for concatenated sign glosses"
    )
    is_continuous_translation: bool = Field(
        False,
        description="False for template-based individual sign composition; True only when a genuine continuous translation model is used"
    )
    notes: Optional[str] = None


# --- Reverse Translation Schemas (Direction B: German -> DGS) ---

class SignClipItem(BaseModel):
    clip_id: str
    canonical_meaning: str
    german_phrase: str
    english_meaning: Optional[str] = None
    source_attribution: str
    licence_status: str
    video_url: Optional[str] = None
    local_clip_path: Optional[str] = None
    sequence_order: List[str] = Field(default_factory=list)
    verification_notes: Optional[str] = None
    is_verified: bool = True


class ReverseTranslateRequest(BaseModel):
    text: str = Field(..., description="German input text to translate into DGS sign sequence")
    source_lang: str = Field("de", description="Source spoken language code, default 'de'")


class ReverseTranslateResponse(BaseModel):
    original_text: str
    normalized_text: str
    match_type: str = Field(
        ...,
        description="'exact_phrase_match' | 'compositional_sequence' | 'word_level_demonstration' | 'unsupported'"
    )
    dgs_gloss_sequence: List[str] = Field(default_factory=list)
    clips: List[SignClipItem] = Field(default_factory=list)
    explanation: str
    is_continuous_translation: bool = False
    unsupported_message: Optional[str] = None


class ReversePhraseListResponse(BaseModel):
    total_phrases: int
    phrases: List[SignClipItem]


# --- Dataset Manifest & Readiness Schemas ---

class DatasetManifestItem(BaseModel):
    phrase_id: str
    category: str
    canonical_meaning: str
    german_text: str
    english_text: str
    source_gloss: str
    source_url: Optional[str] = None
    recording_id: Optional[str] = None
    signer_id: Optional[str] = None
    clip_path: Optional[str] = None
    annotation_status: str
    verification_status: str
    licence_notes: str
    model_support_status: str


class DatasetManifestResponse(BaseModel):
    corpus_name: str
    version: str
    language_code: str
    total_items: int
    verified_research_clips: int
    data_required_count: int
    items: List[DatasetManifestItem]


class DatasetReadinessResponse(BaseModel):
    total_target_concepts: int
    ready_count: int
    data_required_count: int
    readiness_percentage: float
    missing_concepts: List[str]
    download_limit_mb: int
    reference_sources: List[dict]# --- Multi-Language Config Schema ---
class LanguageConfigResponse(BaseModel):
    sign_language_code: str
    spoken_language_code: str
    secondary_spoken_lang: str
    feature_schema_version: str
    dataset_limit_mb: int
    gloss_convention: str


# --- Persistence Schemas (SQLite-backed history) ---

class RecognitionEventOut(BaseModel):
    id: int
    created_at: str
    canonical_label: str
    german_text: Optional[str] = None
    english_text: Optional[str] = None
    confidence: float
    status: str
    request_id: Optional[str] = None


class RecognitionHistoryResponse(BaseModel):
    total: int
    items: List[RecognitionEventOut]


class ConversationMessageIn(BaseModel):
    speaker: str = Field(..., description="'dgs_user' or 'hearing_user'")
    speaker_name: Optional[str] = None
    text: str
    translation: Optional[str] = None
    glosses: List[str] = Field(default_factory=list)
    match_type: Optional[str] = None
    clips: List[dict] = Field(default_factory=list)


class ConversationMessageOut(BaseModel):
    id: int
    created_at: str
    speaker: str
    speaker_name: Optional[str] = None
    text: str
    translation: Optional[str] = None
    glosses: List[str] = Field(default_factory=list)
    match_type: Optional[str] = None
    clips: List[dict] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    total: int
    messages: List[ConversationMessageOut]


class PracticeAttemptIn(BaseModel):
    target_sign: str
    recognized_sign: Optional[str] = None
    correct: bool
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class PracticeAttemptOut(BaseModel):
    id: int
    created_at: str
    target_sign: str
    recognized_sign: Optional[str] = None
    correct: int
    confidence: Optional[float] = None


class PracticeHistoryResponse(BaseModel):
    total: int
    correct: int
    accuracy: float
    attempts: List[PracticeAttemptOut]


class RecordedSampleIn(BaseModel):
    label: str = Field(..., min_length=1, description="Canonical sign label")
    frames: List[List[float]] = Field(..., description="30 x 144 landmark sequence")
    signer_id: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, v, info):
        if len(v) != settings.SEQUENCE_LENGTH:
            raise ValueError(
                f"Sample must contain exactly {settings.SEQUENCE_LENGTH} frames, got {len(v)}."
            )
        for fi, frame in enumerate(v):
            if len(frame) != settings.FEATURE_DIM:
                raise ValueError(
                    f"Frame {fi} must have {settings.FEATURE_DIM} features, got {len(frame)}."
                )
            for val in frame:
                if val is None or not isinstance(val, (int, float)):
                    raise ValueError(f"Frame {fi} contains a non-numeric value.")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Frame {fi} contains NaN or Infinite value.")
        return v


class RecordedSampleSummary(BaseModel):
    id: int
    created_at: str
    label: str
    signer_id: Optional[str] = None
    notes: Optional[str] = None
    frame_count: int
    frames: Optional[List[List[float]]] = None


class RecordedSampleListResponse(BaseModel):
    total: int
    samples: List[RecordedSampleSummary]


class DeleteResponse(BaseModel):
    deleted: int
