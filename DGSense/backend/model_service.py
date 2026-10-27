"""
DGSense Temporal Model Service & Two-Way Translation Pipeline — V2
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from backend.config import settings
from backend.schemas import (
    PredictResponse,
    PredictionAlternative,
    ComposeSentenceResponse,
    SignEntry,
    ReverseTranslateResponse,
    SignClipItem,
    DatasetManifestResponse,
    DatasetManifestItem,
    DatasetReadinessResponse,
    LanguageConfigResponse
)


class DGSenseTemporalGRU(nn.Module):
    """
    Lightweight bidirectional GRU temporal neural network for German Sign Language
    isolated sequence classification.
    """
    def __init__(
        self,
        input_dim: int = 144,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Temporal feature encoding
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        direction_factor = 2 if bidirectional else 1
        gru_out_dim = hidden_dim * direction_factor

        # Temporal attention pooling (only when the checkpoint expects it, so older
        # checkpoints without attn weights still load cleanly).
        if use_attention:
            self.attn = nn.Linear(gru_out_dim, 1)

        self.layer_norm = nn.LayerNorm(gru_out_dim)
        self.dropout = nn.Dropout(dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        gru_out, _ = self.gru(x)
        if self.use_attention:
            weights = torch.softmax(self.attn(gru_out).squeeze(-1), dim=1)
            pooled = (gru_out * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = torch.mean(gru_out, dim=1)
        normed = self.layer_norm(pooled)
        dropped = self.dropout(normed)
        logits = self.classifier(dropped)
        return logits


def normalize_frame(v: np.ndarray) -> np.ndarray:
    """Wrist-origin, hand-scale normalization (dgsense-v1).

    Removes camera framing / distance differences so live webcam frames live in the same
    coordinate space as the PHOENIX training clips. MUST stay in sync with the identical
    function in notebooks/DGSense_Training.ipynb.
    """
    out = np.array(v, dtype=np.float32, copy=True)
    for base in (0, 63):  # left hand slot, right hand slot
        hand = out[base:base + 63].reshape(21, 3)
        if np.abs(hand).sum() < 1e-6:
            continue  # hand not detected in this slot
        origin = hand[0].copy()
        scale = float(np.linalg.norm(hand[9] - hand[0]))  # wrist -> middle-finger MCP
        if scale < 1e-6:
            continue
        out[base:base + 63] = ((hand - origin) / scale).reshape(-1)
        ps = 138 if base == 0 else 141  # wrist pose slots follow their hand
        out[ps:ps + 3] = (out[ps:ps + 3] - origin) / scale
    return out


def to_model_input(seq: np.ndarray) -> np.ndarray:
    """144 raw features -> 288 velocity-stacked model input (x_t, x_t - x_{t-1}).

    Applied ONLY when the loaded checkpoint expects input_dim=288 (use_delta models).
    MUST stay in sync with notebooks/DGSense_Training.ipynb::to_model_input.
    """
    if seq.shape[-1] == 2 * settings.FEATURE_DIM:
        return seq.astype(np.float32)
    delta = np.zeros_like(seq, dtype=np.float32)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq.astype(np.float32), delta], axis=-1)


class ModelService:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or settings.ARTIFACTS_DIR
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu"
        )
        self.model: Optional[DGSenseTemporalGRU] = None
        self.labels_data: List[dict] = []
        self.label_map: Dict[int, str] = {}
        self.canonical_to_info: Dict[str, dict] = {}
        self.feature_schema: dict = {}
        self.model_config: dict = {}
        self.is_model_loaded: bool = False
        self.model_version: str = "unavailable"

        # V2: Reverse translation clips & everyday manifest
        self.sign_clips: Dict[str, dict] = {}
        self.everyday_manifest: dict = {}

        # Load schemas, metadata & models
        self._load_metadata()
        self.load_checkpoint()

    def _load_metadata(self):
        """Load labels, feature schema, model configuration, reverse clips, and everyday manifest."""
        labels_path = self.artifacts_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                self.labels_data = content.get("classes", [])
                for item in self.labels_data:
                    c_id = item.get("id")
                    c_label = item.get("canonical_label")
                    if c_id is not None and item.get("is_model_supported", False):
                        self.label_map[c_id] = c_label
                    self.canonical_to_info[c_label] = item

        schema_path = self.artifacts_dir / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.feature_schema = json.load(f)

        config_path = self.artifacts_dir / "model_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)
                self.model_version = self.model_config.get("model_version", "1.0.0")

        clips_path = self.artifacts_dir / "sign_clips.json"
        if clips_path.exists():
            with open(clips_path, "r", encoding="utf-8") as f:
                clips_content = json.load(f)
                for clip in clips_content.get("clips", []):
                    c_meaning = clip.get("canonical_meaning")
                    if c_meaning:
                        self.sign_clips[c_meaning] = clip

        manifest_path = self.artifacts_dir / "everyday_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.everyday_manifest = json.load(f)

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> bool:
        """
        Load trained PyTorch weights from model.pt.

        Supports plain state dictionaries and common wrapped checkpoints.
        Prints detailed diagnostics when loading fails.
        """
        ckpt_path = (
            Path(checkpoint_path)
            if checkpoint_path
            else self.artifacts_dir / "model.pt"
        )

        self.model = None
        self.is_model_loaded = False

        if not ckpt_path.is_file():
            print(
                f"[ModelService] Checkpoint not found: "
                f"{ckpt_path.resolve()}"
            )
            return False

        try:
            arch = self.model_config.get("architecture", {})
            configured_classes = arch.get("num_classes")
            label_count = len(self.label_map)

            if label_count == 0:
                raise ValueError(
                    "No supported labels loaded. Check labels.json "
                    "and the is_model_supported field."
                )

            num_classes = int(
                configured_classes
                if configured_classes is not None
                else label_count
            )

            if label_count != num_classes:
                raise ValueError(
                    f"Label count mismatch: labels.json contains "
                    f"{label_count} supported labels, but the model "
                    f"configuration specifies {num_classes} classes."
                )

            expected_ids = list(range(num_classes))
            actual_ids = sorted(int(key) for key in self.label_map)

            if actual_ids != expected_ids:
                raise ValueError(
                    f"Label IDs must be contiguous from 0 to "
                    f"{num_classes - 1}. Found: {actual_ids}"
                )

            input_dim = int(
                arch.get("input_dim", settings.FEATURE_DIM)
            )
            hidden_dim = int(arch.get("hidden_dim", 128))
            num_layers = int(arch.get("num_layers", 2))
            bidirectional = bool(arch.get("bidirectional", True))
            dropout = float(arch.get("dropout", 0.3))
            use_attention = bool(arch.get("use_attention", False))

            print(
                f"[ModelService] Loading checkpoint: "
                f"{ckpt_path.resolve()}"
            )
            print(
                f"[ModelService] Classes: {num_classes}, "
                f"Input dimension: {input_dim}, "
                f"Hidden dimension: {hidden_dim}, "
                f"Attention: {use_attention}"
            )
            print(
                f"[ModelService] Device: {self.device}"
            )

            model = DGSenseTemporalGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bidirectional,
                use_attention=use_attention,
            )

            # Load the checkpoint safely when supported by this
            # installed version of PyTorch.
            try:
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                # Compatibility with older PyTorch versions.
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=self.device,
                )

            if not isinstance(checkpoint, dict):
                raise TypeError(
                    "Unsupported checkpoint format: expected a "
                    "PyTorch state dictionary."
                )

            # Support common checkpoint wrappers.
            if (
                "state_dict" in checkpoint
                and isinstance(checkpoint["state_dict"], dict)
            ):
                state_dict = checkpoint["state_dict"]
            elif (
                "model_state_dict" in checkpoint
                and isinstance(checkpoint["model_state_dict"], dict)
            ):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            if not state_dict:
                raise ValueError("The checkpoint state dictionary is empty.")

            # Remove the DataParallel prefix if present.
            if all(
                str(key).startswith("module.")
                for key in state_dict.keys()
            ):
                state_dict = {
                    str(key)[7:]: value
                    for key, value in state_dict.items()
                }

            # Strict loading ensures architecture mismatches are not
            # silently ignored.
            model.load_state_dict(state_dict, strict=True)

            model.to(self.device)
            model.eval()

            self.model = model
            self.is_model_loaded = True

            print(
                f"[ModelService] Model loaded successfully. "
                f"Version: {self.model_version}; "
                f"Classes: {num_classes}; "
                f"Device: {self.device}"
            )

            return True

        except Exception as e:
            self.model = None
            self.is_model_loaded = False

            print(
                f"[ModelService] Error loading checkpoint "
                f"'{ckpt_path}': {type(e).__name__}: {e}"
            )

            return False

    def predict(
        self,
        landmarks: List[List[float]],
        feature_schema_version: str,
        sequence_length: int,
        feature_dim: int,
        request_id: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> PredictResponse:
        """Evaluates a temporal sequence of landmarks (Direction A)."""
        if not self.is_model_loaded or self.model is None:
            raise RuntimeError(
                "Model checkpoint is not loaded. Train the model using the notebook or "
                "place 'model.pt' in the backend/artifacts directory."
            )

        expected_version = settings.FEATURE_SCHEMA_VERSION
        if feature_schema_version != expected_version:
            raise ValueError(
                f"Feature schema version '{feature_schema_version}' incompatible. Expected '{expected_version}'."
            )

        if sequence_length != settings.SEQUENCE_LENGTH or feature_dim != settings.FEATURE_DIM:
            raise ValueError(
                f"Invalid dimensions: got ({sequence_length}, {feature_dim}), "
                f"expected ({settings.SEQUENCE_LENGTH}, {settings.FEATURE_DIM})."
            )

        arr = np.array(landmarks, dtype=np.float32)
        if arr.shape != (settings.SEQUENCE_LENGTH, settings.FEATURE_DIM):
            raise ValueError(
                f"Array shape mismatch: received {arr.shape}, expected ({settings.SEQUENCE_LENGTH}, {settings.FEATURE_DIM})"
            )

        # Wrist-origin / hand-scale normalization — identical to the one applied to
        # PHOENIX training clips in notebooks/DGSense_Training.ipynb (keep both in sync).
        arr = np.stack([normalize_frame(frame) for frame in arr]).astype(np.float32)

        # Velocity stacking for checkpoints trained with use_delta (144 -> 288).
        if self.model.input_dim == 2 * settings.FEATURE_DIM and arr.shape[1] == settings.FEATURE_DIM:
            arr = to_model_input(arr)

        threshold = min_confidence if min_confidence is not None else settings.MIN_CONFIDENCE

        tensor_in = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probabilities)[::-1]
        top_idx = int(sorted_indices[0])
        top_confidence = float(probabilities[top_idx])

        top_label = self.label_map.get(top_idx, f"UNKNOWN_{top_idx}")
        info = self.canonical_to_info.get(top_label, {})
        german_text = info.get("german_text", top_label)
        english_text = info.get("english_text", top_label)

        alternatives: List[PredictionAlternative] = []
        for alt_idx in sorted_indices[1:4]:
            alt_id = int(alt_idx)
            alt_conf = float(probabilities[alt_id])
            alt_label = self.label_map.get(alt_id, f"CLASS_{alt_id}")
            alt_info = self.canonical_to_info.get(alt_label, {})
            alternatives.append(
                PredictionAlternative(
                    canonical_label=alt_label,
                    confidence=round(alt_conf, 4),
                    german_text=alt_info.get("german_text", alt_label),
                    english_text=alt_info.get("english_text", alt_label)
                )
            )

        if top_confidence < threshold:
            return PredictResponse(
                status="uncertain",
                prediction=None,
                confidence=round(top_confidence, 4),
                german_text=None,
                english_text=None,
                is_supported=True,
                model_version=self.model_version,
                top_alternatives=alternatives,
                request_id=request_id,
                note=f"Confidence {top_confidence:.1%} is below required threshold {threshold:.1%}."
            )

        return PredictResponse(
            status="ok",
            prediction=top_label,
            confidence=round(top_confidence, 4),
            german_text=german_text,
            english_text=english_text,
            is_supported=True,
            model_version=self.model_version,
            top_alternatives=alternatives,
            request_id=request_id,
            note=None
        )

    def get_supported_signs(self) -> List[SignEntry]:
        """Returns vocabulary with clear distinction between model-supported and dictionary-only entries."""
        signs: List[SignEntry] = []
        for item in self.labels_data:
            c_label = item.get("canonical_label")
            is_model_supported = item.get("is_model_supported", False)
            signs.append(
                SignEntry(
                    canonical_label=c_label,
                    german_text=item.get("german_text", c_label),
                    english_text=item.get("english_text", c_label),
                    is_model_supported=is_model_supported,
                    is_dictionary_only=item.get("is_dictionary_only", not is_model_supported),
                    description=item.get("description"),
                    cultural_notes=item.get("cultural_notes")
                )
            )
        return signs

    def compose_sentence(self, sign_sequence: List[str]) -> ComposeSentenceResponse:
        """Combines recognized signs into German and English text."""
        if not sign_sequence:
            return ComposeSentenceResponse(
                gloss_sequence=[],
                german_sentence="",
                english_sentence="",
                composition_type="empty",
                is_continuous_translation=False,
                notes="No signs provided in sequence."
            )

        normalized_seq = [s.strip().upper() for s in sign_sequence if s.strip()]

        phrase_key = "+".join(normalized_seq)
        verified_phrases = {
            "HELLO+HOW_ARE_YOU": {
                "de": "Hallo! Wie geht es dir?",
                "en": "Hello! How are you?"
            },
            "PLEASE+HELP": {
                "de": "Bitte hilf mir!",
                "en": "Please help me!"
            },
            "HELLO+GOOD_MORNING": {
                "de": "Hallo, guten Morgen!",
                "en": "Hello, good morning!"
            },
            "THANK_YOU+GOODBYE": {
                "de": "Vielen Dank und auf Wiedersehen!",
                "en": "Thank you and goodbye!"
            },
            "SORRY+HELP": {
                "de": "Entschuldigung, kannst du mir helfen?",
                "en": "Excuse me, can you help me?"
            },
            "YES+THANK_YOU": {
                "de": "Ja, vielen Dank!",
                "en": "Yes, thank you very much!"
            },
            "NO+THANK_YOU": {
                "de": "Nein, danke.",
                "en": "No, thank you."
            }
        }

        if phrase_key in verified_phrases:
            match = verified_phrases[phrase_key]
            return ComposeSentenceResponse(
                gloss_sequence=normalized_seq,
                german_sentence=match["de"],
                english_sentence=match["en"],
                composition_type="verified_phrase",
                is_continuous_translation=False,
                notes="Matched a verified conversational DGS phrase template."
            )

        de_words = []
        en_words = []
        for sign in normalized_seq:
            info = self.canonical_to_info.get(sign, {})
            de_words.append(info.get("german_text", sign.capitalize()))
            en_words.append(info.get("english_text", sign.capitalize()))

        german_result = " ".join(de_words)
        english_result = " ".join(en_words)

        return ComposeSentenceResponse(
            gloss_sequence=normalized_seq,
            german_sentence=german_result,
            english_sentence=english_result,
            composition_type="approximate_composition",
            is_continuous_translation=False,
            notes="Approximate sign sequence composition. DGS grammar differs significantly from spoken German word order."
        )

    # --- V2: Reverse Translation Pipeline (Direction B: German -> DGS) ---

    def normalize_german_text(self, text: str) -> str:
        """Strips punctuation, trims whitespace, and standardizes lower-case."""
        clean = re.sub(r"[^\w\säöüßÄÖÜ]", " ", text)
        clean = " ".join(clean.split()).lower()
        return clean

    def reverse_translate(self, input_text: str) -> ReverseTranslateResponse:
        """
        Translates German text or speech transcript into verified DGS sign representations.
        Priority:
        1. Exact verified DGS phrase/sequence match.
        2. Verified compositional sequence.
        3. Word-level sign demonstration lookup (explicitly labeled).
        4. Unsupported phrase message state (no fake signs generated).
        """
        raw_text = input_text.strip()
        if not raw_text:
            return ReverseTranslateResponse(
                original_text=raw_text,
                normalized_text="",
                match_type="unsupported",
                dgs_gloss_sequence=[],
                clips=[],
                explanation="Input text is empty.",
                is_continuous_translation=False,
                unsupported_message="Bitte gib einen deutschen Satz oder ein Wort ein."
            )

        normalized = self.normalize_german_text(raw_text)

        # 1. Exact verified phrase dictionary
        phrase_mappings = {
            "hallo": ("HELLO", ["HALLO"], "Verified greeting sign HALLO."),
            "guten morgen": ("GOOD_MORNING", ["GUT", "MORGEN"], "Verified compound greeting GUT^MORGEN."),
            "guten tag": ("GOOD_DAY", ["GUT", "TAG"], "Verified polite greeting GUT^TAG."),
            "guten abend": ("GOOD_EVENING", ["GUT", "ABEND"], "Verified evening greeting GUT^ABEND."),
            "tschuess": ("BYE", ["TSCHÜSS"], "Verified informal parting sign."),
            "tschüss": ("BYE", ["TSCHÜSS"], "Verified informal parting sign."),
            "auf wiedersehen": ("GOODBYE", ["AUF-WIEDERSEHEN"], "Verified formal parting sign."),
            "danke": ("THANK_YOU", ["DANKE"], "Verified gratitude sign."),
            "vielen dank": ("THANK_YOU_VERY_MUCH", ["DANKE", "VIEL"], "Verified intensified gratitude sign."),
            "danke schoen": ("THANK_YOU_VERY_MUCH", ["DANKE", "VIEL"], "Verified gratitude expression."),
            "danke schön": ("THANK_YOU_VERY_MUCH", ["DANKE", "VIEL"], "Verified gratitude expression."),
            "bitte": ("PLEASE", ["BITTE"], "Verified request sign."),
            "entschuldigung": ("SORRY", ["ENTSCHULDIGUNG"], "Verified apology sign."),
            "ja": ("YES", ["JA"], "Verified affirmative sign."),
            "nein": ("NO", ["NEIN"], "Verified negation sign."),
            "ich verstehe nicht": ("NOT_UNDERSTAND", ["VERSTEHEN", "NICHT"], "Verified DGS phrase: VERSTEHEN NICHT."),
            "verstehe nicht": ("NOT_UNDERSTAND", ["VERSTEHEN", "NICHT"], "Verified DGS phrase: VERSTEHEN NICHT."),
            "kannst du mir helfen": ("CAN_HELP", ["DU", "HELFEN-MIR", "KÖNNEN"], "Verified directional question DU HELFEN-MIR KÖNNEN."),
            "kannst du mir helfen ": ("CAN_HELP", ["DU", "HELFEN-MIR", "KÖNNEN"], "Verified directional question."),
            "wie geht es dir": ("HOW_ARE_YOU", ["WIE-GEHT", "DIR"], "Verified conversational inquiry WIE-GEHT DIR."),
            "wie gehts": ("HOW_ARE_YOU", ["WIE-GEHT", "DIR"], "Verified conversational inquiry."),
            "komm her": ("COME_HERE", ["KOMMEN", "HIER"], "Verified directive KOMMEN HIER."),
            "lass uns gehen": ("LETS_GO", ["WIR", "GEHEN"], "Verified collective departure sign."),
            "warte bitte": ("WAIT_PLEASE", ["WARTEN", "BITTE"], "Verified polite request WARTEN BITTE."),
            "ich brauche hilfe": ("NEED_HELP", ["ICH", "BRAUCHEN", "HILFE"], "Verified expression of need ICH BRAUCHEN HILFE."),
            "hilfe": ("HELP", ["HILFE"], "Verified directional sign HILFE."),
            "stopp": ("STOP", ["STOPP"], "Verified decisive halt sign STOPP."),
            "stop": ("STOP", ["STOPP"], "Verified decisive halt sign STOPP.")
        }

        # Check exact phrase match
        if normalized in phrase_mappings:
            canonical, gloss_seq, expl = phrase_mappings[normalized]
            clip = self._build_clip_item(canonical)
            return ReverseTranslateResponse(
                original_text=raw_text,
                normalized_text=normalized,
                match_type="exact_phrase_match",
                dgs_gloss_sequence=gloss_seq,
                clips=[clip] if clip else [],
                explanation=expl,
                is_continuous_translation=False
            )

        # 2. Compositional sequence matching (multi-phrase combinations)
        compositional_patterns = [
            ("hallo wie geht es dir", ["HELLO", "HOW_ARE_YOU"], ["HALLO", "WIE-GEHT", "DIR"], "Compositional phrase: Hallo + Wie geht es dir?"),
            ("hallo guten morgen", ["HELLO", "GOOD_MORNING"], ["HALLO", "GUT", "MORGEN"], "Compositional greeting sequence."),
            ("ja danke", ["YES", "THANK_YOU"], ["JA", "DANKE"], "Polite affirmation sequence: Ja, danke."),
            ("nein danke", ["NO", "THANK_YOU"], ["NEIN", "DANKE"], "Polite refusal sequence: Nein, danke."),
            ("bitte hilfe", ["PLEASE", "HELP"], ["BITTE", "HILFE"], "Urgent request: Bitte, Hilfe!"),
            ("entschuldigung kannst du mir helfen", ["SORRY", "CAN_HELP"], ["ENTSCHULDIGUNG", "DU", "HELFEN-MIR", "KÖNNEN"], "Polite inquiry sequence.")
        ]

        for pattern, canons, glosses, expl in compositional_patterns:
            if pattern in normalized or normalized in pattern:
                clips = [c for c in [self._build_clip_item(can) for can in canons] if c is not None]
                return ReverseTranslateResponse(
                    original_text=raw_text,
                    normalized_text=normalized,
                    match_type="compositional_sequence",
                    dgs_gloss_sequence=glosses,
                    clips=clips,
                    explanation=f"{expl} Composed from verified DGS gesture elements.",
                    is_continuous_translation=False
                )

        # 3. Word-level sign demonstration fallback
        words = normalized.split()
        matched_clips = []
        matched_glosses = []
        for word in words:
            if word in phrase_mappings:
                can, glosses, _ = phrase_mappings[word]
                clip = self._build_clip_item(can)
                if clip:
                    matched_clips.append(clip)
                    matched_glosses.extend(glosses)

        if matched_clips:
            return ReverseTranslateResponse(
                original_text=raw_text,
                normalized_text=normalized,
                match_type="word_level_demonstration",
                dgs_gloss_sequence=matched_glosses,
                clips=matched_clips,
                explanation="Word-level sign demonstrations found. Note: Concatenating sign words does not represent full continuous DGS grammar.",
                is_continuous_translation=False
            )

        # 4. Informative unsupported-message state
        return ReverseTranslateResponse(
            original_text=raw_text,
            normalized_text=normalized,
            match_type="unsupported",
            dgs_gloss_sequence=[],
            clips=[],
            explanation="No verified DGS sign or phrase sequence is currently available for this sentence.",
            is_continuous_translation=False,
            unsupported_message=(
                f"Für den Satz \"{raw_text}\" liegt derzeit keine verifizierte DGS-Gebärdenfolge vor. "
                "Du kannst den Originaltext kopieren oder vorlesen lassen."
            )
        )

    def _build_clip_item(self, canonical_meaning: str) -> Optional[SignClipItem]:
        """Looks up or constructs a SignClipItem for a canonical meaning."""
        clip_data = self.sign_clips.get(canonical_meaning)
        if clip_data:
            return SignClipItem(
                clip_id=clip_data.get("sign_id", f"CLIP_{canonical_meaning}"),
                canonical_meaning=canonical_meaning,
                german_phrase=clip_data.get("german_phrase", canonical_meaning),
                english_meaning=clip_data.get("english_meaning"),
                source_attribution=clip_data.get("source_attribution", "DGS Corpus / Meine DGS"),
                licence_status=clip_data.get("licence_status", "Educational / Research"),
                video_url=clip_data.get("video_url"),
                local_clip_path=clip_data.get("local_clip_path"),
                sequence_order=clip_data.get("sequence_order", [canonical_meaning]),
                verification_notes=clip_data.get("verification_notes"),
                is_verified=clip_data.get("is_verified", True)
            )

        # Fallback to info in canonical_to_info
        info = self.canonical_to_info.get(canonical_meaning)
        if info:
            return SignClipItem(
                clip_id=f"CLIP_{canonical_meaning}",
                canonical_meaning=canonical_meaning,
                german_phrase=info.get("german_text", canonical_meaning),
                english_meaning=info.get("english_text"),
                source_attribution="DGSense Verified DGS Catalog",
                licence_status="Educational / Research",
                video_url=None,
                local_clip_path=None,
                sequence_order=[canonical_meaning],
                verification_notes=info.get("description"),
                is_verified=True
            )
        return None

    def get_all_reverse_phrases(self) -> List[SignClipItem]:
        """Returns catalog of all verified reverse translation phrases and clips."""
        items: List[SignClipItem] = []
        for canonical, clip_data in self.sign_clips.items():
            clip = self._build_clip_item(canonical)
            if clip:
                items.append(clip)
        return items

    def get_dataset_manifest(self) -> DatasetManifestResponse:
        """Returns target everyday vocabulary manifest."""
        items_raw = self.everyday_manifest.get("target_manifest", [])
        meta = self.everyday_manifest.get("dataset_metadata", {})
        items: List[DatasetManifestItem] = [DatasetManifestItem(**i) for i in items_raw]
        verified_count = sum(1 for i in items if i.verification_status == "verified_research_clip")
        data_req_count = sum(1 for i in items if i.verification_status == "data_required")

        return DatasetManifestResponse(
            corpus_name=meta.get("corpus_name", "DGSense Everyday Manifest"),
            version=meta.get("version", "2.0.0"),
            language_code=meta.get("language_code", "dgs-DE"),
            total_items=len(items),
            verified_research_clips=verified_count,
            data_required_count=data_req_count,
            items=items
        )

    def get_dataset_readiness(self) -> DatasetReadinessResponse:
        """Returns readiness audit metrics and data requirement flags."""
        manifest = self.get_dataset_manifest()
        missing = [i.canonical_meaning for i in manifest.items if i.verification_status == "data_required"]
        ready = manifest.total_items - len(missing)
        pct = (ready / manifest.total_items * 100.0) if manifest.total_items > 0 else 0.0

        ref_sources = self.everyday_manifest.get("dataset_metadata", {}).get("reference_sources", [])

        return DatasetReadinessResponse(
            total_target_concepts=manifest.total_items,
            ready_count=ready,
            data_required_count=len(missing),
            readiness_percentage=round(pct, 1),
            missing_concepts=missing,
            download_limit_mb=settings.DATASET_SIZE_LIMIT_MB,
            reference_sources=ref_sources
        )

    def get_language_config(self) -> LanguageConfigResponse:
        """Returns multi-language extensibility configuration."""
        return LanguageConfigResponse(
            sign_language_code=settings.SIGN_LANGUAGE_CODE,
            spoken_language_code=settings.SPOKEN_LANGUAGE_CODE,
            secondary_spoken_lang=settings.SECONDARY_SPOKEN_LANG,
            feature_schema_version=settings.FEATURE_SCHEMA_VERSION,
            dataset_limit_mb=settings.DATASET_SIZE_LIMIT_MB,
            gloss_convention="IDGS Standard Gloss Notation (Uppercase German Lemma)"
        )


# Singleton service instance
model_service = ModelService()
