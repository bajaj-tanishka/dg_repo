export interface SignEntry {
  canonical_label: string;
  german_text: string;
  english_text: string;
  is_model_supported: boolean;
  is_dictionary_only: boolean;
  description?: string;
  cultural_notes?: string;
}

export interface SignListResponse {
  total_signs: number;
  model_supported_count: number;
  dictionary_only_count: number;
  signs: SignEntry[];
}

export interface PredictionAlternative {
  canonical_label: string;
  confidence: number;
  german_text: string;
  english_text: string;
}

export interface PredictResponse {
  status: 'ok' | 'uncertain';
  prediction: string | null;
  confidence: number;
  german_text: string | null;
  english_text: string | null;
  is_supported: boolean;
  model_version: string;
  top_alternatives: PredictionAlternative[];
  request_id?: string;
  note?: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  model_loaded: boolean;
  model_version?: string | null;
  num_supported_labels: number;
  device: string;
  feature_schema_version: string;
  details?: string;
  two_way_supported?: boolean;
  sign_language_code?: string;
  spoken_language_code?: string;
}

export interface ComposeSentenceResponse {
  gloss_sequence: string[];
  german_sentence: string;
  english_sentence: string;
  composition_type: 'verified_phrase' | 'approximate_composition' | 'empty';
  is_continuous_translation: boolean;
  notes?: string;
}

export interface ModelInfoResponse {
  model_loaded: boolean;
  model_version: string;
  model_config: Record<string, any>;
  feature_schema: Record<string, any>;
  supported_classes: string[];
  total_vocabulary_in_dict: number;
  min_confidence_threshold: number;
  device: string;
  two_way_translation?: boolean;
  reverse_clips_available?: number;
}

export interface RecognitionHistoryItem {
  id: string;
  timestamp: string;
  canonical_label: string;
  german_text: string;
  english_text: string;
  confidence: number;
}

// --- V2 Two-Way Schemas ---

export interface SignClipItem {
  clip_id: string;
  canonical_meaning: string;
  german_phrase: string;
  english_meaning?: string;
  source_attribution: string;
  licence_status: string;
  video_url?: string | null;
  local_clip_path?: string | null;
  sequence_order: string[];
  verification_notes?: string | null;
  is_verified: boolean;
}

export interface ReverseTranslateResponse {
  original_text: string;
  normalized_text: string;
  match_type: 'exact_phrase_match' | 'compositional_sequence' | 'word_level_demonstration' | 'unsupported';
  dgs_gloss_sequence: string[];
  clips: SignClipItem[];
  explanation: string;
  is_continuous_translation: boolean;
  unsupported_message?: string | null;
}

export interface ReversePhraseListResponse {
  total_phrases: number;
  phrases: SignClipItem[];
}

export interface DatasetManifestItem {
  phrase_id: string;
  category: string;
  canonical_meaning: string;
  german_text: string;
  english_text: string;
  source_gloss: string;
  source_url?: string | null;
  recording_id?: string | null;
  signer_id?: string | null;
  clip_path?: string | null;
  annotation_status: string;
  verification_status: string;
  licence_notes: string;
  model_support_status: string;
}

export interface DatasetManifestResponse {
  corpus_name: string;
  version: string;
  language_code: string;
  total_items: number;
  verified_research_clips: number;
  data_required_count: number;
  items: DatasetManifestItem[];
}

export interface DatasetReadinessResponse {
  total_target_concepts: number;
  ready_count: number;
  data_required_count: number;
  readiness_percentage: number;
  missing_concepts: string[];
  download_limit_mb: number;
  reference_sources: Array<{
    name: string;
    institution?: string;
    url: string;
    licence: string;
    licence_notes: string;
  }>;
}

export interface LanguageConfigResponse {
  sign_language_code: string;
  spoken_language_code: string;
  secondary_spoken_lang: string;
  feature_schema_version: string;
  dataset_limit_mb: number;
  gloss_convention: string;
}

export interface ConversationMessage {
  id: string;
  speaker: 'dgs_user' | 'hearing_user';
  speaker_name: string;
  text: string;
  translation?: string;
  glosses?: string[];
  match_type?: string;
  clips?: SignClipItem[];
  timestamp: string;
}

// --- V3 Persistence Schemas (SQLite-backed) ---

export interface RecognitionEvent {
  id: number;
  created_at: string;
  canonical_label: string;
  german_text?: string | null;
  english_text?: string | null;
  confidence: number;
  status: string;
  request_id?: string | null;
}

export interface RecognitionHistoryResponse {
  total: number;
  items: RecognitionEvent[];
}

export interface ConversationMessageOut {
  id: number;
  created_at: string;
  speaker: 'dgs_user' | 'hearing_user';
  speaker_name?: string | null;
  text: string;
  translation?: string | null;
  glosses: string[];
  match_type?: string | null;
  clips: SignClipItem[];
}

export interface ConversationListResponse {
  total: number;
  messages: ConversationMessageOut[];
}

export interface PracticeAttemptIn {
  target_sign: string;
  recognized_sign?: string | null;
  correct: boolean;
  confidence?: number | null;
}

export interface PracticeAttemptOut {
  id: number;
  created_at: string;
  target_sign: string;
  recognized_sign?: string | null;
  correct: number;
  confidence?: number | null;
}

export interface PracticeHistoryResponse {
  total: number;
  correct: number;
  accuracy: number;
  attempts: PracticeAttemptOut[];
}

export interface RecordedSampleSummary {
  id: number;
  created_at: string;
  label: string;
  signer_id?: string | null;
  notes?: string | null;
  frame_count: number;
  frames?: number[][];
}

export interface RecordedSampleListResponse {
  total: number;
  samples: RecordedSampleSummary[];
}
