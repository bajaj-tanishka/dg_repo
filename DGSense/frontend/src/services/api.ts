
import {
  HealthResponse,
  SignListResponse,
  PredictResponse,
  ComposeSentenceResponse,
  ModelInfoResponse,
  ReverseTranslateResponse,
  ReversePhraseListResponse,
  DatasetManifestResponse,
  DatasetReadinessResponse,
  LanguageConfigResponse,
  RecognitionHistoryResponse,
  ConversationListResponse,
  ConversationMessageOut,
  PracticeAttemptIn,
  PracticeAttemptOut,
  PracticeHistoryResponse,
  RecordedSampleListResponse,
  RecordedSampleSummary,
} from '../types/signs';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export const api = {
  async getHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE}/health`);

      if (!res.ok) {
        throw new Error(`Health check failed: ${res.statusText}`);
      }

      return await res.json();
    } catch (err) {
      return {
        status: 'degraded',
        model_loaded: false,
        num_supported_labels: 0,
        device: 'unknown',
        feature_schema_version: 'dgsense-v1',
        details:
          'Backend unreachable. Please verify that the FastAPI backend server is running.',
        two_way_supported: true,
        sign_language_code: 'dgs-DE',
        spoken_language_code: 'de-DE',
      };
    }
  },

  async getSignList(): Promise<SignListResponse> {
    const res = await fetch(`${API_BASE}/api/signs`);

    if (!res.ok) {
      throw new Error(`Failed to load signs: ${res.statusText}`);
    }

    return await res.json();
  },

  // --- Sign-to-Text: Webcam Recognition ---

  async predictSign(
    landmarks: number[][],
    requestId?: string
  ): Promise<PredictResponse> {
    // Validate the expected 30-frame sequence.
    if (landmarks.length !== 30) {
      throw new Error(
        `Expected 30 frames, but received ${landmarks.length}.`
      );
    }

    // Every frame must contain exactly 144 numeric features.
    const invalidFrame = landmarks.findIndex(
      (frame) =>
        !Array.isArray(frame) ||
        frame.length !== 144 ||
        !frame.every(
          (value) => typeof value === 'number' && Number.isFinite(value)
        )
    );

    if (invalidFrame !== -1) {
      throw new Error(
        `Frame ${invalidFrame + 1} must contain exactly 144 finite numeric features.`
      );
    }

    // Get the schema version from the backend.
    const health = await this.getHealth();

    if (!health.model_loaded) {
      throw new Error(
        'The recognition model is not loaded. Check the backend status.'
      );
    }

    const payload = {
      feature_schema_version: health.feature_schema_version,
      sequence_length: 30,
      feature_dim: 144,
      landmarks,
      request_id: requestId,
    };

    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res
        .json()
        .catch(() => ({ detail: res.statusText }));

      const detail =
        typeof errData.detail === 'string'
          ? errData.detail
          : `Prediction failed (${res.status})`;

      throw new Error(detail);
    }

    return await res.json();
  },

  // --- Sentence Composition ---

  async composeSentence(
    signs: string[]
  ): Promise<ComposeSentenceResponse> {
    const res = await fetch(`${API_BASE}/api/compose-sentence`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sign_sequence: signs,
      }),
    });

    if (!res.ok) {
      throw new Error(
        `Sentence composition error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  // --- Text-to-Sign: Reverse Translation ---

  async reverseTranslate(
    text: string
  ): Promise<ReverseTranslateResponse> {
    const res = await fetch(`${API_BASE}/api/reverse-translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        source_lang: 'de',
      }),
    });

    if (!res.ok) {
      throw new Error(
        `Reverse translation error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  async getReversePhrases(): Promise<ReversePhraseListResponse> {
    const res = await fetch(
      `${API_BASE}/api/reverse-translate/phrases`
    );

    if (!res.ok) {
      throw new Error(
        `Failed to load reverse phrases: ${res.statusText}`
      );
    }

    return await res.json();
  },

  // --- Dataset ---

  async getDatasetManifest(): Promise<DatasetManifestResponse> {
    const res = await fetch(`${API_BASE}/api/dataset/manifest`);

    if (!res.ok) {
      throw new Error(
        `Failed to load dataset manifest: ${res.statusText}`
      );
    }

    return await res.json();
  },

  async getDatasetReadiness(): Promise<DatasetReadinessResponse> {
    const res = await fetch(`${API_BASE}/api/dataset/readiness`);

    if (!res.ok) {
      throw new Error(
        `Failed to load dataset readiness: ${res.statusText}`
      );
    }

    return await res.json();
  },

  // --- Model Information ---

  async getModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetch(`${API_BASE}/api/model/info`);

    if (!res.ok) {
      throw new Error(`Model info error: ${res.statusText}`);
    }

    return await res.json();
  },

  // --- Recognition History ---

  async getHistory(
    limit = 50
  ): Promise<RecognitionHistoryResponse> {
    const res = await fetch(
      `${API_BASE}/api/history?limit=${limit}`
    );

    if (!res.ok) {
      throw new Error(`History fetch error: ${res.statusText}`);
    }

    return await res.json();
  },

  async clearHistory(): Promise<{ deleted: number }> {
    const res = await fetch(`${API_BASE}/api/history`, {
      method: 'DELETE',
    });

    if (!res.ok) {
      throw new Error(`History clear error: ${res.statusText}`);
    }

    return await res.json();
  },

  // --- Conversation Transcripts ---

  async getConversations(
    limit = 500
  ): Promise<ConversationListResponse> {
    const res = await fetch(
      `${API_BASE}/api/conversations?limit=${limit}`
    );

    if (!res.ok) {
      throw new Error(
        `Conversation fetch error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  async addConversation(
    msg: Omit<ConversationMessageOut, 'id' | 'created_at'>
  ): Promise<ConversationMessageOut> {
    const res = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(msg),
    });

    if (!res.ok) {
      throw new Error(
        `Conversation append error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  async clearConversations(): Promise<{ deleted: number }> {
    const res = await fetch(`${API_BASE}/api/conversations`, {
      method: 'DELETE',
    });

    if (!res.ok) {
      throw new Error(
        `Conversation clear error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  // --- Practice Results ---

  async getPracticeAttempts(
    limit = 100
  ): Promise<PracticeHistoryResponse> {
    const res = await fetch(
      `${API_BASE}/api/practice/attempts?limit=${limit}`
    );

    if (!res.ok) {
      throw new Error(
        `Practice history error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  async addPracticeAttempt(
    attempt: PracticeAttemptIn
  ): Promise<PracticeAttemptOut> {
    const res = await fetch(`${API_BASE}/api/practice/attempts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(attempt),
    });

    if (!res.ok) {
      throw new Error(
        `Practice append error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  // --- Recorded Samples ---

  async getSamples(
    includeFrames = false
  ): Promise<RecordedSampleListResponse> {
    const res = await fetch(
      `${API_BASE}/api/samples?include_frames=${includeFrames}`
    );

    if (!res.ok) {
      throw new Error(
        `Samples fetch error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  async uploadSample(payload: {
    label: string;
    frames: number[][];
    signer_id?: string;
    notes?: string;
  }): Promise<RecordedSampleSummary> {
    const res = await fetch(`${API_BASE}/api/samples`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res
        .json()
        .catch(() => ({ detail: res.statusText }));

      const detail =
        typeof errData.detail === 'string'
          ? errData.detail
          : `Sample upload failed (${res.status})`;

      throw new Error(detail);
    }

    return await res.json();
  },

  async deleteSample(
    sampleId: number
  ): Promise<{ deleted: number }> {
    const res = await fetch(
      `${API_BASE}/api/samples/${sampleId}`,
      {
        method: 'DELETE',
      }
    );

    if (!res.ok) {
      throw new Error(
        `Sample delete error: ${res.statusText}`
      );
    }

    return await res.json();
  },

  // --- Language Configuration ---

  async getLanguageConfig(): Promise<LanguageConfigResponse> {
    const res = await fetch(`${API_BASE}/api/languages`);

    if (!res.ok) {
      throw new Error(
        `Language config error: ${res.statusText}`
      );
    }

    return await res.json();
  },
};
