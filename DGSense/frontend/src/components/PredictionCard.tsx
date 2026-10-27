import React from 'react';
import { Plus, AlertCircle, CheckCircle2 } from 'lucide-react';
import { PredictResponse } from '../types/signs';

interface PredictionCardProps {
  prediction: PredictResponse | null;
  isPredicting: boolean;
  modelReady: boolean;
  onAddToSentence: (sign: string) => void;
  autoAddEnabled: boolean;
  setAutoAddEnabled: (enabled: boolean) => void;
  threshold: number;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  isPredicting,
  modelReady,
  onAddToSentence,
  autoAddEnabled,
  setAutoAddEnabled,
  threshold,
}) => {
  if (!modelReady) {
    return (
      <div className="prediction-box">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--color-warning)' }}>
          <AlertCircle size={24} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Inspection Mode Active</h3>
        </div>
        <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
          Real-time hand and body landmark tracking is running in your browser. The backend is awaiting a trained checkpoint.
        </p>
        <div
          style={{
            marginTop: '0.75rem',
            padding: '0.6rem 0.85rem',
            background: 'rgba(255, 255, 255, 0.7)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--color-border)',
            fontSize: '0.8rem',
            color: 'var(--color-text-muted)',
          }}
        >
          Train with <code>notebooks/DGSense_Training.ipynb</code> in Google Colab and place <code>model.pt</code> into <code>backend/artifacts/</code> to activate live recognition.
        </div>
      </div>
    );
  }

  const isConfident = prediction && prediction.status === 'ok' && prediction.prediction;
  const confidencePercent = prediction ? Math.round(prediction.confidence * 100) : 0;

  return (
    <div className="prediction-box">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
            Live Recognition
          </span>
          <div className="prediction-canonical">
            {isConfident ? (
              prediction.prediction
            ) : prediction?.status === 'uncertain' ? (
              <span style={{ color: 'var(--color-text-muted)', fontSize: '1.4rem' }}>
                Uncertain Sign
              </span>
            ) : isPredicting ? (
              <span style={{ color: 'var(--color-text-muted)', fontSize: '1.4rem' }}>
                Evaluating buffer...
              </span>
            ) : (
              <span style={{ color: 'var(--color-text-muted)', fontSize: '1.4rem' }}>
                Perform a sign
              </span>
            )}
          </div>
        </div>

        {isConfident && (
          <button
            className="btn btn-primary btn-sm"
            onClick={() => onAddToSentence(prediction.prediction!)}
            title="Append this sign to the sentence composer"
          >
            <Plus size={16} />
            <span>Add Sign</span>
          </button>
        )}
      </div>

      {isConfident && (
        <div className="prediction-translations">
          <div className="translation-pill">
            <span className="translation-label">German (DGS)</span>
            <div className="translation-val">{prediction.german_text}</div>
          </div>
          <div className="translation-pill">
            <span className="translation-label">English Meaning</span>
            <div className="translation-val">{prediction.english_text}</div>
          </div>
        </div>
      )}

      {prediction?.status === 'uncertain' && (
        <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--color-warning)' }}>
          {prediction.note || `Model confidence (${confidencePercent}%) is below minimum threshold (${Math.round(threshold * 100)}%).`}
        </p>
      )}

      {/* Confidence Bar */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
          <span>Confidence Score</span>
          <span style={{ fontWeight: 600 }}>{confidencePercent}%</span>
        </div>
        <div className="confidence-bar-container">
          <div
            className="confidence-bar-fill"
            style={{ width: `${Math.min(100, Math.max(0, confidencePercent))}%` }}
          />
        </div>
      </div>

      {/* Top alternatives */}
      {prediction && prediction.top_alternatives && prediction.top_alternatives.length > 0 && (
        <div style={{ marginTop: '0.85rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(0,0,0,0.06)' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
            Top Alternatives:
          </span>
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
            {prediction.top_alternatives.map((alt) => (
              <span
                key={alt.canonical_label}
                style={{
                  fontSize: '0.75rem',
                  padding: '2px 8px',
                  background: 'rgba(255,255,255,0.7)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--color-border)',
                }}
              >
                {alt.canonical_label} ({Math.round(alt.confidence * 100)}%)
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Auto-add toggle */}
      <div style={{ marginTop: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
        <input
          type="checkbox"
          id="autoAddToggle"
          checked={autoAddEnabled}
          onChange={(e) => setAutoAddEnabled(e.target.checked)}
          style={{ cursor: 'pointer' }}
        />
        <label htmlFor="autoAddToggle" style={{ cursor: 'pointer', userSelect: 'none' }}>
          Auto-add recognized signs with transition cooldown
        </label>
      </div>
    </div>
  );
};
