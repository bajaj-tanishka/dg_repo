import React, { useState, useEffect } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  Database,
  Layers,
  FileCode,
  ShieldCheck,
  RefreshCw,
  ExternalLink,
  HardDrive,
  Globe,
  ArrowLeftRight,
} from 'lucide-react';
import { api } from '../services/api';
import {
  HealthResponse,
  ModelInfoResponse,
  DatasetReadinessResponse,
  LanguageConfigResponse,
} from '../types/signs';

interface ModelStatusViewProps {
  health: HealthResponse | null;
  onRefreshHealth: () => void;
}

export const ModelStatusView: React.FC<ModelStatusViewProps> = ({
  health,
  onRefreshHealth,
}) => {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [readiness, setReadiness] = useState<DatasetReadinessResponse | null>(null);
  const [langConfig, setLangConfig] = useState<LanguageConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchDiagnostics = async () => {
    setLoading(true);
    try {
      const [info, ready, lang] = await Promise.all([
        api.getModelInfo().catch(() => null),
        api.getDatasetReadiness().catch(() => null),
        api.getLanguageConfig().catch(() => null),
      ]);
      setModelInfo(info);
      setReadiness(ready);
      setLangConfig(lang);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  const isModelReady = health?.model_loaded ?? false;

  return (
    <div style={{ maxWidth: '1080px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <Activity size={24} color="var(--color-primary)" />
            <h2 style={{ fontSize: '1.5rem' }}>Two-Way Model, Dataset & Corpus Status</h2>
          </div>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
            Inspect system diagnostics, neural network weights, everyday vocabulary readiness, research corpus citations, and multi-language configuration.
          </p>
        </div>

        <button
          className="btn btn-secondary btn-sm"
          onClick={() => {
            onRefreshHealth();
            fetchDiagnostics();
          }}
        >
          <RefreshCw size={14} />
          <span>Refresh All</span>
        </button>
      </div>

      {/* Top Status Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        {/* Backend Connectivity */}
        <div className="dgs-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            style={{
              width: '44px',
              height: '44px',
              borderRadius: 'var(--radius-md)',
              background: health ? 'var(--color-success-bg)' : 'var(--color-error-bg)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {health ? <CheckCircle2 size={24} color="var(--color-success)" /> : <AlertTriangle size={24} color="var(--color-error)" />}
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              Backend Service
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              {health ? 'Online (V2 Two-Way)' : 'Disconnected'}
            </div>
          </div>
        </div>

        {/* Model Checkpoint */}
        <div className="dgs-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            style={{
              width: '44px',
              height: '44px',
              borderRadius: 'var(--radius-md)',
              background: isModelReady ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {isModelReady ? <CheckCircle2 size={24} color="var(--color-success)" /> : <AlertTriangle size={24} color="var(--color-warning)" />}
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              Temporal BiGRU (model.pt)
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              {isModelReady ? `Loaded (v${health?.model_version || '1.0'})` : 'Inspection Mode'}
            </div>
          </div>
        </div>

        {/* Two-Way Translation Ready */}
        <div className="dgs-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            style={{
              width: '44px',
              height: '44px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--color-accent-violet-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <ArrowLeftRight size={24} color="var(--color-accent-violet)" />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              Reverse Translator
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              Ready (20+ Verified Clips)
            </div>
          </div>
        </div>

        {/* Compute Device */}
        <div className="dgs-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            style={{
              width: '44px',
              height: '44px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--color-primary-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Cpu size={24} color="var(--color-primary)" />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              Inference Hardware
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              {health?.device?.toUpperCase() || 'CPU'}
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Readiness Audit Banner */}
      {readiness && (
        <div className="dgs-card" style={{ marginBottom: '1.5rem', borderLeft: '4px solid var(--color-primary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <h3 style={{ fontSize: '1.1rem' }}>Everyday Communication Readiness Audit</h3>
            <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--color-primary)' }}>
              {readiness.readiness_percentage}% Verified Coverage ({readiness.ready_count}/{readiness.total_target_concepts})
            </span>
          </div>

          <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.75rem' }}>
            To uphold academic and linguistic integrity, requested concepts that lack verified public research clips are flagged as <code>DATA_REQUIRED</code> instead of fabricating artificial signs.
          </p>

          {readiness.missing_concepts.length > 0 && (
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-error)', textTransform: 'uppercase' }}>
                Concepts requiring real video recording (DATA_REQUIRED):
              </span>
              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                {readiness.missing_concepts.map((concept) => (
                  <span
                    key={concept}
                    style={{
                      padding: '2px 8px',
                      background: 'var(--color-error-bg)',
                      color: 'var(--color-error)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      border: '1px solid rgba(220, 38, 38, 0.2)',
                    }}
                  >
                    {concept}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Research Corpus Sources & Licensing Requirements */}
      <div className="dgs-card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <Database size={18} color="var(--color-primary)" />
          <h3 style={{ fontSize: '1.1rem' }}>Recognized Public DGS Research Corpora</h3>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {readiness?.reference_sources.map((src) => (
            <div
              key={src.name}
              style={{
                background: 'var(--color-surface-muted)',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.85rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong style={{ color: 'var(--color-primary)' }}>{src.name}</strong>
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '3px',
                    fontSize: '0.75rem',
                    color: 'var(--color-primary)',
                    textDecoration: 'none',
                  }}
                >
                  <span>Portal</span>
                  <ExternalLink size={12} />
                </a>
              </div>
              {src.institution && (
                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                  {src.institution}
                </div>
              )}
              <div style={{ marginTop: '0.35rem', color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>
                <strong>Licence:</strong> {src.licence} — {src.licence_notes}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            marginTop: '1rem',
            padding: '0.6rem 0.85rem',
            background: 'var(--color-surface-card)',
            border: '1px dashed var(--color-border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8rem',
            color: 'var(--color-text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <HardDrive size={16} color="var(--color-accent-violet)" />
          <span>
            <strong>Download Limit Policy:</strong> Total dataset downloads are capped under 5 GB (5,120 MB) by default to prevent silent multi-gigabyte bandwidth consumption.
          </span>
        </div>
      </div>

      {/* Multi-Language Extensibility & Architecture Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Multi-Language Config */}
        <div className="dgs-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Globe size={18} color="var(--color-primary)" />
            <h3 style={{ fontSize: '1.1rem' }}>Multi-Language Configuration</h3>
          </div>
          <table style={{ width: '100%', fontSize: '0.85rem', borderCollapse: 'collapse' }}>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Sign Language Code:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>{langConfig?.sign_language_code || 'dgs-DE'}</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Primary Spoken Language:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>{langConfig?.spoken_language_code || 'de-DE'}</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Secondary Spoken Language:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>{langConfig?.secondary_spoken_lang || 'en-US'}</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Gloss Convention:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>IDGS Uppercase German Lemma</td>
              </tr>
              <tr>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Extensibility Note:</td>
                <td style={{ padding: '0.5rem 0', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                  Adding another sign language requires its own verified vocabulary manifest and landmark models.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Feature Schema Specs */}
        <div className="dgs-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <FileCode size={18} color="var(--color-accent-violet)" />
            <h3 style={{ fontSize: '1.1rem' }}>Feature Schema (dgsense-v1)</h3>
          </div>
          <table style={{ width: '100%', fontSize: '0.85rem', borderCollapse: 'collapse' }}>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Schema Version:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>dgsense-v1</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Temporal Sequence:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>30 Frames (~1.0 - 1.5 seconds)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Feature Dimension:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>144 Floats per Frame</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Landmark Layout:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>Left Hand (63) + Right Hand (63) + Pose (18)</td>
              </tr>
              <tr>
                <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>Missing Fill Strategy:</td>
                <td style={{ padding: '0.5rem 0', fontWeight: 600 }}>0.0 (Zero-padded when hand outside frame)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Limitations and Ethical Considerations */}
      <div className="dgs-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <ShieldCheck size={18} color="var(--color-success)" />
          <h3 style={{ fontSize: '1.1rem' }}>Ethical Considerations & Two-Way Scope</h3>
        </div>
        <ul style={{ paddingLeft: '1.25rem', fontSize: '0.85rem', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
          <li>
            <strong>Privacy Protection:</strong> Client-side MediaPipe landmark extraction ensures that raw camera pixels never leave your device.
          </li>
          <li>
            <strong>Linguistic Scope:</strong> DGS is a natural visual language with rich spatial grammar and facial non-manual expressions. Concatenating words does not constitute fluent continuous DGS. DGSense clearly identifies isolated sign recognition, verified phrase templates, and word-level support.
          </li>
          <li>
            <strong>Honest Unsupported Messaging:</strong> In Reverse Translation (German $\to$ DGS), if a verified sign is not available, the application explicitly states that no verified sign exists, instead of generating inaccurate or invented signs.
          </li>
        </ul>
      </div>
    </div>
  );
};
