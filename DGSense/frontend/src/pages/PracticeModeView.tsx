import React, { useState, useEffect, useRef } from 'react';
import {
  GraduationCap,
  Play,
  CheckCircle2,
  AlertCircle,
  Trophy,
  ArrowRight,
  RefreshCw,
  Camera,
} from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { LandmarkCanvas } from '../components/LandmarkCanvas';
import {
  initMediaPipe,
  extract144Features,
  ExtractedFrameFeatures,
} from '../services/mediapipe';
import { api } from '../services/api';
import { SignEntry, HealthResponse } from '../types/signs';

interface PracticeModeViewProps {
  health: HealthResponse | null;
}

export const PracticeModeView: React.FC<PracticeModeViewProps> = ({ health }) => {
  const { videoRef, isStreaming, startCamera, stopCamera } = useWebcam();
  const [signs, setSigns] = useState<SignEntry[]>([]);
  const [selectedSign, setSelectedSign] = useState<SignEntry | null>(null);
  const [landmarksResult, setLandmarksResult] = useState<any | null>(null);

  const [practiceStatus, setPracticeStatus] = useState<'idle' | 'signing' | 'success' | 'retry'>('idle');
  const [currentConfidence, setCurrentConfidence] = useState<number>(0);
  const [practiceStreak, setPracticeStreak] = useState<number>(0);
  const [feedbackMessage, setFeedbackMessage] = useState<string>('Select a sign below to begin practice.');
  const [stats, setStats] = useState<{ total: number; correct: number; accuracy: number } | null>(null);

  const bufferRef = useRef<number[][]>([]);
  const lastPushRef = useRef<number>(0); // ~30 fps sampling (matches training clips)
  const lastInferenceRef = useRef<number>(0);
  // Last outcome we recorded, so each completed attempt is stored exactly once
  // (and the streak is not inflated by repeated 350 ms polls on a held sign).
  const lastOutcomeRef = useRef<string>('');

  const refreshStats = () => {
    api
      .getPracticeAttempts(200)
      .then((r) => setStats({ total: r.total, correct: r.correct, accuracy: r.accuracy }))
      .catch(() => {
        /* backend offline */
      });
  };

  useEffect(() => {
    initMediaPipe().catch(console.warn);
    refreshStats();
    api.getSignList().then((res) => {
      const supported = res.signs.filter((s) => s.is_model_supported);
      setSigns(supported);
      if (supported.length > 0) {
        setSelectedSign(supported[0]);
      }
    });
  }, []);

  // Switching target sign starts a fresh attempt round
  useEffect(() => {
    lastOutcomeRef.current = '';
  }, [selectedSign]);

  // Frame processing loop
  useEffect(() => {
    if (!isStreaming || !videoRef.current) return;
    let animId: number;

    const loop = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2) {
        const now = performance.now();
        const extracted: ExtractedFrameFeatures = extract144Features(video, now);
        setLandmarksResult(extracted.landmarksRaw);

        // ~30 fps sampling so a 30-frame window spans ~1 s of signing (training scale)
        if (now - lastPushRef.current >= 33) {
          lastPushRef.current = now;
          bufferRef.current.push(extracted.vector);
          if (bufferRef.current.length > 30) bufferRef.current.shift();
        }

        if (
          selectedSign &&
          bufferRef.current.length === 30 &&
          now - lastInferenceRef.current >= 350 &&
          health?.model_loaded
        ) {
          lastInferenceRef.current = now;
          api
            .predictSign([...bufferRef.current])
            .then((res) => {
              const matchesTarget = res.prediction === selectedSign.canonical_label;
              const conf = res.confidence;
              setCurrentConfidence(conf);

              const outcome: 'success' | 'retry' | 'signing' =
                res.status === 'ok' && matchesTarget && conf >= 0.70
                  ? 'success'
                  : res.status === 'ok' && !matchesTarget
                    ? 'retry'
                    : 'signing';

              if (outcome === 'signing') {
                // Sign in progress: reset the round so the next completed outcome records.
                lastOutcomeRef.current = '';
                setPracticeStatus('signing');
                setFeedbackMessage(`Sign in progress... Keep moving at steady tempo.`);
              } else if (outcome !== lastOutcomeRef.current) {
                // Completed attempt: record it once (server-side SQLite) and update UI.
                lastOutcomeRef.current = outcome;
                if (outcome === 'success') {
                  setPracticeStatus('success');
                  setFeedbackMessage(`Excellent! Recognized "${selectedSign.german_text}" (${Math.round(conf * 100)}% match).`);
                  setPracticeStreak((s) => s + 1);
                } else {
                  setPracticeStatus('retry');
                  setFeedbackMessage(`Detected "${res.prediction || 'Unknown'}" instead of "${selectedSign.canonical_label}". Adjust your hand shape.`);
                }
                api
                  .addPracticeAttempt({
                    target_sign: selectedSign.canonical_label,
                    recognized_sign: res.prediction,
                    correct: outcome === 'success',
                    confidence: conf,
                  })
                  .then(refreshStats)
                  .catch(() => {
                    /* non-fatal: attempt stays local */
                  });
              }
            })
            .catch(() => {});
        }
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isStreaming, selectedSign, health?.model_loaded, videoRef]);

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <GraduationCap size={24} color="var(--color-primary)" />
            <h2 style={{ fontSize: '1.5rem' }}>Interactive Practice Mode</h2>
          </div>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
            Choose a target DGS sign, review execution cues, and perform the sign in front of the camera for real-time recognition feedback.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: '#ffffff', padding: '0.5rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Trophy size={18} color="#eab308" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Practice Streak:</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--color-primary)' }}>{practiceStreak}</span>
          </div>
          {stats && stats.total > 0 && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                borderLeft: '1px solid var(--color-border)',
                paddingLeft: '1rem',
                fontSize: '0.85rem',
              }}
              title="Stored practice results (persisted on the backend)"
            >
              <span style={{ fontWeight: 600 }}>All-time:</span>
              <span style={{ fontWeight: 800, color: 'var(--color-primary)' }}>
                {Math.round(stats.accuracy * 100)}%
              </span>
              <span style={{ color: 'var(--color-text-muted)' }}>
                ({stats.correct}/{stats.total} attempts)
              </span>
            </div>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.5rem' }}>
        {/* Left: Camera View */}
        <div className="dgs-card" style={{ padding: '1rem' }}>
          <div className="camera-stage" style={{ aspectRatio: '4 / 3' }}>
            <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
            <LandmarkCanvas
              landmarksResult={landmarksResult}
              videoWidth={videoRef.current?.videoWidth || 640}
              videoHeight={videoRef.current?.videoHeight || 480}
            />

            {!isStreaming && (
              <div style={{ position: 'absolute', textAlign: 'center', padding: '2rem', color: '#94a3b8' }}>
                <Camera size={44} style={{ margin: '0 auto 0.75rem', opacity: 0.6 }} />
                <h4 style={{ color: '#ffffff', marginBottom: '0.5rem' }}>Camera Paused</h4>
                <button className="btn btn-primary" onClick={startCamera}>
                  <Play size={16} />
                  <span>Start Camera</span>
                </button>
              </div>
            )}
          </div>

          <div style={{ marginTop: '0.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {!isStreaming ? (
                <button className="btn btn-primary btn-sm" onClick={startCamera}>
                  Start Camera
                </button>
              ) : (
                <button className="btn btn-secondary btn-sm" onClick={stopCamera}>
                  Pause Camera
                </button>
              )}
            </div>

            <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
              Target: <strong style={{ color: 'var(--color-primary)' }}>{selectedSign?.canonical_label || 'None'}</strong>
            </div>
          </div>
        </div>

        {/* Right: Practice Instructions & Feedback */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Target Card */}
          {selectedSign && (
            <div className="dgs-card" style={{ borderLeft: '4px solid var(--color-primary)' }}>
              <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
                Target Sign
              </span>
              <h3 style={{ fontSize: '1.6rem', color: 'var(--color-primary)', margin: '0.25rem 0' }}>
                {selectedSign.canonical_label}
              </h3>
              <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--color-text-main)', marginBottom: '0.5rem' }}>
                {selectedSign.german_text} ({selectedSign.english_text})
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
                {selectedSign.description}
              </p>
            </div>
          )}

          {/* Feedback Card */}
          <div className="dgs-card">
            <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              Real-Time Feedback
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem' }}>
              {practiceStatus === 'success' ? (
                <CheckCircle2 size={28} color="var(--color-success)" />
              ) : practiceStatus === 'retry' ? (
                <AlertCircle size={28} color="var(--color-warning)" />
              ) : (
                <RefreshCw size={28} color="var(--color-primary)" className={isStreaming ? 'spin' : ''} />
              )}
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--color-text-main)' }}>
                {feedbackMessage}
              </div>
            </div>

            {/* Confidence Meter */}
            <div style={{ marginTop: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                <span>Match Alignment</span>
                <span style={{ fontWeight: 700 }}>{Math.round(currentConfidence * 100)}%</span>
              </div>
              <div className="confidence-bar-container">
                <div
                  className="confidence-bar-fill"
                  style={{
                    width: `${Math.round(currentConfidence * 100)}%`,
                    background:
                      practiceStatus === 'success'
                        ? 'var(--color-success)'
                        : 'linear-gradient(90deg, #3b82f6, #7c3aed)',
                  }}
                />
              </div>
            </div>

            {!health?.model_loaded && (
              <p style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--color-warning)' }}>
                Note: Backend running in inspection mode. Feedback simulates feature tracking until trained model.pt is loaded.
              </p>
            )}
          </div>

          {/* Sign Selector List */}
          <div className="dgs-card">
            <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Select Practice Concept</h4>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {signs.map((s) => (
                <button
                  key={s.canonical_label}
                  className={`btn btn-sm ${
                    selectedSign?.canonical_label === s.canonical_label
                      ? 'btn-primary'
                      : 'btn-secondary'
                  }`}
                  onClick={() => {
                    setSelectedSign(s);
                    setPracticeStatus('idle');
                    setCurrentConfidence(0);
                    setFeedbackMessage(`Practicing "${s.german_text}". Perform the sign steadily.`);
                  }}
                >
                  {s.canonical_label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
