import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Square,
  Camera,
  RefreshCw,
  AlertTriangle,
  History,
  ToggleLeft,
  ToggleRight,
  HelpCircle,
} from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { LandmarkCanvas } from '../components/LandmarkCanvas';
import { PredictionCard } from '../components/PredictionCard';
import { SentenceComposer } from '../components/SentenceComposer';
import {
  initMediaPipe,
  extract144Features,
  ExtractedFrameFeatures,
} from '../services/mediapipe';
import { api } from '../services/api';
import {
  PredictResponse,
  ComposeSentenceResponse,
  RecognitionHistoryItem,
  HealthResponse,
} from '../types/signs';

interface LiveTranslatorViewProps {
  health: HealthResponse | null;
  onOpenHelp: () => void;
  onAddToConversation?: (germanText: string, englishText: string, glosses: string[]) => void;
}

export const LiveTranslatorView: React.FC<LiveTranslatorViewProps> = ({
  health,
  onOpenHelp,
  onAddToConversation,
}) => {
  const {
    videoRef,
    isStreaming,
    permissionState,
    errorMessage,
    startCamera,
    stopCamera,
    availableDevices,
    selectedDeviceId,
    setSelectedDeviceId,
  } = useWebcam();

  // Inference & buffer state
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [isMediaPipeReady, setIsMediaPipeReady] = useState(false);
  const [mediaPipeStatus, setMediaPipeStatus] = useState<string>('Initializing vision pipeline...');
  const [landmarksResult, setLandmarksResult] = useState<any | null>(null);

  const [currentPrediction, setCurrentPrediction] = useState<PredictResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.65);
  const [autoAddEnabled, setAutoAddEnabled] = useState(false);

  // Sentence composer state
  const [composedSigns, setComposedSigns] = useState<string[]>([]);
  const [sentenceComposition, setSentenceComposition] = useState<ComposeSentenceResponse | null>(null);

  // History state (persisted server-side; restored on mount)
  const [history, setHistory] = useState<RecognitionHistoryItem[]>([]);

  useEffect(() => {
    api
      .getHistory(30)
      .then((res) => {
        setHistory(
          res.items.map((e) => ({
            id: `hist-${e.id}`,
            timestamp: new Date(e.created_at).toLocaleTimeString(),
            canonical_label: e.canonical_label,
            german_text: e.german_text || e.canonical_label,
            english_text: e.english_text || e.canonical_label,
            confidence: e.confidence,
          }))
        );
      })
      .catch(() => {
        /* backend offline: start with empty history */
      });
  }, []);

  const handleClearHistory = useCallback(() => {
    setHistory([]);
    api.clearHistory().catch(() => {
      /* non-fatal */
    });
  }, []);

  // Continuous translation experimental mode
  const [continuousMode, setContinuousMode] = useState(false);

  // Buffer refs
  const sequenceBufferRef = useRef<number[][]>([]); // array of 144-dim arrays
  const lastPushTimeRef = useRef<number>(0);        // ~30 fps sampling (matches training clips)
  const lastInferenceTimeRef = useRef<number>(0);
  const lastAddedSignRef = useRef<string | null>(null);
  const lastAddedTimestampRef = useRef<number>(0);
  const animFrameIdRef = useRef<number | null>(null);

  // Initialize MediaPipe Tasks Vision
  useEffect(() => {
    let mounted = true;
    initMediaPipe()
      .then(() => {
        if (mounted) {
          setIsMediaPipeReady(true);
          setMediaPipeStatus('MediaPipe Vision Ready');
        }
      })
      .catch((err) => {
        if (mounted) {
          console.error('Failed to init MediaPipe:', err);
          setMediaPipeStatus('MediaPipe initialization fallback active');
          // Allow operation even with CPU fallback
          setIsMediaPipeReady(true);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Update sentence translation whenever sign sequence changes
  const updateComposition = useCallback(async (signsList: string[]) => {
    if (signsList.length === 0) {
      setSentenceComposition(null);
      return;
    }
    try {
      const res = await api.composeSentence(signsList);
      setSentenceComposition(res);
    } catch (err) {
      console.error('Failed to compose sentence:', err);
    }
  }, []);

  const handleAddSign = useCallback((sign: string) => {
    setComposedSigns((prev) => {
      const updated = [...prev, sign];
      updateComposition(updated);
      return updated;
    });
  }, [updateComposition]);

  const handleUndoSign = useCallback(() => {
    setComposedSigns((prev) => {
      const updated = prev.slice(0, -1);
      updateComposition(updated);
      return updated;
    });
  }, [updateComposition]);

  const handleRemoveSign = useCallback((idx: number) => {
    setComposedSigns((prev) => {
      const updated = prev.filter((_, i) => i !== idx);
      updateComposition(updated);
      return updated;
    });
  }, [updateComposition]);

  const handleClearSentence = useCallback(() => {
    setComposedSigns([]);
    setSentenceComposition(null);
  }, []);

  // Frame processing loop
  useEffect(() => {
    if (!isStreaming || !videoRef.current) {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      return;
    }

    const video = videoRef.current;

    const processFrame = async () => {
      if (video.readyState >= 2) {
        const now = performance.now();
        const extracted: ExtractedFrameFeatures = extract144Features(video, now);
        setLandmarksResult(extracted.landmarksRaw);

        // Sample at ~30 fps so a 30-frame window spans ~1 s of signing, matching the
        // temporal scale of the PHOENIX training clips (raw rAF would run at up to 60 fps).
        if (isRecognitionActive && isMediaPipeReady && now - lastPushTimeRef.current >= 33) {
          lastPushTimeRef.current = now;
          // Push into rolling buffer of 30 frames
          sequenceBufferRef.current.push(extracted.vector);
          if (sequenceBufferRef.current.length > 30) {
            sequenceBufferRef.current.shift();
          }

          // Trigger inference if buffer has 30 frames and interval >= 300ms
          const timeSinceLastInference = now - lastInferenceTimeRef.current;
          if (
            sequenceBufferRef.current.length === 30 &&
            timeSinceLastInference >= 300 &&
            !isPredicting &&
            health?.model_loaded
          ) {
            lastInferenceTimeRef.current = now;
            setIsPredicting(true);

            // Copy snapshot of buffer
            const bufferSnapshot = [...sequenceBufferRef.current];

            api
              .predictSign(bufferSnapshot, `req-${Math.round(now)}`)
              .then((pred) => {
                setCurrentPrediction(pred);

                // Handle history and auto-add with deduplication cooldown
                if (pred.status === 'ok' && pred.prediction) {
                  const label = pred.prediction;
                  const cooldownMs = 1500; // 1.5 seconds cooldown between identical signs
                  const canAdd =
                    label !== lastAddedSignRef.current ||
                    now - lastAddedTimestampRef.current > cooldownMs;

                  if (canAdd) {
                    setHistory((prev) => [
                      {
                        id: `hist-${Date.now()}`,
                        timestamp: new Date().toLocaleTimeString(),
                        canonical_label: label,
                        german_text: pred.german_text || label,
                        english_text: pred.english_text || label,
                        confidence: pred.confidence,
                      },
                      ...prev.slice(0, 19),
                    ]);

                    // Auto-add: manual toggle OR continuous translation mode
                    if (autoAddEnabled || continuousMode) {
                      lastAddedSignRef.current = label;
                      lastAddedTimestampRef.current = now;
                      handleAddSign(label);
                    }
                  }
                }
              })
              .catch((err) => {
                console.debug('Inference response:', err.message);
              })
              .finally(() => {
                setIsPredicting(false);
              });
          }
        }
      }

      animFrameIdRef.current = requestAnimationFrame(processFrame);
    };

    animFrameIdRef.current = requestAnimationFrame(processFrame);

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [
    isStreaming,
    isRecognitionActive,
    isMediaPipeReady,
    isPredicting,
    health?.model_loaded,
    autoAddEnabled,
    continuousMode,
    handleAddSign,
    videoRef,
  ]);

  return (
    <div className="translator-grid">
      {/* Left Column: Live Webcam Stage & Controls */}
      <div>
        <div className="dgs-card" style={{ padding: '1rem' }}>
          <div className="camera-stage">
            <video
              ref={videoRef}
              className="camera-video"
              playsInline
              muted
              autoPlay
            />
            <LandmarkCanvas
              landmarksResult={landmarksResult}
              videoWidth={videoRef.current?.videoWidth || 640}
              videoHeight={videoRef.current?.videoHeight || 480}
            />

            {/* Stage status overlays */}
            <div className="camera-overlay-badge">
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: isStreaming ? '#22c55e' : '#94a3b8',
                }}
              />
              <span>
                {isStreaming
                  ? isRecognitionActive
                    ? 'Tracking & Evaluating'
                    : 'Camera Live (Recognition Paused)'
                  : 'Camera Idle'}
              </span>
            </div>

            {!isStreaming && (
              <div
                style={{
                  position: 'absolute',
                  textAlign: 'center',
                  padding: '2rem',
                  color: '#94a3b8',
                }}
              >
                <Camera size={48} style={{ margin: '0 auto 1rem', opacity: 0.6 }} />
                <h3 style={{ color: '#ffffff', marginBottom: '0.5rem' }}>Camera is Turned Off</h3>
                <p style={{ fontSize: '0.85rem', maxWidth: '340px', margin: '0 auto 1.25rem' }}>
                  Click "Start Camera" to enable client-side hand tracking. Your video remains strictly on your device.
                </p>
                <button className="btn btn-primary" onClick={startCamera}>
                  <Play size={16} />
                  <span>Start Camera</span>
                </button>
              </div>
            )}
          </div>

          {/* Camera Error / Permission Fallback */}
          {permissionState === 'denied' && (
            <div
              style={{
                marginTop: '1rem',
                padding: '0.85rem 1rem',
                background: 'var(--color-error-bg)',
                color: 'var(--color-error)',
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                fontSize: '0.875rem',
              }}
            >
              <AlertTriangle size={20} />
              <div>
                <strong>Camera Access Denied:</strong> {errorMessage}
              </div>
            </div>
          )}

          {/* Camera Controls Bar */}
          <div className="camera-controls-bar">
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {!isStreaming ? (
                <button className="btn btn-primary" onClick={startCamera}>
                  <Camera size={16} />
                  <span>Start Camera</span>
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={stopCamera}>
                  <Square size={16} />
                  <span>Stop Camera</span>
                </button>
              )}

              <button
                className={`btn ${isRecognitionActive ? 'btn-secondary' : 'btn-primary'}`}
                onClick={() => setIsRecognitionActive((prev) => !prev)}
                disabled={!isStreaming}
                title={isRecognitionActive ? 'Pause model recognition' : 'Start model recognition'}
              >
                {isRecognitionActive ? <Square size={16} /> : <Play size={16} />}
                <span>{isRecognitionActive ? 'Pause Recognition' : 'Start Recognition'}</span>
              </button>
            </div>

            {/* Device selection dropdown */}
            {availableDevices.length > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <select
                  value={selectedDeviceId}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                  style={{
                    padding: '0.4rem 0.6rem',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--color-border)',
                    fontSize: '0.8rem',
                  }}
                >
                  {availableDevices.map((d, i) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Vision Pipeline & Buffer Status */}
          <div
            style={{
              marginTop: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.75rem',
              color: 'var(--color-text-muted)',
              borderTop: '1px solid var(--color-border)',
              paddingTop: '0.6rem',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            <span>{mediaPipeStatus}</span>
            <span>
              Buffer: {sequenceBufferRef.current.length}/30 frames (144 features)
            </span>
          </div>
        </div>

        {/* Sentence Composer Component */}
        <SentenceComposer
          signs={composedSigns}
          composition={sentenceComposition}
          onRemoveSign={handleRemoveSign}
          onUndoLastSign={handleUndoSign}
          onClearSentence={handleClearSentence}
          onPostToConversation={onAddToConversation}
        />
      </div>

      {/* Right Column: Prediction Results, Continuous Mode Toggle & History */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Prediction Card */}
        <PredictionCard
          prediction={currentPrediction}
          isPredicting={isPredicting}
          modelReady={health?.model_loaded ?? false}
          onAddToSentence={handleAddSign}
          autoAddEnabled={autoAddEnabled}
          setAutoAddEnabled={setAutoAddEnabled}
          threshold={confidenceThreshold}
        />

        {/* Continuous Translation Mode Card (clearly marked as experimental) */}
        <div className="dgs-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
                onClick={() => setContinuousMode((prev) => !prev)}
                title="Toggle continuous sequence mode"
              >
                {continuousMode ? (
                  <ToggleRight size={28} color="var(--color-primary)" />
                ) : (
                  <ToggleLeft size={28} color="var(--color-text-muted)" />
                )}
              </button>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <h4 style={{ fontSize: '0.95rem' }}>Continuous Translation Mode</h4>
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 700,
                      background: 'var(--color-warning-bg)',
                      color: 'var(--color-warning)',
                      padding: '1px 6px',
                      borderRadius: 'var(--radius-full)',
                    }}
                  >
                    EXPERIMENTAL
                  </span>
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                  {continuousMode
                    ? 'Continuous mode is active: every confident recognition is appended to the sentence composer automatically, with a 1.5 s cooldown on repeated signs.'
                    : 'Continuous multi-sign translation requires dedicated aligned sequence-to-sequence models (e.g. PHOENIX14T).'}
                </p>
              </div>
            </div>
          </div>

          {continuousMode && (
            <div
              style={{
                marginTop: '0.75rem',
                padding: '0.6rem 0.85rem',
                background: 'var(--color-surface-muted)',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.8rem',
                color: 'var(--color-text-secondary)',
              }}
            >
              Hands-free composition running: signs are recognized from the live buffer and
              appended to the sentence below as you sign. Individual sign glosses are composed
              via the backend phrase templates; full grammatical DGS-to-German sentence
              translation still requires a continuous translation model.
            </div>
          )}
        </div>

        {/* Recognition History Card */}
        <div className="dgs-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <History size={16} color="var(--color-text-secondary)" />
              <h4 style={{ fontSize: '0.95rem' }}>Recognition History</h4>
            </div>
            {history.length > 0 && (
              <button
                className="btn btn-secondary btn-sm"
                style={{ padding: '2px 8px', fontSize: '0.7rem' }}
                onClick={handleClearHistory}
              >
                Clear
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', textAlign: 'center', padding: '1rem 0' }}>
              No signs recognized yet — history is stored on the backend and survives page reloads.
            </p>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div key={item.id} className="history-item">
                  <div>
                    <strong style={{ color: 'var(--color-primary)' }}>{item.canonical_label}</strong>
                    <span style={{ marginLeft: '0.5rem', color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>
                      ({item.german_text} / {item.english_text})
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)' }}>
                      {Math.round(item.confidence * 100)}%
                    </span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                      {item.timestamp}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
