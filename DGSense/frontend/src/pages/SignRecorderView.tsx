import React, { useState, useEffect, useRef } from 'react';
import {
  Disc,
  Play,
  Square,
  Download,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Camera,
  Layers,
} from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { LandmarkCanvas } from '../components/LandmarkCanvas';
import {
  initMediaPipe,
  extract144Features,
  ExtractedFrameFeatures,
} from '../services/mediapipe';
import { api } from '../services/api';
import { SignEntry } from '../types/signs';

interface RecordedSample {
  sample_id: string;
  label: string;
  timestamp: string;
  frames: number[][]; // 30 frames x 144 floats
  handsDetectedCount: number; // Quality check: how many frames had hands
  serverId?: number; // id in the backend SQLite store (undefined = local only)
}

export const SignRecorderView: React.FC = () => {
  const { videoRef, isStreaming, startCamera, stopCamera } = useWebcam();
  const [signs, setSigns] = useState<SignEntry[]>([]);
  const [selectedLabel, setSelectedLabel] = useState<string>('HELLO');
  const [signerId, setSignerId] = useState<string>('signer_01');

  const [recordedSamples, setRecordedSamples] = useState<RecordedSample[]>([]);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);

  const [landmarksResult, setLandmarksResult] = useState<any | null>(null);
  const recordingFramesRef = useRef<number[][]>([]);
  const handsDetectedFramesRef = useRef<number>(0);
  const lastRecordPushRef = useRef<number>(0); // ~30 fps sampling (matches training clips)

  useEffect(() => {
    initMediaPipe().catch(console.warn);
    api.getSignList().then((res) => {
      setSigns(res.signs);
      if (res.signs.length > 0) {
        setSelectedLabel(res.signs[0].canonical_label);
      }
    });
    // Restore samples previously uploaded to the backend
    api
      .getSamples()
      .then((res) => {
        setRecordedSamples(
          res.samples.map((s) => {
            const hands = /(\d+)\s*\/\s*30/.exec(s.notes || '');
            return {
              sample_id: `srv-${s.id}`,
              label: s.label,
              timestamp: new Date(s.created_at).toLocaleTimeString(),
              frames: [], // frames are fetched from the backend on export
              handsDetectedCount: hands ? parseInt(hands[1], 10) : 0,
              serverId: s.id,
            };
          })
        );
      })
      .catch(() => {
        /* backend offline: list starts empty */
      });
  }, []);

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

        // If recording active, record 30 frames at ~30 fps so a recorded clip spans
        // ~1 s of signing - the same temporal scale as the PHOENIX training clips.
        if (isRecording && now - lastRecordPushRef.current >= 33) {
          lastRecordPushRef.current = now;
          recordingFramesRef.current.push(extracted.vector);
          if (extracted.leftHandDetected || extracted.rightHandDetected) {
            handsDetectedFramesRef.current += 1;
          }
          const currentCount = recordingFramesRef.current.length;
          setRecordProgress(currentCount);

          if (currentCount >= 30) {
            // Completed 30 frames!
            const handsCount = handsDetectedFramesRef.current;
            const newSample: RecordedSample = {
              sample_id: `sample_${selectedLabel}_${Date.now()}`,
              label: selectedLabel,
              timestamp: new Date().toLocaleTimeString(),
              frames: [...recordingFramesRef.current],
              handsDetectedCount: handsCount,
            };
            setRecordedSamples((prev) => [newSample, ...prev]);
            setIsRecording(false);
            setRecordProgress(0);
            recordingFramesRef.current = [];
            handsDetectedFramesRef.current = 0;

            // Upload to the backend store (survives refresh, usable for retraining)
            api
              .uploadSample({
                label: selectedLabel,
                frames: newSample.frames,
                signer_id: signerId,
                notes: `${handsCount}/30 hands`,
              })
              .then((saved) => {
                setRecordedSamples((prev) =>
                  prev.map((s) =>
                    s.sample_id === newSample.sample_id ? { ...s, serverId: saved.id } : s
                  )
                );
              })
              .catch((err) => {
                console.warn('Sample upload failed (kept locally):', err?.message);
              });
          }
        }
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isStreaming, isRecording, selectedLabel, signerId, videoRef]);

  // Start recording workflow with 3s countdown
  const handleStartCountdown = () => {
    if (!isStreaming) return;
    setCountdown(3);

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(timer);
          // Start 30-frame capture
          recordingFramesRef.current = [];
          handsDetectedFramesRef.current = 0;
          setIsRecording(true);
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleDeleteSample = (sampleId: string) => {
    const sample = recordedSamples.find((s) => s.sample_id === sampleId);
    if (sample?.serverId !== undefined) {
      api.deleteSample(sample.serverId).catch(() => {
        /* non-fatal */
      });
    }
    setRecordedSamples((prev) => prev.filter((s) => s.sample_id !== sampleId));
  };

  const handleExportJSON = async () => {
    if (recordedSamples.length === 0) return;

    // Prefer the backend store (has frames for every sample, incl. previous sessions)
    let exportSamples: Array<{
      sample_id: string;
      label: string;
      signer_id: string;
      hands_detected_frames: number;
      landmarks: number[][];
    }>;
    try {
      const res = await api.getSamples(true);
      exportSamples = res.samples.map((s) => {
        const hands = /(\d+)\s*\/\s*30/.exec(s.notes || '');
        return {
          sample_id: `srv-${s.id}`,
          label: s.label,
          signer_id: s.signer_id || signerId,
          hands_detected_frames: hands ? parseInt(hands[1], 10) : 0,
          landmarks: s.frames || [],
        };
      });
    } catch {
      // Backend unreachable: fall back to whatever is loaded locally
      exportSamples = recordedSamples.map((s) => ({
        sample_id: s.sample_id,
        label: s.label,
        signer_id: signerId,
        hands_detected_frames: s.handsDetectedCount,
        landmarks: s.frames,
      }));
    }

    const exportData = {
      dataset_name: 'DGSense_Custom_Landmarks_Dataset',
      created_at: new Date().toISOString(),
      schema_version: 'dgsense-v1',
      sequence_length: 30,
      feature_dim: 144,
      signer_id: signerId,
      total_samples: exportSamples.length,
      samples: exportSamples,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dgsense_dataset_${selectedLabel}_${exportSamples.length}samples.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <Disc size={24} color="var(--color-primary)" />
          <h2 style={{ fontSize: '1.5rem' }}>Dataset Collector & Sign Recorder</h2>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
          Record real, independent DGS landmark sequences for training or fine-tuning. Each sample captures exactly 30 frames (144 features) conforming to the <code>dgsense-v1</code> schema. Samples are uploaded to the backend database automatically and survive page reloads.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem' }}>
        {/* Left: Camera Stage & Recording Controls */}
        <div className="dgs-card" style={{ padding: '1rem' }}>
          <div className="camera-stage" style={{ position: 'relative', aspectRatio: '4 / 3' }}>
            <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
            <LandmarkCanvas
              landmarksResult={landmarksResult}
              videoWidth={videoRef.current?.videoWidth || 640}
              videoHeight={videoRef.current?.videoHeight || 480}
            />

            {/* Countdown Overlay */}
            {countdown !== null && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(0,0,0,0.5)',
                  backdropFilter: 'blur(4px)',
                }}
              >
                <span
                  style={{
                    fontSize: '6rem',
                    fontWeight: 900,
                    color: '#ffffff',
                    animation: 'pulse-dot 1s infinite',
                  }}
                >
                  {countdown}
                </span>
              </div>
            )}

            {/* Recording Indicator */}
            {isRecording && (
              <div
                style={{
                  position: 'absolute',
                  top: '1rem',
                  left: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  background: 'rgba(220, 38, 38, 0.9)',
                  color: '#ffffff',
                  padding: '0.4rem 0.85rem',
                  borderRadius: 'var(--radius-full)',
                  fontWeight: 700,
                  fontSize: '0.8rem',
                }}
              >
                <span
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: '#ffffff',
                    animation: 'pulse-dot 0.6s infinite',
                  }}
                />
                <span>RECORDING ({recordProgress}/30 frames)</span>
              </div>
            )}

            {!isStreaming && (
              <div style={{ position: 'absolute', textAlign: 'center', padding: '2rem', color: '#94a3b8' }}>
                <Camera size={44} style={{ margin: '0 auto 0.75rem', opacity: 0.6 }} />
                <h4 style={{ color: '#ffffff', marginBottom: '0.5rem' }}>Camera Idle</h4>
                <button className="btn btn-primary" onClick={startCamera}>
                  <Play size={16} />
                  <span>Start Camera</span>
                </button>
              </div>
            )}
          </div>

          {/* Controls Bar */}
          <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {!isStreaming ? (
                <button className="btn btn-primary" onClick={startCamera}>
                  Start Camera
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={stopCamera}>
                  Stop Camera
                </button>
              )}

              <button
                className="btn btn-danger"
                onClick={handleStartCountdown}
                disabled={!isStreaming || isRecording || countdown !== null}
                title="Start 3-second countdown to record 30-frame sequence"
              >
                <Disc size={16} />
                <span>Record Example (30 Frames)</span>
              </button>
            </div>

            <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
              Label: <strong>{selectedLabel}</strong>
            </span>
          </div>
        </div>

        {/* Right: Label Selection, Configuration & Recorded Samples List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Settings Card */}
          <div className="dgs-card">
            <h4 style={{ fontSize: '0.95rem', marginBottom: '0.75rem' }}>Recording Parameters</h4>

            <div style={{ marginBottom: '0.75rem' }}>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
                Target Sign Label:
              </label>
              <select
                value={selectedLabel}
                onChange={(e) => setSelectedLabel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--color-border)',
                  fontSize: '0.875rem',
                }}
              >
                {signs.map((s) => (
                  <option key={s.canonical_label} value={s.canonical_label}>
                    {s.canonical_label} ({s.german_text} / {s.english_text})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
                Signer Identifier (for independent split):
              </label>
              <input
                type="text"
                value={signerId}
                onChange={(e) => setSignerId(e.target.value)}
                placeholder="e.g. signer_01"
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--color-border)',
                  fontSize: '0.875rem',
                }}
              />
            </div>
          </div>

          {/* Recorded Samples Review */}
          <div className="dgs-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Layers size={16} color="var(--color-primary)" />
                <h4 style={{ fontSize: '0.95rem' }}>
                  Recorded Examples ({recordedSamples.length})
                </h4>
              </div>

              {recordedSamples.length > 0 && (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={handleExportJSON}
                  title="Export recorded samples as JSON dataset"
                >
                  <Download size={14} />
                  <span>Export JSON</span>
                </button>
              )}
            </div>

            {recordedSamples.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', textAlign: 'center', padding: '1.5rem 0' }}>
                No samples recorded yet. Click "Record Example" to capture a 30-frame sequence.
              </p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '280px', overflowY: 'auto' }}>
                {recordedSamples.map((sample) => {
                  const qualityGood = sample.handsDetectedCount >= 20;
                  return (
                    <div
                      key={sample.sample_id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '0.6rem 0.75rem',
                        background: 'var(--color-surface-muted)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.85rem',
                      }}
                    >
                      <div>
                        <strong>{sample.label}</strong>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '2px' }}>
                          <span>{sample.timestamp}</span>
                          <span>•</span>
                          <span style={{ color: qualityGood ? 'var(--color-success)' : 'var(--color-warning)' }}>
                            {sample.handsDetectedCount}/30 frames tracked
                          </span>
                        </div>
                      </div>

                      <button
                        className="btn btn-secondary btn-sm"
                        style={{ padding: '4px 6px', color: 'var(--color-error)' }}
                        onClick={() => handleDeleteSample(sample.sample_id)}
                        title="Delete this sample"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
