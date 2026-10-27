import React from 'react';
import { X, CheckCircle, Shield, AlertTriangle } from 'lucide-react';

interface InstructionsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const InstructionsModal: React.FC<InstructionsModalProps> = ({
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
      }}
      onClick={onClose}
    >
      <div
        className="dgs-card"
        style={{
          maxWidth: '620px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          position: 'relative',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '1.25rem',
            right: '1.25rem',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--color-text-muted)',
          }}
          title="Close modal"
        >
          <X size={20} />
        </button>

        <h2 style={{ fontSize: '1.4rem', marginBottom: '0.75rem' }}>
          Welcome to DGSense
        </h2>
        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
          DGSense is a real-time German Sign Language (Deutsche Gebärdensprache - DGS) recognition system developed for MCA research.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <CheckCircle size={20} color="var(--color-primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>1. Camera Setup</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                Position yourself approximately 0.5 – 1.0 meter away with your face, upper chest, and both hands comfortably inside the camera frame. Ensure good lighting.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <CheckCircle size={20} color="var(--color-primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>2. Performing Signs</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                DGS signs are temporal gestures (motion across time). The system samples 30 frames (~1.0 - 1.5 seconds) to classify supported gestures like <em>Hallo</em>, <em>Danke</em>, <em>Ja</em>, or <em>Nein</em>.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <CheckCircle size={20} color="var(--color-primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>3. Sentence Composer</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                Recognized signs can be appended to create a composed phrase. Verified combinations (e.g. <em>HELLO + HOW_ARE_YOU</em>) map to natural German/English phrasing.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <Shield size={20} color="var(--color-accent-violet)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>4. Privacy by Design</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                All computer vision landmark extraction executes client-side directly in your browser using MediaPipe Tasks Vision. Raw camera video never leaves your device.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <AlertTriangle size={20} color="var(--color-warning)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>5. Academic Distinction</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                Individual sign recognition is distinct from complete continuous sign language translation. Concatenating sign glosses does not substitute for native DGS grammar.
              </p>
            </div>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem', textAlign: 'right' }}>
          <button className="btn btn-primary" onClick={onClose}>
            Got it, let's start!
          </button>
        </div>
      </div>
    </div>
  );
};
