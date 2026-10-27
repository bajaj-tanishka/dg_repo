import React, { useState, useEffect, useRef } from 'react';
import {
  ArrowLeftRight,
  Mic,
  MicOff,
  Send,
  Volume2,
  Copy,
  Check,
  RotateCcw,
  Sparkles,
  Info,
  AlertCircle,
  Play,
  Pause,
  MessageSquarePlus,
  Video,
} from 'lucide-react';
import { api } from '../services/api';
import { speechService } from '../services/speech';
import { ReverseTranslateResponse, SignClipItem } from '../types/signs';

interface ReverseTranslatorViewProps {
  onAddToConversation?: (text: string, glosses: string[], clips: SignClipItem[], matchType: string) => void;
}

const COMMON_PHRASES = [
  'Guten Morgen',
  'Danke',
  'Vielen Dank',
  'Bitte',
  'Entschuldigung',
  'Wie geht es dir?',
  'Kannst du mir helfen?',
  'Ich brauche Hilfe',
  'Lass uns gehen',
  'Warte bitte',
  'Auf Wiedersehen',
  'Stopp',
];

export const ReverseTranslatorView: React.FC<ReverseTranslatorViewProps> = ({
  onAddToConversation,
}) => {
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationResult, setTranslationResult] = useState<ReverseTranslateResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [activeClipIndex, setActiveClipIndex] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isPlayingVideo, setIsPlayingVideo] = useState(false);

  const speechControllerRef = useRef<{ stop: () => void } | null>(null);

  const handleTranslate = async (textToTranslate?: string) => {
    const query = textToTranslate !== undefined ? textToTranslate : inputText;
    if (!query.trim()) return;

    setIsTranslating(true);
    setSpeechError(null);
    try {
      const res = await api.reverseTranslate(query);
      setTranslationResult(res);
      setActiveClipIndex(0);
    } catch (err: any) {
      console.error('Reverse translation error:', err);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleToggleMic = () => {
    if (isListening) {
      speechControllerRef.current?.stop();
      setIsListening(false);
    } else {
      setSpeechError(null);
      const controller = speechService.startSpeechRecognition(
        (transcript) => {
          setInputText(transcript);
        },
        (errMsg) => {
          setSpeechError(errMsg);
          setIsListening(false);
        },
        () => {
          setIsListening(false);
        }
      );

      if (controller) {
        speechControllerRef.current = controller;
        setIsListening(true);
      }
    }
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeakGerman = () => {
    const textToSpeak = translationResult?.original_text || inputText;
    if (textToSpeak) {
      speechService.speakGerman(textToSpeak);
    }
  };

  const handleAddConversation = () => {
    if (!translationResult || !onAddToConversation) return;
    onAddToConversation(
      translationResult.original_text,
      translationResult.dgs_gloss_sequence,
      translationResult.clips,
      translationResult.match_type
    );
  };

  const currentClip = translationResult?.clips?.[activeClipIndex];

  return (
    <div style={{ maxWidth: '1080px', margin: '0 auto' }}>
      {/* Page Title */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <ArrowLeftRight size={24} color="var(--color-primary)" />
          <h2 style={{ fontSize: '1.5rem' }}>Reverse Translation: Text & Speech → DGS Signs</h2>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
          Direction B: Enter spoken or typed German to generate verified German Sign Language (DGS) sign representations and gloss sequences.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.5rem' }}>
        {/* Left Column: Input Form & Quick Phrases */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* German Input Card */}
          <div className="dgs-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <label
                htmlFor="germanInput"
                style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}
              >
                German Input (Spoken or Typed)
              </label>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className={`btn btn-sm ${isListening ? 'btn-danger' : 'btn-secondary'}`}
                  onClick={handleToggleMic}
                  title={isListening ? 'Stop listening' : 'Speak into microphone in German'}
                >
                  {isListening ? <MicOff size={14} /> : <Mic size={14} />}
                  <span>{isListening ? 'Listening...' : 'Microphone'}</span>
                </button>

                {inputText && (
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => {
                      setInputText('');
                      setTranslationResult(null);
                    }}
                    title="Clear input"
                  >
                    <RotateCcw size={14} />
                  </button>
                )}
              </div>
            </div>

            {speechError && (
              <div
                style={{
                  padding: '0.5rem 0.75rem',
                  background: 'var(--color-warning-bg)',
                  color: 'var(--color-warning)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.8rem',
                  marginBottom: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                <AlertCircle size={16} />
                <span>{speechError}</span>
              </div>
            )}

            <div style={{ position: 'relative' }}>
              <textarea
                id="germanInput"
                rows={3}
                placeholder="Gib einen deutschen Satz ein (z. B. 'Guten Morgen', 'Danke', 'Wie geht es dir?')..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleTranslate();
                  }
                }}
                style={{
                  width: '100%',
                  padding: '0.85rem',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--color-border)',
                  fontSize: '1rem',
                  fontFamily: 'inherit',
                  resize: 'none',
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.75rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                Press <code>Enter</code> to translate
              </span>

              <button
                className="btn btn-primary"
                onClick={() => handleTranslate()}
                disabled={!inputText.trim() || isTranslating}
              >
                <Send size={15} />
                <span>{isTranslating ? 'Translating...' : 'Translate to DGS'}</span>
              </button>
            </div>
          </div>

          {/* Quick Select Common Phrases */}
          <div className="dgs-card">
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--color-text-secondary)', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
              Common Everyday DGS Expressions
            </h4>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              {COMMON_PHRASES.map((phrase) => (
                <button
                  key={phrase}
                  className="btn btn-secondary btn-sm"
                  style={{ fontSize: '0.8rem', padding: '0.35rem 0.65rem' }}
                  onClick={() => {
                    setInputText(phrase);
                    handleTranslate(phrase);
                  }}
                >
                  {phrase}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: DGS Output Panel */}
        <div>
          <div className="dgs-card" style={{ minHeight: '380px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={18} color="var(--color-accent-violet)" />
                <h3 style={{ fontSize: '1.1rem' }}>DGS Sign Representation</h3>
              </div>

              {translationResult && (
                <span
                  className={`badge-tag ${
                    translationResult.match_type === 'exact_phrase_match'
                      ? 'badge-verified'
                      : translationResult.match_type === 'compositional_sequence'
                      ? 'badge-model-supported'
                      : translationResult.match_type === 'word_level_demonstration'
                      ? 'badge-approx'
                      : 'badge-dictionary-only'
                  }`}
                >
                  {translationResult.match_type === 'exact_phrase_match'
                    ? 'Verified DGS Phrase'
                    : translationResult.match_type === 'compositional_sequence'
                    ? 'Compositional Sequence'
                    : translationResult.match_type === 'word_level_demonstration'
                    ? 'Word-Level Support'
                    : 'Unsupported Message'}
                </span>
              )}
            </div>

            {!translationResult ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-muted)', textAlign: 'center', padding: '2rem' }}>
                <Video size={48} style={{ opacity: 0.4, marginBottom: '1rem' }} />
                <p style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--color-text-secondary)', marginBottom: '0.25rem' }}>
                  No German message entered yet
                </p>
                <p style={{ fontSize: '0.82rem', maxWidth: '300px' }}>
                  Type a German sentence or click one of the quick phrases on the left to see verified DGS signs.
                </p>
              </div>
            ) : translationResult.match_type === 'unsupported' ? (
              /* Unsupported Phrase State */
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '1rem' }}>
                <div
                  style={{
                    background: 'var(--color-warning-bg)',
                    border: '1px solid rgba(217, 119, 6, 0.25)',
                    borderRadius: 'var(--radius-md)',
                    padding: '1.25rem',
                    textAlign: 'center',
                  }}
                >
                  <AlertCircle size={32} color="var(--color-warning)" style={{ margin: '0 auto 0.75rem' }} />
                  <h4 style={{ fontSize: '1rem', color: 'var(--color-text-main)', marginBottom: '0.5rem' }}>
                    No verified DGS sequence available
                  </h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
                    {translationResult.unsupported_message}
                  </p>
                  <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
                    <button className="btn btn-secondary btn-sm" onClick={handleSpeakGerman}>
                      <Volume2 size={14} />
                      <span>Speak German</span>
                    </button>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleCopyText(translationResult.original_text)}
                    >
                      {copied ? <Check size={14} /> : <Copy size={14} />}
                      <span>{copied ? 'Copied' : 'Copy Text'}</span>
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              /* Verified Sign Output Display */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {/* DGS Gloss Sequence Chips */}
                <div>
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
                    DGS Gloss Sequence:
                  </span>
                  <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.35rem' }}>
                    {translationResult.dgs_gloss_sequence.map((gloss, idx) => (
                      <span
                        key={`${gloss}-${idx}`}
                        style={{
                          padding: '3px 10px',
                          background: 'var(--color-primary-light)',
                          color: 'var(--color-primary)',
                          borderRadius: 'var(--radius-full)',
                          fontWeight: 700,
                          fontSize: '0.82rem',
                          border: '1px solid rgba(37, 99, 235, 0.2)',
                        }}
                      >
                        {gloss}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Multi-Clip Navigation if several clips */}
                {translationResult.clips.length > 1 && (
                  <div style={{ display: 'flex', gap: '0.4rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                    {translationResult.clips.map((clip, idx) => (
                      <button
                        key={clip.clip_id}
                        className={`btn btn-sm ${activeClipIndex === idx ? 'btn-primary' : 'btn-secondary'}`}
                        style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}
                        onClick={() => setActiveClipIndex(idx)}
                      >
                        Sign {idx + 1}: {clip.german_phrase}
                      </button>
                    ))}
                  </div>
                )}

                {/* Sign Demonstration Display Card */}
                {currentClip && (
                  <div
                    style={{
                      background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
                      borderRadius: 'var(--radius-lg)',
                      padding: '1.25rem',
                      color: '#ffffff',
                      boxShadow: 'var(--shadow-md)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <span style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>
                          Verified DGS Sign
                        </span>
                        <h4 style={{ fontSize: '1.5rem', color: '#38bdf8', margin: '0.2rem 0' }}>
                          {currentClip.german_phrase}
                        </h4>
                        <div style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>
                          Meaning: {currentClip.english_meaning || currentClip.canonical_meaning}
                        </div>
                      </div>

                      <span
                        style={{
                          fontSize: '0.65rem',
                          fontWeight: 700,
                          background: 'rgba(34, 197, 94, 0.2)',
                          color: '#4ade80',
                          padding: '2px 8px',
                          borderRadius: 'var(--radius-full)',
                          border: '1px solid rgba(74, 222, 128, 0.3)',
                        }}
                      >
                        VERIFIED DGS
                      </span>
                    </div>

                    {/* Signing Execution Description */}
                    {currentClip.verification_notes && (
                      <div
                        style={{
                          marginTop: '0.85rem',
                          padding: '0.6rem 0.85rem',
                          background: 'rgba(255, 255, 255, 0.08)',
                          borderRadius: 'var(--radius-md)',
                          fontSize: '0.82rem',
                          color: '#e2e8f0',
                          lineHeight: 1.45,
                        }}
                      >
                        <strong>Signing form:</strong> {currentClip.verification_notes}
                      </div>
                    )}

                    {/* Research Attribution & Licence */}
                    <div
                      style={{
                        marginTop: '0.85rem',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '0.72rem',
                        color: '#94a3b8',
                        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                        paddingTop: '0.6rem',
                        flexWrap: 'wrap',
                        gap: '0.5rem',
                      }}
                    >
                      <span>Source: {currentClip.source_attribution}</span>
                      <span>Licence: {currentClip.licence_status}</span>
                    </div>
                  </div>
                )}

                {/* Linguistic Explanation */}
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)', background: 'var(--color-surface-muted)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-md)' }}>
                  <Info size={16} color="var(--color-primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>{translationResult.explanation}</div>
                </div>

                {/* Action buttons */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto', paddingTop: '0.5rem' }}>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="btn btn-secondary btn-sm" onClick={handleSpeakGerman} title="Speak German message">
                      <Volume2 size={14} />
                      <span>Speak</span>
                    </button>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleCopyText(translationResult.original_text)}
                      title="Copy German text"
                    >
                      {copied ? <Check size={14} /> : <Copy size={14} />}
                      <span>{copied ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>

                  {onAddToConversation && (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={handleAddConversation}
                      title="Post this message to Two-Way Conversation"
                    >
                      <MessageSquarePlus size={14} />
                      <span>Post to Conversation</span>
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
