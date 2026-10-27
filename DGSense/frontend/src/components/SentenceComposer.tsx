import React, { useState } from 'react';
import {
  Volume2,
  Square,
  Undo2,
  Trash2,
  Copy,
  Check,
  Sparkles,
  Info,
  X,
  MessageSquarePlus,
} from 'lucide-react';
import { ComposeSentenceResponse } from '../types/signs';
import { speechService } from '../services/speech';

interface SentenceComposerProps {
  signs: string[];
  composition: ComposeSentenceResponse | null;
  onRemoveSign: (index: number) => void;
  onUndoLastSign: () => void;
  onClearSentence: () => void;
  onPostToConversation?: (germanText: string, englishText: string, glosses: string[]) => void;
}

export const SentenceComposer: React.FC<SentenceComposerProps> = ({
  signs,
  composition,
  onRemoveSign,
  onUndoLastSign,
  onClearSentence,
  onPostToConversation,
}) => {
  const [copiedType, setCopiedType] = useState<'de' | 'en' | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handleCopy = (text: string, type: 'de' | 'en') => {
    navigator.clipboard.writeText(text);
    setCopiedType(type);
    setTimeout(() => setCopiedType(null), 2000);
  };

  const handleSpeakGerman = async () => {
    if (!composition?.german_sentence) return;
    setIsSpeaking(true);
    await speechService.speakGerman(composition.german_sentence);
    setIsSpeaking(false);
  };

  const handleSpeakEnglish = async () => {
    if (!composition?.english_sentence) return;
    setIsSpeaking(true);
    await speechService.speakEnglish(composition.english_sentence);
    setIsSpeaking(false);
  };

  const handleStopSpeech = () => {
    speechService.stop();
    setIsSpeaking(false);
  };

  const isVerified = composition?.composition_type === 'verified_phrase';

  return (
    <div className="dgs-card" style={{ marginTop: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sparkles size={18} color="var(--color-accent-violet)" />
          <h3 style={{ fontSize: '1.1rem' }}>Sentence Composer</h3>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn btn-secondary btn-sm"
            onClick={onUndoLastSign}
            disabled={signs.length === 0}
            title="Undo last sign"
          >
            <Undo2 size={14} />
            <span>Undo</span>
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={onClearSentence}
            disabled={signs.length === 0}
            title="Clear all signs"
          >
            <Trash2 size={14} />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Sequence Chips */}
      <div className="composer-chip-list">
        {signs.length === 0 ? (
          <span style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>
            No signs added yet. Recognize a sign or click "Add Sign" to compose a sentence.
          </span>
        ) : (
          signs.map((sign, idx) => (
            <span key={`${sign}-${idx}`} className="sign-chip">
              <span>{sign}</span>
              <button
                className="sign-chip-remove"
                onClick={() => onRemoveSign(idx)}
                title={`Remove ${sign}`}
                aria-label={`Remove ${sign}`}
              >
                <X size={12} />
              </button>
            </span>
          ))
        )}
      </div>

      {/* Output Results */}
      {signs.length > 0 && composition && (
        <div className="sentence-result-card">
          <div className="sentence-header">
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              Composed Output
            </span>
            <span
              className={`badge-tag ${
                isVerified ? 'badge-verified' : 'badge-approx'
              }`}
            >
              {isVerified ? 'Verified DGS Phrase' : 'Approximate Composition'}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.75rem' }}>
            {/* German Text */}
            <div style={{ background: 'var(--color-surface-muted)', padding: '0.85rem', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-secondary)' }}>
                  GERMAN (DEUTSCH)
                </span>
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '2px 6px', fontSize: '0.7rem' }}
                  onClick={() => handleCopy(composition.german_sentence, 'de')}
                  title="Copy German text"
                >
                  {copiedType === 'de' ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <p style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--color-text-main)' }}>
                {composition.german_sentence || '—'}
              </p>
            </div>

            {/* English Text */}
            <div style={{ background: 'var(--color-surface-muted)', padding: '0.85rem', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-secondary)' }}>
                  ENGLISH MEANING
                </span>
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '2px 6px', fontSize: '0.7rem' }}
                  onClick={() => handleCopy(composition.english_sentence, 'en')}
                  title="Copy English text"
                >
                  {copiedType === 'en' ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <p style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--color-text-main)' }}>
                {composition.english_sentence || '—'}
              </p>
            </div>
          </div>

          {/* Speech Controls */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleSpeakGerman}
                disabled={!composition.german_sentence}
                title="Speak German sentence using Web Speech API"
              >
                <Volume2 size={15} />
                <span>Speak German</span>
              </button>

              <button
                className="btn btn-secondary btn-sm"
                onClick={handleSpeakEnglish}
                disabled={!composition.english_sentence}
                title="Speak English translation"
              >
                <Volume2 size={15} />
                <span>Speak English</span>
              </button>

              {isSpeaking && (
                <button
                  className="btn btn-danger btn-sm"
                  onClick={handleStopSpeech}
                  title="Stop speech"
                >
                  <Square size={14} />
                  <span>Stop</span>
                </button>
              )}
            </div>

            {composition.notes && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                <Info size={14} />
                <span>{composition.notes}</span>
              </div>
            )}

            {onPostToConversation && (
              <button
                className="btn btn-primary btn-sm"
                onClick={() =>
                  onPostToConversation(
                    composition.german_sentence,
                    composition.english_sentence,
                    composition.gloss_sequence
                  )
                }
                title="Post translated sentence to Two-Way Conversation"
              >
                <MessageSquarePlus size={14} />
                <span>Post to Conversation</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
