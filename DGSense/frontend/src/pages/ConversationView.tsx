import React, { useState } from 'react';
import {
  MessageSquare,
  Trash2,
  Copy,
  Check,
  Volume2,
  Video,
  Mic,
  Send,
  Users,
} from 'lucide-react';
import { ConversationMessage } from '../types/signs';
import { speechService } from '../services/speech';

interface ConversationViewProps {
  messages: ConversationMessage[];
  onClearConversation: () => void;
  onSendHearingMessage: (text: string) => Promise<void>;
}

export const ConversationView: React.FC<ConversationViewProps> = ({
  messages,
  onClearConversation,
  onSendHearingMessage,
}) => {
  const [quickInput, setQuickInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleSend = async () => {
    if (!quickInput.trim() || isSending) return;
    setIsSending(true);
    try {
      await onSendHearingMessage(quickInput);
      setQuickInput('');
    } finally {
      setIsSending(false);
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSpeak = (text: string) => {
    speechService.speakGerman(text);
  };

  return (
    <div style={{ maxWidth: '960px', margin: '0 auto' }}>
      {/* Title & Actions */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <Users size={24} color="var(--color-primary)" />
            <h2 style={{ fontSize: '1.5rem' }}>Two-Way Accessible Conversation</h2>
          </div>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
            Direct dialogue between a Deaf DGS Signer (Direction A) and a Hearing German Speaker (Direction B).
          </p>
        </div>

        {messages.length > 0 && (
          <button
            className="btn btn-secondary btn-sm"
            onClick={onClearConversation}
            title="Clear conversation history"
          >
            <Trash2 size={14} />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Chat Messages Stream */}
      <div
        className="dgs-card"
        style={{
          minHeight: '440px',
          maxHeight: '560px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          padding: '1.25rem',
          background: 'var(--color-bg-base)',
        }}
      >
        {messages.length === 0 ? (
          <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--color-text-muted)', padding: '2rem' }}>
            <MessageSquare size={44} style={{ opacity: 0.4, margin: '0 auto 0.75rem' }} />
            <h4 style={{ color: 'var(--color-text-secondary)', marginBottom: '0.25rem' }}>
              Conversation is empty
            </h4>
            <p style={{ fontSize: '0.85rem', maxWidth: '340px' }}>
              Translate signs on the <strong>Live Translator</strong> tab or submit German messages below to start dialogue.
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            const isDGSUser = msg.speaker === 'dgs_user';
            return (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: isDGSUser ? 'flex-start' : 'flex-end',
                }}
              >
                {/* Speaker Label & Timestamp */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '4px', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                  <span style={{ fontWeight: 700, color: isDGSUser ? 'var(--color-primary)' : 'var(--color-accent-violet)' }}>
                    {msg.speaker_name}
                  </span>
                  <span>•</span>
                  <span>{msg.timestamp}</span>
                </div>

                {/* Message Bubble */}
                <div
                  style={{
                    maxWidth: '82%',
                    background: isDGSUser ? '#ffffff' : 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                    color: isDGSUser ? 'var(--color-text-main)' : '#ffffff',
                    borderRadius: 'var(--radius-lg)',
                    padding: '0.85rem 1.1rem',
                    boxShadow: 'var(--shadow-sm)',
                    border: isDGSUser ? '1px solid var(--color-border)' : 'none',
                  }}
                >
                  {/* Gloss sequence pill tags if from DGS */}
                  {msg.glosses && msg.glosses.length > 0 && (
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                      {msg.glosses.map((g, i) => (
                        <span
                          key={`${g}-${i}`}
                          style={{
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            padding: '1px 6px',
                            borderRadius: 'var(--radius-sm)',
                            background: isDGSUser ? 'var(--color-primary-light)' : 'rgba(255, 255, 255, 0.2)',
                            color: isDGSUser ? 'var(--color-primary)' : '#ffffff',
                          }}
                        >
                          {g}
                        </span>
                      ))}
                    </div>
                  )}

                  <p style={{ fontSize: '1rem', fontWeight: 500, lineHeight: 1.45 }}>
                    {msg.text}
                  </p>

                  {/* Secondary English translation if available */}
                  {msg.translation && (
                    <p style={{ fontSize: '0.82rem', marginTop: '0.35rem', opacity: isDGSUser ? 0.75 : 0.85, fontStyle: 'italic' }}>
                      EN: {msg.translation}
                    </p>
                  )}

                  {/* Message Actions */}
                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', paddingTop: '0.4rem', borderTop: isDGSUser ? '1px solid var(--color-border)' : '1px solid rgba(255,255,255,0.2)' }}>
                    <button
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        color: isDGSUser ? 'var(--color-text-muted)' : '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '2px',
                        fontSize: '0.7rem',
                      }}
                      onClick={() => handleSpeak(msg.text)}
                      title="Speak German message"
                    >
                      <Volume2 size={12} />
                      <span>Speak</span>
                    </button>

                    <button
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        color: isDGSUser ? 'var(--color-text-muted)' : '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '2px',
                        fontSize: '0.7rem',
                      }}
                      onClick={() => handleCopyMessage(msg.id, msg.text)}
                      title="Copy text"
                    >
                      {copiedId === msg.id ? <Check size={12} /> : <Copy size={12} />}
                      <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Quick Input Bar for Hearing User */}
      <div className="dgs-card" style={{ marginTop: '1rem', padding: '0.85rem' }}>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="Type German response from hearing speaker..."
            value={quickInput}
            onChange={(e) => setQuickInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSend();
            }}
            style={{
              flex: 1,
              padding: '0.65rem 0.85rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--color-border)',
              fontSize: '0.95rem',
            }}
          />

          <button
            className="btn btn-primary"
            onClick={handleSend}
            disabled={!quickInput.trim() || isSending}
          >
            <Send size={15} />
            <span>Send</span>
          </button>
        </div>
      </div>
    </div>
  );
};
