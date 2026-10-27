import React, { useState, useEffect, useCallback } from 'react';
import { Header, NavTabType } from './components/Header';
import { LiveTranslatorView } from './pages/LiveTranslatorView';
import { ReverseTranslatorView } from './pages/ReverseTranslatorView';
import { ConversationView } from './pages/ConversationView';
import { SignDictionaryView } from './pages/SignDictionaryView';
import { PracticeModeView } from './pages/PracticeModeView';
import { SignRecorderView } from './pages/SignRecorderView';
import { ModelStatusView } from './pages/ModelStatusView';
import { InstructionsModal } from './components/InstructionsModal';
import { api } from './services/api';
import { HealthResponse, ConversationMessage, SignClipItem } from './types/signs';

export function App() {
  const [activeTab, setActiveTab] = useState<NavTabType>('sign_to_text');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  // Two-way conversation messages state (persisted server-side in SQLite)
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await api.getHealth();
      setHealth(res);
    } catch (err) {
      console.warn('Backend health check error:', err);
    }
  }, []);

  // Restore persisted conversation on mount
  useEffect(() => {
    api
      .getConversations()
      .then((res) => {
        setConversation(
          res.messages.map((m) => ({
            id: `msg-${m.id}`,
            speaker: m.speaker,
            speaker_name: m.speaker_name || (m.speaker === 'dgs_user' ? 'Deaf User (DGS Video)' : 'Hearing User (German Text/Voice)'),
            text: m.text,
            translation: m.translation || undefined,
            glosses: m.glosses,
            match_type: m.match_type || undefined,
            clips: m.clips,
            timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          }))
        );
      })
      .catch(() => {
        /* backend offline: conversation starts empty */
      });
  }, []);

  const persistConversation = useCallback(
    (msg: {
      speaker: 'dgs_user' | 'hearing_user';
      speaker_name: string;
      text: string;
      translation?: string;
      glosses: string[];
      match_type?: string;
      clips?: SignClipItem[];
    }) => {
      api.addConversation({ ...msg, clips: msg.clips || [] }).catch(() => {
        /* non-fatal: message still shows locally */
      });
    },
    []
  );

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  // Handler when Deaf User posts a translated sentence from Direction A
  const handleAddDGSMessage = useCallback(
    (germanText: string, englishText: string, glosses: string[]) => {
      const newMsg: ConversationMessage = {
        id: `msg-dgs-${Date.now()}`,
        speaker: 'dgs_user',
        speaker_name: 'Deaf User (DGS Video)',
        text: germanText,
        translation: englishText,
        glosses: glosses,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setConversation((prev) => [...prev, newMsg]);
      setActiveTab('conversation');
      persistConversation({
        speaker: 'dgs_user',
        speaker_name: newMsg.speaker_name,
        text: newMsg.text,
        translation: newMsg.translation,
        glosses: glosses,
      });
    },
    [persistConversation]
  );

  // Handler when Hearing User posts a reverse-translated message from Direction B
  const handleAddHearingMessage = useCallback(
    (text: string, glosses: string[], clips: SignClipItem[], matchType: string) => {
      const newMsg: ConversationMessage = {
        id: `msg-hearing-${Date.now()}`,
        speaker: 'hearing_user',
        speaker_name: 'Hearing User (German Text/Voice)',
        text: text,
        glosses: glosses,
        match_type: matchType,
        clips: clips,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setConversation((prev) => [...prev, newMsg]);
      setActiveTab('conversation');
      persistConversation({
        speaker: 'hearing_user',
        speaker_name: newMsg.speaker_name,
        text: newMsg.text,
        glosses: glosses,
        match_type: matchType,
        clips: clips,
      });
    },
    [persistConversation]
  );

  // Direct send from Conversation view input bar
  const handleSendHearingFromConversation = useCallback(
    async (text: string) => {
      try {
        const res = await api.reverseTranslate(text);
        const newMsg: ConversationMessage = {
          id: `msg-hearing-${Date.now()}`,
          speaker: 'hearing_user',
          speaker_name: 'Hearing User (German Text/Voice)',
          text: res.original_text,
          glosses: res.dgs_gloss_sequence,
          match_type: res.match_type,
          clips: res.clips,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setConversation((prev) => [...prev, newMsg]);
        persistConversation({
          speaker: 'hearing_user',
          speaker_name: newMsg.speaker_name,
          text: newMsg.text,
          glosses: res.dgs_gloss_sequence,
          match_type: res.match_type,
          clips: res.clips,
        });
      } catch (err) {
        console.error('Failed to translate hearing message:', err);
      }
    },
    [persistConversation]
  );

  const handleClearConversation = useCallback(() => {
    setConversation([]);
    api.clearConversations().catch(() => {
      /* non-fatal */
    });
  }, []);

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        onOpenHelp={() => setIsHelpOpen(true)}
      />

      <main className="main-content">
        {activeTab === 'sign_to_text' && (
          <LiveTranslatorView
            health={health}
            onOpenHelp={() => setIsHelpOpen(true)}
            onAddToConversation={handleAddDGSMessage}
          />
        )}

        {activeTab === 'text_to_sign' && (
          <ReverseTranslatorView onAddToConversation={handleAddHearingMessage} />
        )}

        {activeTab === 'conversation' && (
          <ConversationView
            messages={conversation}
            onClearConversation={handleClearConversation}
            onSendHearingMessage={handleSendHearingFromConversation}
          />
        )}

        {activeTab === 'dictionary' && <SignDictionaryView />}

        {activeTab === 'practice' && <PracticeModeView health={health} />}

        {activeTab === 'recorder' && <SignRecorderView />}

        {activeTab === 'status' && (
          <ModelStatusView health={health} onRefreshHealth={fetchHealth} />
        )}
      </main>

      <InstructionsModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
}

export default App;
