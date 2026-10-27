import React from 'react';
import {
  Video,
  ArrowLeftRight,
  Users,
  BookOpen,
  GraduationCap,
  Disc,
  Activity,
  HelpCircle,
} from 'lucide-react';
import { HealthResponse } from '../types/signs';

export type NavTabType =
  | 'sign_to_text'
  | 'text_to_sign'
  | 'conversation'
  | 'dictionary'
  | 'practice'
  | 'recorder'
  | 'status';

interface HeaderProps {
  activeTab: NavTabType;
  setActiveTab: (tab: NavTabType) => void;
  health: HealthResponse | null;
  onOpenHelp: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  health,
  onOpenHelp,
}) => {
  const isOnline = health !== null && health.status !== 'degraded';
  const isModelReady = health?.model_loaded ?? false;

  return (
    <header className="dgs-header">
      <div className="brand-section">
        <img src="/logo.svg" alt="DGSense Logo" className="brand-logo" />
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <h1 className="brand-title brand-font">DGSense V2</h1>
            <span className="brand-badge">Two-Way DGS System</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
            Real-Time German Sign Language (DGS) ↔ Spoken German
          </p>
        </div>
      </div>

      <nav className="nav-tabs" aria-label="Main Navigation">
        <button
          className={`nav-tab-btn ${activeTab === 'sign_to_text' ? 'active' : ''}`}
          onClick={() => setActiveTab('sign_to_text')}
          title="Direction A: Live Webcam Sign Recognition"
        >
          <Video size={16} />
          <span>Sign → Text</span>
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'text_to_sign' ? 'active' : ''}`}
          onClick={() => setActiveTab('text_to_sign')}
          title="Direction B: Reverse Translation (German Speech/Text -> DGS Signs)"
        >
          <ArrowLeftRight size={16} />
          <span>Text → Sign</span>
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'conversation' ? 'active' : ''}`}
          onClick={() => setActiveTab('conversation')}
          title="Two-Way Dialogue Session"
        >
          <Users size={16} />
          <span>Conversation</span>
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'dictionary' ? 'active' : ''}`}
          onClick={() => setActiveTab('dictionary')}
          title="Everyday DGS Sign Vocabulary & Reference"
        >
          <BookOpen size={16} />
          <span>Dictionary</span>
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'practice' ? 'active' : ''}`}
          onClick={() => setActiveTab('practice')}
          title="Interactive Practice Mode"
        >
          <GraduationCap size={16} />
          <span>Practice</span>
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'recorder' ? 'active' : ''}`}
          onClick={() => setActiveTab('recorder')}
          title="Sign Example Recorder & Dataset Exporter"
        >
          <Disc size={16} />
          <span>Recorder</span>
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'status' ? 'active' : ''}`}
          onClick={() => setActiveTab('status')}
          title="System Diagnostics, Dataset Manifest & Architecture"
        >
          <Activity size={16} />
          <span>Status</span>
        </button>
      </nav>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div
          className={`header-status-pill ${
            isModelReady ? 'status-ok' : 'status-degraded'
          }`}
          title={health?.details || 'System status'}
        >
          <span className="status-indicator-dot" />
          <span>
            {isModelReady
              ? `Model v${health?.model_version || '1.0'}`
              : isOnline
              ? 'Inspection Mode (Two-Way Active)'
              : 'Backend Offline'}
          </span>
        </div>

        <button
          className="btn btn-secondary btn-sm"
          onClick={onOpenHelp}
          title="Open User Guide & Instructions"
          style={{ padding: '0.4rem 0.6rem' }}
        >
          <HelpCircle size={16} />
          <span style={{ display: 'none' }}>Help</span>
        </button>
      </div>
    </header>
  );
};
