import React, { useState, useEffect } from 'react';
import {
  Search,
  Filter,
  BookOpen,
  AlertCircle,
  Volume2,
  ExternalLink,
  ShieldAlert,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetManifestItem } from '../types/signs';
import { speechService } from '../services/speech';

export const SignDictionaryView: React.FC = () => {
  const [manifestItems, setManifestItems] = useState<DatasetManifestItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<
    'all' | 'trained' | 'verified_clips' | 'phrase_mapping' | 'data_required'
  >('all');

  useEffect(() => {
    api
      .getDatasetManifest()
      .then((res) => {
        setManifestItems(res.items);
      })
      .catch((err) => {
        console.error('Failed to fetch manifest dictionary:', err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const filteredItems = manifestItems.filter((item) => {
    const q = searchQuery.toLowerCase();
    const matchesSearch =
      item.canonical_meaning.toLowerCase().includes(q) ||
      item.german_text.toLowerCase().includes(q) ||
      item.english_text.toLowerCase().includes(q) ||
      item.source_gloss.toLowerCase().includes(q) ||
      item.category.toLowerCase().includes(q);

    if (filterType === 'trained') {
      return matchesSearch && item.model_support_status === 'trained_model_class';
    }
    if (filterType === 'verified_clips') {
      return matchesSearch && item.verification_status === 'verified_research_clip';
    }
    if (filterType === 'phrase_mapping') {
      return matchesSearch && item.model_support_status === 'verified_phrase_mapping';
    }
    if (filterType === 'data_required') {
      return matchesSearch && item.verification_status === 'data_required';
    }
    return matchesSearch;
  });

  return (
    <div style={{ maxWidth: '1080px', margin: '0 auto' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <BookOpen size={24} color="var(--color-primary)" />
          <h2 style={{ fontSize: '1.5rem' }}>Everyday DGS Sign Vocabulary & Reference Dictionary</h2>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
          Explore canonical German Sign Language vocabulary across greetings, politeness, basic communication, and interactions. Transparently displays model support, corpus source, and licensing status.
        </p>
      </div>

      {/* Cultural & Linguistic Banner */}
      <div
        style={{
          background: 'var(--color-accent-violet-light)',
          border: '1px solid rgba(124, 58, 237, 0.2)',
          borderRadius: 'var(--radius-md)',
          padding: '0.85rem 1rem',
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-main)',
        }}
      >
        <AlertCircle size={18} color="var(--color-accent-violet)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Linguistic Integrity & Regional Dialects:</strong> German Sign Language (DGS) is an autonomous visual-spatial language with rich regional dialects across Germany. Glosses represent written labels, not word-for-word spoken German equivalents. Entries are verified against the Public DGS Corpus and Meine DGS portal.
        </div>
      </div>

      {/* Search and Filters */}
      <div
        className="dgs-card"
        style={{
          padding: '1rem',
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ position: 'relative', flex: '1 1 280px' }}>
          <Search
            size={18}
            style={{
              position: 'absolute',
              left: '0.75rem',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--color-text-muted)',
            }}
          />
          <input
            type="text"
            placeholder="Search by German phrase, English meaning, or DGS gloss..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: '100%',
              padding: '0.6rem 0.75rem 0.6rem 2.25rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--color-border)',
              fontSize: '0.875rem',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <Filter size={16} color="var(--color-text-muted)" />
          <button
            className={`btn btn-sm ${filterType === 'all' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilterType('all')}
          >
            All ({manifestItems.length})
          </button>
          <button
            className={`btn btn-sm ${filterType === 'trained' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilterType('trained')}
          >
            Trained ({manifestItems.filter((i) => i.model_support_status === 'trained_model_class').length})
          </button>
          <button
            className={`btn btn-sm ${filterType === 'verified_clips' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilterType('verified_clips')}
          >
            Verified Clips ({manifestItems.filter((i) => i.verification_status === 'verified_research_clip').length})
          </button>
          <button
            className={`btn btn-sm ${filterType === 'phrase_mapping' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilterType('phrase_mapping')}
          >
            Phrase Mappings ({manifestItems.filter((i) => i.model_support_status === 'verified_phrase_mapping').length})
          </button>
          <button
            className={`btn btn-sm ${filterType === 'data_required' ? 'btn-danger' : 'btn-secondary'}`}
            onClick={() => setFilterType('data_required')}
          >
            Data Required ({manifestItems.filter((i) => i.verification_status === 'data_required').length})
          </button>
        </div>
      </div>

      {/* Cards Grid */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-text-muted)' }}>
          Loading everyday vocabulary manifest...
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="dgs-card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-text-muted)' }}>
          No vocabulary entries matched your search query.
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: '1.25rem',
          }}
        >
          {filteredItems.map((item) => {
            const isTrained = item.model_support_status === 'trained_model_class';
            const isDataReq = item.verification_status === 'data_required';
            const isPhrase = item.model_support_status === 'verified_phrase_mapping';

            return (
              <div
                key={item.phrase_id}
                className="dgs-card"
                style={{
                  borderLeft: isTrained
                    ? '4px solid var(--color-primary)'
                    : isDataReq
                    ? '4px solid var(--color-error)'
                    : '4px solid var(--color-accent-violet)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
                    {item.category}
                  </span>

                  <span
                    className={`badge-tag ${
                      isTrained
                        ? 'badge-model-supported'
                        : isDataReq
                        ? 'badge-dictionary-only'
                        : isPhrase
                        ? 'badge-verified'
                        : 'badge-dictionary-only'
                    }`}
                    style={isDataReq ? { background: 'var(--color-error-bg)', color: 'var(--color-error)' } : {}}
                  >
                    {isTrained
                      ? 'Trained Model Class'
                      : isDataReq
                      ? 'DATA REQUIRED'
                      : isPhrase
                      ? 'Verified Phrase Mapping'
                      : 'Dictionary Reference'}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <h3 style={{ fontSize: '1.3rem', color: 'var(--color-primary)' }}>
                    {item.german_text}
                  </h3>
                  <button
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-text-muted)' }}
                    onClick={() => speechService.speakGerman(item.german_text)}
                    title={`Pronounce "${item.german_text}"`}
                  >
                    <Volume2 size={16} />
                  </button>
                </div>

                <div style={{ fontSize: '0.88rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
                  EN: {item.english_text}
                </div>

                {/* DGS Gloss */}
                <div style={{ marginBottom: '0.6rem' }}>
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
                    DGS Gloss:
                  </span>
                  <div style={{ display: 'inline-block', marginLeft: '0.5rem' }}>
                    <code style={{ background: 'var(--color-surface-muted)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-text-main)' }}>
                      {item.source_gloss}
                    </code>
                  </div>
                </div>

                {/* Licence & Source Notes */}
                <div
                  style={{
                    background: 'var(--color-surface-muted)',
                    padding: '0.5rem 0.75rem',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.75rem',
                    color: 'var(--color-text-secondary)',
                    lineHeight: 1.4,
                  }}
                >
                  <p><strong>Status:</strong> {item.licence_notes}</p>
                  {item.source_url && (
                    <a
                      href={item.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '3px',
                        color: 'var(--color-primary)',
                        marginTop: '4px',
                        fontSize: '0.72rem',
                        textDecoration: 'none',
                      }}
                    >
                      <span>Research Source</span>
                      <ExternalLink size={10} />
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
