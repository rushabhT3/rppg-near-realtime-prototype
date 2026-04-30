import React from 'react';
import { Upload, FileVideo, AlertCircle } from 'lucide-react';
import type { AppStatus } from '../../types';

interface UploadSectionProps {
  status: AppStatus;
  progress: number;
  error: string | null;
  onUpload: (file: File) => void;
}

export const UploadSection: React.FC<UploadSectionProps> = ({ status, progress, error, onUpload }) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="card">
      <h3 className="card-title">
        <FileVideo size={20} /> Input Source
      </h3>
      <div className="upload-zone" onClick={() => document.getElementById('video-input')?.click()}>
        <input 
          id="video-input" 
          type="file" 
          accept="video/*" 
          hidden 
          onChange={handleFileChange}
        />
        {status === 'idle' || status === 'error' ? (
          <>
            <Upload className="upload-icon" />
            <p style={{ textAlign: 'center' }}>Drag & Drop Video<br/><span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>MP4, AVI, or MKV (Max 60s)</span></p>
          </>
        ) : (
          <div style={{ width: '100%', padding: '0 2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-main)' }}>{status === 'uploading' ? 'Uploading...' : 'Processing...'}</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--primary)' }}>{Math.round(progress)}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
          </div>
        )}
      </div>
      {error && (
        <div style={{ marginTop: '1rem', color: '#d63031', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};
