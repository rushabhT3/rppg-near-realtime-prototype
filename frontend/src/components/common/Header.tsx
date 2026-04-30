import React from 'react';
import { Activity } from 'lucide-react';

interface HeaderProps {
  isLive: boolean;
}

export const Header: React.FC<HeaderProps> = ({ isLive }) => {
  return (
    <header className="header">
      <div className="logo">
        <Activity size={32} />
        <div>
          <span>VITALIS <span style={{ fontWeight: 300, opacity: 0.6 }}>rPPG</span></span>
          <div className="sub">Physiological Monitoring</div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
        {isLive && (
          <div className="pulse-indicator">
            <div className="dot"></div>
            LIVE ANALYSIS
          </div>
        )}
        <div className="card" style={{ padding: '0.5rem 1rem', borderRadius: '8px' }}>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>VITALIS-CORE-X</span>
        </div>
      </div>
    </header>
  );
};
