import React from 'react';
import { Heart, Wind, ShieldCheck, Zap } from 'lucide-react';

// Modules
import { Header } from './components/common/Header';
import { MetricCard } from './components/dashboard/MetricCard';
import { TrendChart } from './components/dashboard/TrendChart';
import { UploadSection } from './components/upload/UploadSection';
import { FinalSummary } from './components/dashboard/FinalSummary';
import { ChunkTable } from './components/dashboard/ChunkTable';

// Hooks
import { useRPPGSession } from './hooks/useRPPGSession';

// Styles
import './styles/index.css';

const App: React.FC = () => {
  const { 
    chunks, 
    finalResult, 
    status, 
    error, 
    progress, 
    uploadVideo 
  } = useRPPGSession();

  const isProcessing = status === 'processing';
  const latestChunk = chunks[chunks.length - 1];

  const displayBPM = isProcessing ? latestChunk?.bpm : finalResult?.overall_bpm;
  const displayResp = isProcessing ? latestChunk?.respiratory_rate : finalResult?.overall_respiratory_rate;
  const displaySQI = (isProcessing ? latestChunk?.sqi : finalResult?.overall_sqi) ?? 0;

  return (
    <div className="app-container">
      <Header isLive={isProcessing} />

      <main className="main-grid">
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <UploadSection 
            status={status} 
            progress={progress} 
            error={error} 
            onUpload={uploadVideo} 
          />

          {/* Performance Panel */}
          <div className="card">
            <h3 className="card-title">
              <Zap size={20} /> Performance
            </h3>
            <table>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', width: '45%' }}>Value</th>
                  <th style={{ textAlign: 'left' }}>Parameter</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="row-highlight">{finalResult?.total_processing_time_sec.toFixed(1) || '—'} <small>s</small></td>
                  <td>Total Time</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Final Analysis Summary */}
          <FinalSummary data={finalResult} />
        </aside>

        <section>
          {/* Main Stats Grid */}
          <div className="stats-grid">
            <MetricCard 
              label="Heart Rate" 
              value={displayBPM?.toFixed(1)} 
              unit="BPM" 
              icon={<Heart size={16} />} 
            />
            <MetricCard 
              label="Respiratory Rate" 
              value={displayResp?.toFixed(1)} 
              unit="Br/m" 
              icon={<Wind size={16} />} 
            />
            <MetricCard 
              label="Signal Quality" 
              value={displaySQI.toFixed(2)} 
              unit="SQI" 
              icon={<ShieldCheck size={16} />} 
            />
          </div>

          {/* Average Rating & Chart Section */}
          <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
            <div>
              <h2 style={{ fontFamily: 'var(--font-heading)', color: 'var(--secondary)', marginBottom: '0.5rem' }}>Biometric Vitality Trend</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Real-time physiological oscillation monitoring</p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Signal Health Rating</div>
              <div style={{ fontSize: '2rem', fontWeight: 300, color: 'var(--secondary)' }}>
                {status === 'idle' ? '2.5' : (displaySQI * 5).toFixed(1)} 
                <span style={{ fontSize: '1rem', color: 'var(--text-muted)', marginLeft: '4px' }}>/ 5.0</span>
              </div>
            </div>
          </div>

          <TrendChart data={chunks} />

          <ChunkTable chunks={chunks} />
        </section>
      </main>
    </div>
  );
};

export default App;
