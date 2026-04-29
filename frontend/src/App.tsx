import React, { useState, useEffect, useRef } from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts';
import { 
  Activity, Heart, Wind, ShieldCheck, Zap, Upload, FileVideo, AlertCircle 
} from 'lucide-react';

// Types for our data
interface ChunkData {
  type: 'chunk_update';
  chunk_index: number;
  start_time: number;
  end_time: number;
  bpm: number | null;
  sqi: number | null;
  respiratory_rate: number | null;
  latency_ms: number;
  processing_speed: number;
}

interface FinalResult {
  type: 'final_result';
  overall_bpm: number | null;
  overall_sqi: number | null;
  overall_respiratory_rate: number | null;
  total_processing_time_sec: number;
  average_latency_ms: number;
  video_duration_sec: number;
}

function App() {
  const [chunks, setChunks] = useState<ChunkData[]>([]);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'done' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  
  const wsRef = useRef<WebSocket | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus('uploading');
    setError(null);
    setChunks([]);
    setFinalResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      connectWebSocket(data.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setStatus('error');
    }
  };

  const connectWebSocket = (id: string) => {
    setStatus('processing');
    const ws = new WebSocket(`ws://localhost:8000/ws/process/${id}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'chunk_update') {
        setChunks(prev => [...prev, data]);
        const totalExpected = 12; 
        setProgress(Math.min(((data.chunk_index + 1) / totalExpected) * 100, 95));
      } else if (data.type === 'final_result') {
        setFinalResult(data);
        setStatus('done');
        setProgress(100);
        ws.close();
      } else if (data.error) {
        setError(data.error);
        setStatus('error');
        ws.close();
      }
    };

    ws.onerror = () => {
      setError('Connection error');
      setStatus('error');
    };
  };

  const latestChunk = chunks.length > 0 ? chunks[chunks.length - 1] : null;

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo">
          <Activity size={36} color="#007c89" strokeWidth={2.5} />
          <div>
            <span>VITALIS</span>
            <div className="sub">rPPG DIAGNOSTICS</div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
          {status === 'processing' && (
            <div className="pulse-indicator">
              <div className="dot"></div>
              LIVE STREAM ANALYSIS
            </div>
          )}
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
            ENV: PRODUCTION v2.1
          </div>
        </div>
      </header>

      <main className="main-grid">
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Source Panel */}
          <div className="card">
            <h3 className="card-title"><FileVideo size={18} /> Patient Source</h3>
            <div className="upload-zone" onClick={() => document.getElementById('video-input')?.click()}>
              <input 
                id="video-input" 
                type="file" 
                accept="video/*" 
                hidden 
                onChange={handleFileUpload}
              />
              {status === 'idle' || status === 'error' ? (
                <>
                  <Upload className="upload-icon" />
                  <p style={{ textAlign: 'center', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                    Select Face Video<br/>
                    <small>MP4, AVI, MKV (Max 60s)</small>
                  </p>
                </>
              ) : (
                <div style={{ width: '100%', padding: '0 1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)' }}>
                      {status === 'uploading' ? 'UPLOADING' : 'ANALYZING'}
                    </span>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700 }}>{Math.round(progress)}%</span>
                  </div>
                  <div style={{ height: '3px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', background: 'var(--primary)', width: `${progress}%`, transition: 'width 0.4s' }}></div>
                  </div>
                </div>
              )}
            </div>
            {error && (
              <div style={{ marginTop: '1.5rem', color: '#d63031', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertCircle size={16} /> {error}
              </div>
            )}
          </div>

          {/* Diagnostics Panel */}
          <div className="card">
            <h3 className="card-title"><Zap size={18} /> System Diagnostics</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', fontSize: '0.9rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Engine Latency</span>
                <span style={{ fontWeight: 600 }}>{latestChunk?.latency_ms.toFixed(0) || 0} ms</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Throughput</span>
                <span style={{ fontWeight: 600 }}>{(latestChunk?.processing_speed || 0).toFixed(1)}x</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Analysis Time</span>
                <span style={{ fontWeight: 600 }}>{finalResult?.total_processing_time_sec.toFixed(1) || '—'} s</span>
              </div>
            </div>
          </div>
        </aside>

        <section>
          {/* Vitals Summary */}
          <div className="stats-grid">
            <div className="card stat-card">
              <div className="stat-label">HEART RATE</div>
              <div className="stat-value">
                {status === 'processing' ? latestChunk?.bpm?.toFixed(1) : finalResult?.overall_bpm?.toFixed(1) || '—'}
                <span className="stat-unit">BPM</span>
              </div>
            </div>
            <div className="card stat-card">
              <div className="stat-label">RESPIRATORY</div>
              <div className="stat-value">
                {status === 'processing' ? latestChunk?.respiratory_rate?.toFixed(1) : finalResult?.overall_respiratory_rate?.toFixed(1) || '—'}
                <span className="stat-unit">RPM</span>
              </div>
            </div>
            <div className="card stat-card">
              <div className="stat-label">SIGNAL QUALITY</div>
              <div className="stat-value">
                {((status === 'processing' ? latestChunk?.sqi : finalResult?.overall_sqi) || 0).toFixed(2)}
                <span className="stat-unit">SQI</span>
              </div>
            </div>
          </div>

          {/* Chart Section */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <h3 className="card-title">Heart Rate Variability Trend</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chunks}>
                  <defs>
                    <linearGradient id="colorBpm" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#007c89" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#007c89" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="#f0f0f0" vertical={false} />
                  <XAxis 
                    dataKey="end_time" 
                    stroke="#94a3b8" 
                    tickFormatter={(val) => `${val}s`}
                    fontSize={11}
                    tickMargin={10}
                  />
                  <YAxis 
                    domain={['dataMin - 10', 'dataMax + 10']} 
                    stroke="#94a3b8" 
                    fontSize={11}
                    tickMargin={10}
                    tickFormatter={(val) => val.toFixed(1)}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #edf2f2',
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="bpm" 
                    stroke="#007c89" 
                    strokeWidth={3}
                    fillOpacity={1} 
                    fill="url(#colorBpm)" 
                    name="BPM"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* History Section */}
          <div className="card">
            <h3 className="card-title" style={{ justifyContent: 'space-between' }}>
              Detailed Window Analysis
              {finalResult && (
                <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--primary)', background: 'var(--primary-light)', padding: '4px 10px', borderRadius: '4px' }}>
                  AGGREGATION: MEDIAN
                </span>
              )}
            </h3>
            <table style={{ marginTop: '1rem' }}>
              <thead>
                <tr>
                  <th>TIME WINDOW</th>
                  <th>HEART RATE</th>
                  <th>RESPIRATORY</th>
                  <th>SIGNAL SQI</th>
                  <th>EFFICIENCY</th>
                </tr>
              </thead>
              <tbody>
                {chunks.map((chunk, i) => (
                  <tr key={i}>
                    <td>{chunk.start_time.toFixed(0)} – {chunk.end_time.toFixed(0)}s</td>
                    <td className="row-highlight">
                      {chunk.bpm?.toFixed(1) || '0.0'} <small>BPM</small>
                    </td>
                    <td>{chunk.respiratory_rate?.toFixed(1) || '0.0'} <small>RPM</small></td>
                    <td>{chunk.sqi?.toFixed(2) || '0.00'}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{chunk.processing_speed.toFixed(1)}x</td>
                  </tr>
                ))}
                {chunks.length === 0 && (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }}>
                      No data windows captured. Begin analysis to populate results.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
