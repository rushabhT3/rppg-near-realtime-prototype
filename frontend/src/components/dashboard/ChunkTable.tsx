import React from 'react';
import { Activity } from 'lucide-react';
import type { ChunkData } from '../../types';

interface ChunkTableProps {
  chunks: ChunkData[];
}

export const ChunkTable: React.FC<ChunkTableProps> = ({ chunks }) => {
  if (chunks.length === 0) {
    return (
      <div className="card">
        <h3 className="card-title">
          <Activity size={20} /> Chunk Analysis
        </h3>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '2rem' }}>
          Waiting for video processing...
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="card-title">
        <Activity size={20} /> Chunk Analysis
      </h3>
      <div style={{ overflowX: 'auto' }}>
        <table className="chunk-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Time Range</th>
              <th>HR (BPM)</th>
              <th>Resp (Br/m)</th>
              <th>SQI</th>
            </tr>
          </thead>
          <tbody>
            {[...chunks].reverse().map((chunk) => (
              <tr key={chunk.chunk_index}>
                <td style={{ fontWeight: 600, color: 'var(--primary)' }}>
                  {chunk.chunk_index + 1}
                </td>
                <td>{chunk.start_time.toFixed(1)}s – {chunk.end_time.toFixed(1)}s</td>
                <td><span className="row-highlight">{chunk.bpm?.toFixed(1) ?? 'N/A'}</span></td>
                <td><span className="row-highlight">{chunk.respiratory_rate?.toFixed(1) ?? 'N/A'}</span></td>
                <td><span className="row-highlight">{chunk.sqi?.toFixed(2) ?? 'N/A'}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
