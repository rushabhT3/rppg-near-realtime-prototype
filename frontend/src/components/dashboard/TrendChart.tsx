import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { ChunkData } from '../../types';

interface TrendChartProps {
  data: ChunkData[];
}

export const TrendChart: React.FC<TrendChartProps> = ({ data }) => {
  return (
    <div className="card" style={{ height: '500px' }}>
      <h3 className="card-title">Biometric Trend Analysis</h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorBpm" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--primary)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
            <XAxis 
              dataKey="end_time" 
              stroke="var(--text-muted)" 
              tickFormatter={(val) => `${val}s`}
              fontSize={12}
            />
            <YAxis domain={['dataMin - 5', 'dataMax + 5']} stroke="var(--text-muted)" fontSize={12} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'var(--bg-section)', 
                borderColor: 'var(--border)',
                borderRadius: '8px',
                color: 'var(--text-main)'
              }}
            />
            <Area 
              type="monotone" 
              dataKey="bpm" 
              stroke="var(--primary)" 
              strokeWidth={3}
              fillOpacity={1} 
              fill="url(#colorBpm)" 
              name="Heart Rate"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
