import React from 'react';
import { FileCheck } from 'lucide-react';
import type { FinalResult } from '../../types';

interface FinalSummaryProps {
  data: FinalResult | null;
}

export const FinalSummary: React.FC<FinalSummaryProps> = ({ data }) => {
  if (!data) return null;

  const vitalMetrics = [
    { label: 'Overall Heart Rate', value: data.overall_bpm?.toFixed(1) ?? 'N/A', unit: data.overall_bpm ? 'BPM' : '' },
    { label: 'Respiratory Rate', value: data.overall_respiratory_rate?.toFixed(1) ?? 'N/A', unit: data.overall_respiratory_rate ? 'Br/m' : '' },
    { label: 'Signal Quality Index', value: data.overall_sqi?.toFixed(2) ?? 'N/A', unit: data.overall_sqi ? 'SQI' : '' },
  ];

  const processingMetrics = [
    { label: 'Video Duration', value: data.video_duration_sec?.toFixed(1) ?? 'N/A', unit: data.video_duration_sec ? 's' : '' },
    { label: 'Processing Time', value: data.total_processing_time_sec?.toFixed(1) ?? 'N/A', unit: data.total_processing_time_sec ? 's' : '' },
    { label: 'Processing Speed', value: data.total_processing_time_sec ? (data.video_duration_sec / data.total_processing_time_sec).toFixed(2) : 'N/A', unit: data.total_processing_time_sec ? 'x' : '' },
  ];

  const sections = [
    { title: 'Vital Signs', items: vitalMetrics },
    { title: 'Processing Summary', items: processingMetrics },
  ].filter(s => s.items.length > 0);

  return (
    <div className="card">
      <h3 className="card-title">
        <FileCheck size={20} /> Final Analysis Summary
      </h3>
      {sections.map((section) => (
        <div key={section.title} style={{ marginBottom: '1.5rem' }}>
          <div style={{
            fontSize: '0.7rem',
            fontWeight: 700,
            color: 'var(--accent)',
            textTransform: 'uppercase',
            letterSpacing: '0.15em',
            marginBottom: '0.5rem',
            paddingLeft: '0.5rem',
            borderLeft: '3px solid var(--accent)',
          }}>
            {section.title}
          </div>
          <table>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', width: '45%' }}>Result</th>
                <th style={{ textAlign: 'left' }}>Parameter</th>
              </tr>
            </thead>
            <tbody>
              {section.items.map((m, i) => (
                <tr key={i}>
                  <td className="row-highlight">
                    {m.value} <small style={{ fontWeight: 400, opacity: 0.7, fontSize: '0.7rem' }}>{m.unit}</small>
                  </td>
                  <td>{m.label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
};
