import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number | undefined;
  unit: string;
  icon: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({ label, value, unit, icon }) => {
  return (
    <div className="card stat-card">
      <div className="stat-label">
        {icon} <span style={{ marginLeft: '4px' }}>{label}</span>
      </div>
      <div className="stat-value">
        {value || '0.0'}
        <span className="stat-unit">{unit}</span>
      </div>
    </div>
  );
};
