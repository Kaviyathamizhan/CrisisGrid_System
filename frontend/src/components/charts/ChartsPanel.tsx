import React from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { usePlaybackStore } from '../../stores/usePlaybackStore';
import { Activity } from 'lucide-react';

interface ChartsPanelProps {
  survivalCurve: number[] | undefined;
  severityCurve: number[] | undefined;
}

export const ChartsPanel: React.FC<ChartsPanelProps> = ({ survivalCurve, severityCurve }) => {
  const { currentStep } = usePlaybackStore();

  const data = (survivalCurve || []).map((surv, idx) => ({
    step: `T-${idx}`,
    survival: Number((surv * 100).toFixed(1)),
    severity: Number(((severityCurve?.[idx] || 0) * 100).toFixed(1)),
  }));

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-2.5 flex flex-col gap-1.5 h-40 shrink-0 shadow-lg">
      <div className="flex items-center justify-between font-title font-bold text-xs uppercase tracking-wider text-slate-400">
        <div className="flex items-center gap-1.5 text-cyan-400">
          <Activity className="w-3.5 h-3.5" />
          <span>Operational Metrics Curve</span>
        </div>
        <span className="text-[10px] font-mono text-slate-500">
          Active: Step {currentStep} / 50
        </span>
      </div>

      <div className="flex-1 w-full min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorSurv" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00f2fe" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00f2fe" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="colorSev" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#64748b' }} interval={9} />
            <YAxis tick={{ fontSize: 9, fill: '#64748b' }} domain={[0, 100]} unit="%" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '6px',
                fontSize: '11px',
              }}
            />
            <Area
              type="monotone"
              dataKey="survival"
              stroke="#00f2fe"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorSurv)"
              name="Survival Rate (%)"
            />
            <Area
              type="monotone"
              dataKey="severity"
              stroke="#f43f5e"
              strokeWidth={1.5}
              fillOpacity={1}
              fill="url(#colorSev)"
              name="Mean Severity (%)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
