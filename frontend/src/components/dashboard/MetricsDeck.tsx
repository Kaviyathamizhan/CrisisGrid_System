import React from 'react';
import { MetricsSummary } from '../../types/simulation';
import { TrendingUp, TrendingDown, AlertTriangle, Users, ShieldCheck, HeartPulse, Zap } from 'lucide-react';

interface MetricsDeckProps {
  metrics: MetricsSummary | undefined;
  survivalRate?: number;
}

export const MetricsDeck: React.FC<MetricsDeckProps> = ({ metrics, survivalRate }) => {
  const currentSurvival = survivalRate !== undefined ? survivalRate : metrics?.final_survival || 1.0;
  const initialPop = metrics?.initial_population || 1800;
  const savedPop = Math.round(currentSurvival * initialPop);

  return (
    <section className="grid grid-cols-5 gap-3 shrink-0">
      {/* Survival Rate */}
      <div className={`bg-slate-900/80 backdrop-blur-md border rounded-lg p-2.5 flex flex-col justify-between h-18 relative overflow-hidden shadow-lg ${
        currentSurvival < 0.95 ? 'border-rose-500/60 shadow-[0_0_15px_rgba(244,63,94,0.15)]' : 'border-slate-800/80'
      }`}>
        <div className="flex justify-between items-center text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          <span>Survival Rate</span>
          <HeartPulse className="w-3.5 h-3.5 text-cyan-400" />
        </div>
        <div className="flex items-baseline justify-between mt-1">
          <span className="font-title text-xl font-bold text-slate-100">
            {(currentSurvival * 100).toFixed(1)}%
          </span>
          <span className={`text-[10px] font-semibold flex items-center gap-0.5 ${currentSurvival < 0.95 ? 'text-rose-400' : 'text-emerald-400'}`}>
            {currentSurvival < 0.95 ? <TrendingDown className="w-3 h-3" /> : <TrendingUp className="w-3 h-3" />}
            {currentSurvival < 0.95 ? 'Critical' : 'Stable'}
          </span>
        </div>
      </div>

      {/* Population Saved */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-2.5 flex flex-col justify-between h-18 shadow-lg">
        <div className="flex justify-between items-center text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          <span>Population Saved</span>
          <Users className="w-3.5 h-3.5 text-emerald-400" />
        </div>
        <div className="flex items-baseline justify-between mt-1">
          <span className="font-title text-xl font-bold text-slate-100">
            {savedPop.toLocaleString()}
          </span>
          <span className="text-[10px] font-semibold text-emerald-400 flex items-center gap-0.5">
            <TrendingUp className="w-3 h-3" />
            <span>/ {initialPop}</span>
          </span>
        </div>
      </div>

      {/* Resource Efficiency */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-2.5 flex flex-col justify-between h-18 shadow-lg">
        <div className="flex justify-between items-center text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          <span>Resource Efficiency</span>
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
        </div>
        <div className="flex items-baseline justify-between mt-1">
          <span className="font-title text-xl font-bold text-slate-100">
            {((metrics?.resource_efficiency || 1.0) * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] font-semibold text-cyan-400 flex items-center gap-0.5">
            <TrendingUp className="w-3 h-3" />
            Optimal
          </span>
        </div>
      </div>

      {/* Agent Reliability */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-2.5 flex flex-col justify-between h-18 shadow-lg">
        <div className="flex justify-between items-center text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          <span>Agent Reliability</span>
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
        </div>
        <div className="flex items-baseline justify-between mt-1">
          <span className="font-title text-xl font-bold text-slate-100">
            {((metrics?.agent_reliability || 1.0) * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] font-semibold text-emerald-400 flex items-center gap-0.5">
            <TrendingUp className="w-3 h-3" />
            Verified
          </span>
        </div>
      </div>

      {/* Active Emergencies */}
      <div className={`bg-slate-900/80 backdrop-blur-md border rounded-lg p-2.5 flex flex-col justify-between h-18 relative overflow-hidden shadow-lg ${
        (metrics?.active_emergencies || 0) > 5 ? 'border-amber-500/60 shadow-[0_0_15px_rgba(245,158,11,0.15)]' : 'border-slate-800/80'
      }`}>
        <div className="flex justify-between items-center text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          <span>Active Emergencies</span>
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
        </div>
        <div className="flex items-baseline justify-between mt-1">
          <span className="font-title text-xl font-bold text-slate-100">
            {metrics?.active_emergencies || 0}
          </span>
          <span className={`text-[10px] font-semibold ${(metrics?.active_emergencies || 0) > 5 ? 'text-amber-400' : 'text-cyan-400'}`}>
            {(metrics?.active_emergencies || 0) > 5 ? 'Elevated' : 'Controlled'}
          </span>
        </div>
      </div>
    </section>
  );
};
