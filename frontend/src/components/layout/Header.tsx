import React from 'react';
import { usePlaybackStore } from '../../stores/usePlaybackStore';
import { HealthResponse } from '../../types/simulation';
import { Shield, Cpu, Activity, Zap } from 'lucide-react';

interface HeaderProps {
  health: HealthResponse | undefined;
  driftStatus?: {
    status: 'NORMAL' | 'WARNING' | 'RECOVERING' | 'STABLE';
  };
}

export const Header: React.FC<HeaderProps> = ({ health, driftStatus }) => {
  const { executionMode } = usePlaybackStore();

  const getStatusBadge = () => {
    const status = driftStatus?.status || 'NORMAL';
    switch (status) {
      case 'WARNING':
        return 'border-amber-500/40 bg-amber-500/10 text-amber-400 shadow-[0_0_12px_rgba(245,158,11,0.2)]';
      case 'RECOVERING':
        return 'border-amber-400/40 bg-amber-400/10 text-amber-300';
      case 'STABLE':
        return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400';
      default:
        return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.15)]';
    }
  };

  return (
    <header className="h-12 px-4 bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg flex items-center justify-between shrink-0 shadow-lg">
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#00f2fe] animate-pulse" />
        </div>
        <div className="flex items-baseline gap-2">
          <h1 className="font-title text-base font-bold tracking-wide text-white">
            CrisisGrid
          </h1>
          <span className="font-title text-xs tracking-widest text-cyan-400 uppercase font-semibold">
            AI Operations Platform
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs font-medium">
        <div className="px-2.5 py-1 bg-slate-800/60 border border-slate-700/50 rounded-md flex items-center gap-1.5 text-slate-300">
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
          <span>Agent: <strong className="text-white">ACTIVE</strong></span>
        </div>

        <div className="px-2.5 py-1 bg-slate-800/60 border border-slate-700/50 rounded-md flex items-center gap-1.5 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span>Model: <strong className="text-white">Qwen2-1.5B + LoRA</strong></span>
        </div>

        <div className="px-2.5 py-1 bg-slate-800/60 border border-slate-700/50 rounded-md flex items-center gap-1.5 text-slate-300">
          <Shield className="w-3.5 h-3.5 text-cyan-400" />
          <span>Mode: <strong className="text-white capitalize">{executionMode}</strong></span>
        </div>

        <div className={`px-3 py-1 border rounded-md font-bold uppercase tracking-wider flex items-center gap-1.5 text-[11px] ${getStatusBadge()}`}>
          <Activity className="w-3 h-3" />
          <span>System Status: {driftStatus?.status || 'NORMAL'}</span>
        </div>
      </div>
    </header>
  );
};
