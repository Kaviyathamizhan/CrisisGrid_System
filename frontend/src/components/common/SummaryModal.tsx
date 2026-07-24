import React from 'react';
import { usePlaybackStore } from '../../stores/usePlaybackStore';
import { SimulationResponse } from '../../types/simulation';
import { X, Download, ShieldCheck, HeartPulse, Zap, AlertTriangle } from 'lucide-react';

interface SummaryModalProps {
  data: SimulationResponse | undefined;
  onExport: () => void;
}

export const SummaryModal: React.FC<SummaryModalProps> = ({ data, onExport }) => {
  const { isSummaryModalOpen, setSummaryModalOpen } = usePlaybackStore();

  if (!isSummaryModalOpen || !data) return null;

  const m = data.metrics;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl max-w-lg w-full p-6 shadow-2xl flex flex-col gap-4 text-slate-100 relative">
        <button
          onClick={() => setSummaryModalOpen(false)}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-200"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2 font-title font-bold text-lg text-cyan-400 border-b border-slate-800 pb-3">
          <ShieldCheck className="w-6 h-6 text-cyan-400" />
          <span>Mission Debrief & Incident Report</span>
        </div>

        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 flex items-center gap-1"><HeartPulse className="w-3.5 h-3.5 text-cyan-400" /> Survival Rate</span>
            <span className="text-xl font-bold font-title text-slate-100">{(m.final_survival * 100).toFixed(1)}%</span>
          </div>

          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 flex items-center gap-1"><Zap className="w-3.5 h-3.5 text-emerald-400" /> Survivors Saved</span>
            <span className="text-xl font-bold font-title text-slate-100">{m.population_saved.toLocaleString()}</span>
          </div>

          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 flex items-center gap-1"><ShieldCheck className="w-3.5 h-3.5 text-cyan-400" /> Agent Reliability</span>
            <span className="text-xl font-bold font-title text-slate-100">{(m.agent_reliability * 100).toFixed(1)}%</span>
          </div>

          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 flex items-center gap-1"><AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> Active Emergencies</span>
            <span className="text-xl font-bold font-title text-slate-100">{m.active_emergencies}</span>
          </div>
        </div>

        <div className="text-xs text-slate-300 bg-slate-950/40 p-3 rounded-lg border border-slate-800/80 leading-relaxed font-body">
          <h4 className="font-title font-bold text-slate-200 uppercase text-[11px] mb-1">Executive Summary</h4>
          The mission executed across 50 simulation intervals. At interval 25, an intentional API schema drift was injected (POST /allocate deprecated to PATCH /distribution). The Command Agent adapted, maintaining a {(m.final_survival * 100).toFixed(1)}% final survival rate.
        </div>

        <div className="flex justify-between items-center mt-2">
          <button
            onClick={() => setSummaryModalOpen(false)}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-xs font-semibold"
          >
            Close Window
          </button>

          <button
            onClick={onExport}
            className="px-4 py-2 bg-cyan-400 hover:bg-cyan-300 text-slate-950 font-title font-bold rounded-md text-xs flex items-center gap-2 shadow-lg"
          >
            <Download className="w-4 h-4" />
            <span>Download Incident Report (.md)</span>
          </button>
        </div>
      </div>
    </div>
  );
};
