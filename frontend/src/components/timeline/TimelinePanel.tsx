import React, { useState } from 'react';
import { StepData, EventItem } from '../../types/simulation';
import { usePlaybackStore } from '../../stores/usePlaybackStore';
import { ShieldAlert, Clock, AlertTriangle, CheckCircle, Info } from 'lucide-react';

interface TimelinePanelProps {
  steps: StepData[] | undefined;
  events: EventItem[] | undefined;
}

export const TimelinePanel: React.FC<TimelinePanelProps> = ({ steps, events }) => {
  const [activeTab, setActiveTab] = useState<'oversight' | 'timeline'>('oversight');
  const { currentStep } = usePlaybackStore();

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'critical': return <AlertTriangle className="w-3 h-3 text-rose-400" />;
      case 'success': return <CheckCircle className="w-3 h-3 text-emerald-400" />;
      case 'warning': return <AlertTriangle className="w-3 h-3 text-amber-400" />;
      default: return <Info className="w-3 h-3 text-cyan-400" />;
    }
  };

  const getTagColor = (type: string) => {
    switch (type) {
      case 'critical': return 'text-rose-400 font-bold';
      case 'success': return 'text-emerald-400 font-bold';
      case 'warning': return 'text-amber-400 font-bold';
      default: return 'text-cyan-400 font-bold';
    }
  };

  // Collect oversight terminal logs up to current step
  const activeOversightLogs: { step: number; text: string; type: string }[] = [];
  if (steps) {
    for (let i = 0; i <= Math.min(currentStep, steps.length - 1); i++) {
      const s = steps[i];
      if (i > 0 && s.cmd_msg) {
        activeOversightLogs.push({
          step: i,
          text: `Command Agent generated action: ${JSON.stringify(s.cmd_msg)}`,
          type: 'info'
        });
      }
      (s.oversight_logs || []).forEach((flag: any) => {
        let ftype = 'info';
        if (flag.type === 'schema_drift' || flag.type === 'population_loss' || flag.type === 'malformed_message') {
          ftype = 'critical';
        } else if (flag.type === 'schema_recovery') {
          ftype = 'success';
        } else if (flag.type === 'default_action') {
          ftype = 'warning';
        }
        activeOversightLogs.push({ step: i, text: `Event log: ${flag.type}`, type: ftype });
      });
    }
  }

  const activeEvents = (events || []).filter(e => e.step <= currentStep).slice(-6);

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-3 flex flex-col flex-1 min-h-0 shadow-lg">
      <div className="flex border-b border-slate-800 pb-2 mb-2 gap-4 text-xs font-title font-bold">
        <button
          onClick={() => setActiveTab('oversight')}
          className={`flex items-center gap-1.5 pb-1 border-b-2 transition-colors ${
            activeTab === 'oversight'
              ? 'border-cyan-400 text-cyan-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Oversight Monitor</span>
        </button>

        <button
          onClick={() => setActiveTab('timeline')}
          className={`flex items-center gap-1.5 pb-1 border-b-2 transition-colors ${
            activeTab === 'timeline'
              ? 'border-cyan-400 text-cyan-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Clock className="w-3.5 h-3.5" />
          <span>Operational Timeline</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto font-mono text-[11px] flex flex-col gap-1.5 pr-1">
        {activeTab === 'oversight' ? (
          activeOversightLogs.length > 0 ? (
            activeOversightLogs.map((log, idx) => (
              <div key={idx} className="flex gap-2 items-start py-0.5 border-b border-slate-800/40">
                <span className="text-slate-500 shrink-0">[T-{String(log.step).padStart(2, '0')}]</span>
                <span className={`shrink-0 uppercase ${getTagColor(log.type)}`}>[{log.type}]</span>
                <span className="text-slate-300 break-all">{log.text}</span>
              </div>
            ))
          ) : (
            <div className="text-slate-500 italic py-2">Command Center standby. Ready for mission launch...</div>
          )
        ) : (
          activeEvents.map((evt, idx) => (
            <div key={idx} className="flex gap-2.5 items-start py-1 border-b border-slate-800/40">
              <div className="mt-0.5">{getEventIcon(evt.type)}</div>
              <div className="flex flex-col">
                <span className="text-[10px] text-slate-500">T-{String(evt.step).padStart(2, '0')}</span>
                <span className="text-slate-200 font-sans font-medium text-[11px]">{evt.text}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
