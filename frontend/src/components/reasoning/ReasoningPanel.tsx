import React from 'react';
import { StepData } from '../../types/simulation';
import { Brain, Terminal } from 'lucide-react';

interface ReasoningPanelProps {
  stepData: StepData | undefined;
}

export const ReasoningPanel: React.FC<ReasoningPanelProps> = ({ stepData }) => {
  const explanation = stepData?.decision_explanation || 'Initializing mission parameters. Establishing communication channels.';
  const cmdMsg = stepData?.cmd_msg;

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-3 flex flex-col gap-2 shrink-0 shadow-lg">
      <div className="flex items-center justify-between font-title font-bold text-xs uppercase tracking-wider text-cyan-400">
        <div className="flex items-center gap-1.5">
          <Brain className="w-3.5 h-3.5" />
          <span>Decision Explanation & Reasoning</span>
        </div>
      </div>

      {/* JSON Payload viewer */}
      <div className="bg-slate-950/80 border border-slate-800/80 rounded-md p-2 font-mono text-[11px] text-cyan-300 overflow-x-auto max-h-20 flex items-start gap-2">
        <Terminal className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-0.5" />
        <pre className="whitespace-pre-wrap">
          {cmdMsg ? JSON.stringify(cmdMsg, null, 2) : '// No message output for current timestep.'}
        </pre>
      </div>

      {/* Natural language reasoning text */}
      <div className="bg-slate-800/30 border-l-2 border-cyan-400 rounded-r-md p-2.5 text-xs text-slate-200 leading-relaxed font-body">
        {explanation}
      </div>
    </div>
  );
};
