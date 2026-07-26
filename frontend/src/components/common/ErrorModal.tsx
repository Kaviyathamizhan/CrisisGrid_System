import React from 'react';
import { AlertOctagon, RefreshCw, X } from 'lucide-react';

interface ErrorModalProps {
  isOpen: boolean;
  errorMessage: string | null;
  errorReason?: string;
  onRetry: () => void;
  onClose: () => void;
}

export const ErrorModal: React.FC<ErrorModalProps> = ({
  isOpen,
  errorMessage,
  errorReason = 'Network or backend execution error occurred',
  onRetry,
  onClose,
}) => {
  if (!isOpen || !errorMessage) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-rose-500/60 rounded-xl max-w-md w-full p-5 shadow-[0_0_30px_rgba(244,63,94,0.25)] flex flex-col gap-4 text-slate-100 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-200"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2.5 text-rose-400 font-title font-bold text-base border-b border-slate-800 pb-3">
          <AlertOctagon className="w-6 h-6 text-rose-500 animate-pulse" />
          <span>Mission Execution Failed</span>
        </div>

        <div className="bg-rose-950/30 border border-rose-500/30 rounded-lg p-3 flex flex-col gap-1 text-xs">
          <span className="text-rose-400 font-semibold uppercase tracking-wider text-[10px]">
            Failure Diagnostic Reason
          </span>
          <p className="text-slate-200 font-medium font-sans">{errorReason}</p>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-32">
          <div className="text-slate-500 text-[10px] uppercase font-bold mb-1">
            System Error Trace:
          </div>
          <code>{errorMessage}</code>
        </div>

        <div className="flex justify-end items-center gap-3 mt-1">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-xs font-semibold"
          >
            Dismiss
          </button>

          <button
            onClick={onRetry}
            className="px-4 py-2 bg-rose-500 hover:bg-rose-400 text-slate-950 font-title font-bold rounded-md text-xs flex items-center gap-2 shadow-lg transition-all active:scale-[0.99]"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Retry Mission</span>
          </button>
        </div>
      </div>
    </div>
  );
};
