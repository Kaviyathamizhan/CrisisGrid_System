import React from 'react';
import { usePlaybackStore, ExecutionMode } from '../../stores/usePlaybackStore';
import { Play, Pause, SkipBack, SkipForward, Rocket, FileText } from 'lucide-react';

interface MissionControlsProps {
  seeds: number[];
  onLaunch: () => void;
  onExportReport: () => void;
  isLoading: boolean;
}

export const MissionControls: React.FC<MissionControlsProps> = ({
  seeds,
  onLaunch,
  onExportReport,
  isLoading,
}) => {
  const {
    currentStep,
    maxSteps,
    isPlaying,
    playbackSpeed,
    selectedSeed,
    executionMode,
    setStep,
    togglePlayPause,
    setSpeed,
    setSelectedSeed,
    setExecutionMode,
  } = usePlaybackStore();

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-3 flex flex-col gap-3 flex-1 min-h-0 text-xs shadow-lg">
      <div className="font-title font-bold uppercase tracking-wider text-slate-400 text-[11px]">
        Operations Deck
      </div>

      {/* Execution Mode */}
      <div className="flex flex-col gap-1">
        <label className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          Execution Mode
        </label>
        <select
          value={executionMode}
          onChange={(e) => setExecutionMode(e.target.value as ExecutionMode)}
          className="bg-slate-950/80 border border-slate-700/60 rounded-md px-2.5 py-1.5 text-slate-200 focus:border-cyan-400 focus:outline-none cursor-pointer"
        >
          <option value="replay">Fast Replay Mode (Pre-cached Trajectories)</option>
          <option value="live">Live AI Inference Mode (Qwen2-1.5B + LoRA)</option>
        </select>
      </div>

      {/* Seed Select */}
      <div className="flex flex-col gap-1">
        <label className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          Disaster Scenario Seed
        </label>
        <select
          value={selectedSeed}
          onChange={(e) => setSelectedSeed(Number(e.target.value))}
          className="bg-slate-950/80 border border-slate-700/60 rounded-md px-2.5 py-1.5 text-slate-200 focus:border-cyan-400 focus:outline-none cursor-pointer"
        >
          {seeds.map((s) => (
            <option key={s} value={s}>
              Scenario Seed {s} {s === 123 ? '(Standard Benchmark)' : s === 42 ? '(Water Shortage)' : '(Flood Escalation)'}
            </option>
          ))}
        </select>
      </div>

      {/* Launch Mission Button */}
      <button
        onClick={onLaunch}
        disabled={isLoading}
        className="w-full py-2.5 px-3 rounded-md bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-title font-bold text-xs flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(0,242,254,0.2)] hover:shadow-[0_0_20px_rgba(0,242,254,0.35)] transition-all active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed mt-1"
      >
        {isLoading ? (
          <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
        ) : (
          <Rocket className="w-4 h-4" />
        )}
        <span>{isLoading ? 'LAUNCHING SIMULATION...' : 'LAUNCH MISSION SIMULATION'}</span>
      </button>

      {/* Playback Controls */}
      <div className="flex items-center gap-2 mt-1">
        <button
          onClick={togglePlayPause}
          className="flex-1 py-1.5 px-3 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 rounded-md font-semibold text-slate-200 flex items-center justify-center gap-1.5 transition-colors"
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5 text-amber-400" /> : <Play className="w-3.5 h-3.5 text-cyan-400" />}
          <span>{isPlaying ? 'Pause' : 'Play'}</span>
        </button>

        <button
          onClick={() => setStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="p-1.5 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 rounded-md text-slate-300 disabled:opacity-40"
        >
          <SkipBack className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={() => setStep(Math.min(maxSteps, currentStep + 1))}
          disabled={currentStep >= maxSteps}
          className="p-1.5 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 rounded-md text-slate-300 disabled:opacity-40"
        >
          <SkipForward className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Timeline Range Scrubber */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          <span>Playback Scrub Timeline</span>
          <span className="font-mono text-cyan-400">T-{String(currentStep).padStart(2, '0')}</span>
        </div>
        <input
          type="range"
          min={0}
          max={maxSteps}
          value={currentStep}
          onChange={(e) => setStep(Number(e.target.value))}
          className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Speed Controller */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          <span>Interval Speed</span>
          <span className="font-mono text-slate-300">{(playbackSpeed / 1000).toFixed(1)}s</span>
        </div>
        <input
          type="range"
          min={100}
          max={1500}
          step={100}
          value={playbackSpeed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Incident Report Download */}
      <button
        onClick={onExportReport}
        className="w-full mt-auto py-2 px-3 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/60 rounded-md font-semibold text-slate-300 flex items-center justify-center gap-2 text-xs transition-colors"
      >
        <FileText className="w-3.5 h-3.5 text-cyan-400" />
        <span>Generate Incident Report</span>
      </button>
    </div>
  );
};
