import React from 'react';
import { usePlaybackStore, HeatmapMode } from '../../stores/usePlaybackStore';
import { StepData } from '../../types/simulation';

interface GridMapProps {
  stepTrained: StepData | undefined;
  stepRandom: StepData | undefined;
}

export const GridMap: React.FC<GridMapProps> = ({ stepTrained, stepRandom }) => {
  const { activeHeatmap, viewMode, setHeatmap, setViewMode } = usePlaybackStore();

  const getSeverityColor = (sev: number) => {
    if (sev < 0.2) return `rgba(16, 185, 129, ${sev * 0.4})`;
    if (sev < 0.5) return `rgba(245, 158, 11, ${(sev - 0.2) * 0.8 + 0.15})`;
    if (sev < 0.8) return `rgba(249, 115, 22, ${(sev - 0.5) * 1.2 + 0.3})`;
    return `rgba(244, 63, 94, ${(sev - 0.8) * 1.5 + 0.65})`;
  };

  const getPopulationColor = (pop: number) => {
    const ratio = pop / 100;
    return `rgba(99, 102, 241, ${ratio * 0.45})`;
  };

  const getResourceColor = (units: number) => {
    if (units === 0) return 'rgba(15, 23, 42, 0.4)';
    return `rgba(0, 242, 254, ${Math.min(1.0, 0.2 + units * 0.08)})`;
  };

  const getResourceIcon = (res?: string) => {
    switch (res) {
      case 'medicine': return '🏥';
      case 'food': return '🍏';
      case 'rescue': return '🚁';
      case 'water': return '💧';
      case 'shelter': return '🏠';
      default: return '📦';
    }
  };

  const renderGridContent = (stepData: StepData | undefined, prefix: string) => {
    if (!stepData) return null;
    const grid = stepData.grid;
    const cells = [];

    // Col Labels
    cells.push(<div key="hdr-corner" className="text-[10px] font-mono text-slate-500 font-bold text-center"></div>);
    for (let j = 0; j < 5; j++) {
      cells.push(<div key={`hdr-${j}`} className="text-[10px] font-mono text-slate-400 font-bold text-center">C{j}</div>);
    }

    // Grid Rows
    for (let i = 0; i < 5; i++) {
      cells.push(<div key={`row-lbl-${i}`} className="text-[10px] font-mono text-slate-400 font-bold flex items-center justify-center">R{i}</div>);

      for (let j = 0; j < 5; j++) {
        const cellIdx = i * 5 + j;
        const pop = Math.round(grid[i][j][0]);
        const sev = grid[i][j][1];
        const resUnits = Math.round(grid[i][j][2]);
        const isCmdZone = i < 2;

        let bg = 'rgba(15, 23, 42, 0.6)';
        let statDisplay = '-';

        if (activeHeatmap === 'severity') {
          statDisplay = `${(sev * 100).toFixed(0)}%`;
          bg = getSeverityColor(sev);
        } else if (activeHeatmap === 'population') {
          statDisplay = `${pop}`;
          bg = getPopulationColor(pop);
        } else if (activeHeatmap === 'resource') {
          statDisplay = resUnits > 0 ? `+${resUnits}` : '-';
          bg = getResourceColor(resUnits);
        }

        const isCritical = sev > 0.9;
        const lastAction: any = stepData.res_action || {};
        const isAllocatedHere = (lastAction.action === 'allocate' || lastAction.action === 'default') && lastAction.zone === cellIdx;

        cells.push(
          <div
            key={`cell-${cellIdx}`}
            style={{ backgroundColor: bg }}
            className={`group relative rounded-md border flex flex-col justify-between p-1.5 transition-all duration-200 aspect-square cursor-pointer hover:scale-105 hover:z-20 hover:shadow-lg ${
              isCritical
                ? 'border-rose-500 animate-pulse shadow-[0_0_12px_rgba(244,63,94,0.4)]'
                : isCmdZone
                ? 'border-cyan-500/25 hover:border-cyan-400'
                : 'border-amber-500/25 hover:border-amber-400'
            }`}
          >
            <div className="flex justify-between items-center text-[9px] font-mono font-bold text-slate-400">
              <span>{prefix}{i}{j}</span>
              <span className="text-[8px] text-slate-500">{isCmdZone ? 'CMD' : 'RES'}</span>
            </div>

            <div className="self-center font-title font-bold text-sm text-slate-100 my-auto">
              {statDisplay}
            </div>

            <div className="flex items-center justify-between text-[9px] font-semibold text-slate-300">
              <span>👥 {pop}</span>
              {isAllocatedHere ? (
                <span className="px-1 rounded bg-cyan-400/20 border border-cyan-400 text-[8px] font-bold">
                  {getResourceIcon(lastAction.resource)}
                </span>
              ) : resUnits > 0 ? (
                <span className="px-1 rounded bg-slate-700/50 text-[8px]">📦</span>
              ) : null}
            </div>

            {/* Hover Tooltip */}
            <div className="absolute bottom-[110%] left-1/2 -translate-x-1/2 hidden group-hover:flex flex-col gap-1 bg-slate-900/95 border border-slate-700/80 rounded-md p-2 text-[10px] w-36 shadow-2xl backdrop-blur-md z-30 pointer-events-none">
              <div className="flex justify-between text-slate-400 border-b border-slate-800 pb-1">
                <span>Zone {cellIdx}:</span>
                <span className="font-semibold text-white">{isCmdZone ? 'Command (0)' : 'Resource (1)'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Severity:</span>
                <span className="font-bold text-cyan-400">{(sev * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Survivors:</span>
                <span className="font-bold text-slate-200">{pop}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Resource:</span>
                <span className="font-bold text-emerald-400">{resUnits > 0 ? `${resUnits} units` : 'None'}</span>
              </div>
            </div>
          </div>
        );
      }
    }

    return cells;
  };

  return (
    <div className="flex-1 flex flex-col gap-2 min-h-0">
      {/* Top View Controls */}
      <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-2 flex justify-between items-center shrink-0">
        <div className="flex bg-slate-950/80 border border-slate-800 p-0.5 rounded-md text-xs font-semibold text-slate-400">
          {(['severity', 'population', 'resource'] as HeatmapMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setHeatmap(mode)}
              className={`px-3 py-1 rounded-sm capitalize transition-colors ${
                activeHeatmap === mode
                  ? 'bg-slate-800 text-cyan-400 font-bold shadow-sm'
                  : 'hover:text-slate-200'
              }`}
            >
              {mode} Map
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('comparison')}
            className={`px-3 py-1 rounded-md text-xs font-semibold border transition-colors ${
              viewMode === 'comparison'
                ? 'border-cyan-400/60 bg-cyan-400/10 text-cyan-400'
                : 'border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            Split Screen Comparison
          </button>

          <button
            onClick={() => setViewMode('single')}
            className={`px-3 py-1 rounded-md text-xs font-semibold border transition-colors ${
              viewMode === 'single'
                ? 'border-cyan-400/60 bg-cyan-400/10 text-cyan-400'
                : 'border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            Single Grid Focus
          </button>
        </div>
      </div>

      {/* Grid Maps */}
      <div className="flex-1 grid grid-cols-2 gap-3 min-h-0">
        {/* Left View: Random Baseline */}
        {viewMode === 'comparison' && (
          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-3 flex flex-col items-center justify-between min-h-0">
            <div className="font-title font-bold text-xs uppercase tracking-wider text-amber-400 flex items-center gap-1.5 self-start">
              <div className="w-2 h-2 rounded-full bg-amber-400" />
              Random Baseline Agent
            </div>
            <div className="grid grid-cols-6 grid-rows-6 gap-1.5 w-full max-w-[340px] aspect-square my-auto">
              {renderGridContent(stepRandom, 'R')}
            </div>
          </div>
        )}

        {/* Right View: Trained Agent */}
        <div className={`bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-lg p-3 flex flex-col items-center justify-between min-h-0 ${viewMode === 'single' ? 'col-span-2' : ''}`}>
          <div className="font-title font-bold text-xs uppercase tracking-wider text-cyan-400 flex items-center gap-1.5 self-start">
            <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#00f2fe]" />
            Trained RL Policy Trajectory
          </div>
          <div className="grid grid-cols-6 grid-rows-6 gap-1.5 w-full max-w-[340px] aspect-square my-auto">
            {renderGridContent(stepTrained, 'T')}
          </div>
        </div>
      </div>
    </div>
  );
};
