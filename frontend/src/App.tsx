import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { usePlaybackStore } from './stores/usePlaybackStore';
import { getHealth } from './services/health';
import { getAvailableSeeds, getReplayTrajectory } from './services/replay';
import { runSimulation } from './services/simulation';
import { SimulationResponse } from './types/simulation';

// Components
import { Header } from './components/layout/Header';
import { MetricsDeck } from './components/dashboard/MetricsDeck';
import { MissionControls } from './components/dashboard/MissionControls';
import { GridMap } from './components/grid/GridMap';
import { ReasoningPanel } from './components/reasoning/ReasoningPanel';
import { TimelinePanel } from './components/timeline/TimelinePanel';
import { ChartsPanel } from './components/charts/ChartsPanel';
import { SummaryModal } from './components/common/SummaryModal';

export const App: React.FC = () => {
  const {
    currentStep,
    maxSteps,
    isPlaying,
    playbackSpeed,
    selectedSeed,
    executionMode,
    setStep,
    setMaxSteps,
    setIsPlaying,
    setSelectedSeed,
    setSummaryModalOpen,
  } = usePlaybackStore();

  // 1. Health check query
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 5000,
  });

  // 2. Seeds query
  const { data: seeds = [123, 42, 999] } = useQuery({
    queryKey: ['seeds'],
    queryFn: getAvailableSeeds,
  });

  // 3. Trajectory query
  const [activeTrajectory, setActiveTrajectory] = React.useState<SimulationResponse | undefined>(undefined);
  const [isLoadingSim, setIsLoadingSim] = React.useState<boolean>(false);

  // Load initial replay trajectory on startup or seed change
  useEffect(() => {
    let isMounted = true;
    setIsLoadingSim(true);
    getReplayTrajectory(selectedSeed)
      .then((data) => {
        if (isMounted) {
          setActiveTrajectory(data);
          setMaxSteps(data.steps.length - 1);
          setStep(0);
        }
      })
      .catch((err) => console.error('Failed to load initial trajectory', err))
      .finally(() => {
        if (isMounted) setIsLoadingSim(false);
      });

    return () => { isMounted = false; };
  }, [selectedSeed, setMaxSteps, setStep]);

  // Handle Launch Mission
  const handleLaunchMission = async () => {
    setIsLoadingSim(true);
    setIsPlaying(false);
    try {
      if (executionMode === 'live') {
        const data = await runSimulation({ seed: selectedSeed, mode: 'live' });
        setActiveTrajectory(data);
        setMaxSteps(data.steps.length - 1);
      } else {
        const data = await getReplayTrajectory(selectedSeed);
        setActiveTrajectory(data);
        setMaxSteps(data.steps.length - 1);
      }
      setStep(0);
      setIsPlaying(true);
    } catch (err) {
      console.error('Mission launch failed', err);
    } finally {
      setIsLoadingSim(false);
    }
  };

  // Autoplay playback timer loop
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      usePlaybackStore.setState((state) => {
        if (state.currentStep < state.maxSteps) {
          return { currentStep: state.currentStep + 1 };
        } else {
          return { isPlaying: false, isSummaryModalOpen: true };
        }
      });
    }, playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        usePlaybackStore.getState().togglePlayPause();
      } else if (e.code === 'ArrowLeft') {
        const step = usePlaybackStore.getState().currentStep;
        usePlaybackStore.getState().setStep(Math.max(0, step - 1));
      } else if (e.code === 'ArrowRight') {
        const step = usePlaybackStore.getState().currentStep;
        const max = usePlaybackStore.getState().maxSteps;
        usePlaybackStore.getState().setStep(Math.min(max, step + 1));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Export Incident Report (.md)
  const handleExportReport = () => {
    if (!activeTrajectory) return;
    const m = activeTrajectory.metrics;
    let md = `# CrisisGrid AI Operations Platform — Incident Report\n\n`;
    md += `## Mission Parameters\n`;
    md += `- **Scenario Seed**: ${selectedSeed}\n`;
    md += `- **Execution Mode**: ${executionMode.toUpperCase()}\n`;
    md += `- **Policy Evaluated**: Qwen2-1.5B + LoRA (GRPO-Trained)\n\n`;
    md += `## Final Outcome Metrics\n`;
    md += `- **Initial Population**: ${m.initial_population}\n`;
    md += `- **Population Saved**: ${m.population_saved}\n`;
    md += `- **Final Survival Rate**: ${(m.final_survival * 100).toFixed(1)}%\n`;
    md += `- **Agent Reliability**: ${(m.agent_reliability * 100).toFixed(1)}%\n`;
    md += `- **Active Emergencies at T-50**: ${m.active_emergencies}\n\n`;
    md += `## Timeline Events\n`;
    activeTrajectory.events.forEach((e) => {
      md += `- **T-${String(e.step).padStart(2, '0')}:00** [${e.type.toUpperCase()}] ${e.text}\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CrisisGrid_Incident_Report_Seed_${selectedSeed}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const stepTrained = activeTrajectory?.steps[currentStep];

  return (
    <div className="h-screen max-h-screen w-screen p-3 flex flex-col gap-3 bg-[#090d16] text-slate-100 overflow-hidden box-border">
      {/* 1. Top Navigation */}
      <Header health={health} driftStatus={stepTrained?.drift_status} />

      {/* 2. Executive KPI Deck */}
      <MetricsDeck metrics={activeTrajectory?.metrics} survivalRate={stepTrained?.survival_rate} />

      {/* 3. Core Main Dashboard Viewport */}
      <main className="flex-1 flex gap-3 min-h-0">
        {/* Left Ops Deck */}
        <aside className="w-[300px] flex flex-col gap-3 shrink-0 min-h-0">
          <MissionControls
            seeds={seeds}
            onLaunch={handleLaunchMission}
            onExportReport={handleExportReport}
            isLoading={isLoadingSim}
          />
        </aside>

        {/* Center Grid Focal Map */}
        <section className="flex-1 flex flex-col gap-3 min-h-0">
          <GridMap stepTrained={stepTrained} stepRandom={stepTrained} />
        </section>

        {/* Right Information & Timeline Panel */}
        <aside className="w-[340px] flex flex-col gap-3 shrink-0 min-h-0">
          <ReasoningPanel stepData={stepTrained} />
          <TimelinePanel steps={activeTrajectory?.steps} events={activeTrajectory?.events} />
          <ChartsPanel
            survivalCurve={activeTrajectory?.survival_curve}
            severityCurve={activeTrajectory?.severity_curve}
          />
        </aside>
      </main>

      {/* Post-Mission Debrief Modal */}
      <SummaryModal data={activeTrajectory} onExport={handleExportReport} />
    </div>
  );
};
