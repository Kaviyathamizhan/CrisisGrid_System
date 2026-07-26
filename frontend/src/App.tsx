import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { usePlaybackStore } from './stores/usePlaybackStore';
import { getHealth } from './services/health';
import { getAvailableSeeds, getComparisonTrajectories } from './services/replay';
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

  // 3. TWO distinct trajectory states — trained policy vs random baseline
  const [trainedTrajectory, setTrainedTrajectory] = React.useState<SimulationResponse | undefined>(undefined);
  const [randomTrajectory, setRandomTrajectory] = React.useState<SimulationResponse | undefined>(undefined);
  const [isLoadingSim, setIsLoadingSim] = React.useState<boolean>(false);

  // Load both trajectories on startup or seed change via /api/comparison
  useEffect(() => {
    let isMounted = true;
    setIsLoadingSim(true);
    getComparisonTrajectories(selectedSeed)
      .then((data) => {
        if (isMounted) {
          setTrainedTrajectory(data.trained);
          setRandomTrajectory(data.random);
          setMaxSteps(data.trained.steps.length - 1);
          setStep(0);
        }
      })
      .catch((err) => console.error('Failed to load comparison trajectories', err))
      .finally(() => {
        if (isMounted) setIsLoadingSim(false);
      });

    return () => { isMounted = false; };
  }, [selectedSeed, setMaxSteps, setStep]);

  // Handle Launch Mission
  const handleLaunchMission = async () => {
    setIsLoadingSim(true);
    setIsPlaying(false);
    setStep(0);

    if (executionMode === 'live') {
      // Use WebSocket streaming for live mode — each step arrives as it's computed
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/api/ws/simulate?seed=${selectedSeed}&mode=live`;

      const liveSteps: any[] = [];
      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        try {
          const frame = JSON.parse(event.data);

          if (frame.type === 'init') {
            setMaxSteps(frame.total_steps);
          } else if (frame.type === 'status') {
            console.log('[CrisisGrid] Backend:', frame.message);
          } else if (frame.type === 'step') {
            liveSteps.push(frame.data);
            setTrainedTrajectory((prev) => ({
              agent_type: 'trained',
              seed: selectedSeed,
              mode: 'live',
              steps: [...liveSteps],
              survival_curve: liveSteps.map((s) => s.survival_rate),
              severity_curve: liveSteps.map((s) => s.mean_severity),
              events: prev?.events || [],
              metrics: prev?.metrics || {
                final_survival: 0, population_saved: 0, initial_population: 1800,
                total_reward: 0, agent_reliability: 0, active_emergencies: 0, resource_efficiency: 0,
              },
            }));
            setStep(frame.step);
            setMaxSteps(Math.max(frame.step, usePlaybackStore.getState().maxSteps));
          } else if (frame.type === 'complete') {
            setTrainedTrajectory((prev) => prev ? ({
              ...prev,
              metrics: frame.metrics,
              events: frame.events || prev.events,
            }) : prev);
            setIsLoadingSim(false);
            setIsPlaying(false);
            setSummaryModalOpen(true);
          } else if (frame.type === 'error') {
            console.error('[CrisisGrid] WebSocket error:', frame.message);
            setIsLoadingSim(false);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket frame', e);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket connection error', err);
        setIsLoadingSim(false);
      };

      ws.onclose = () => {
        setIsLoadingSim(false);
      };
    } else {
      // Replay mode — fetch both trained and random via comparison endpoint
      try {
        const data = await getComparisonTrajectories(selectedSeed);
        setTrainedTrajectory(data.trained);
        setRandomTrajectory(data.random);
        setMaxSteps(data.trained.steps.length - 1);
        setStep(0);
        setIsPlaying(true);
      } catch (err) {
        console.error('Mission launch failed', err);
      } finally {
        setIsLoadingSim(false);
      }
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
    if (!trainedTrajectory) return;
    const m = trainedTrajectory.metrics;
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
    trainedTrajectory.events.forEach((e) => {
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

  // Current step data for each agent — these are DISTINCT objects
  const stepTrained = trainedTrajectory?.steps[currentStep];
  const stepRandom = randomTrajectory?.steps[currentStep];

  return (
    <div className="h-screen max-h-screen w-screen p-3 flex flex-col gap-3 bg-[#090d16] text-slate-100 overflow-hidden box-border">
      {/* 1. Top Navigation */}
      <Header health={health} driftStatus={stepTrained?.drift_status} />

      {/* 2. Executive KPI Deck */}
      <MetricsDeck metrics={trainedTrajectory?.metrics} survivalRate={stepTrained?.survival_rate} />

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

        {/* Center Grid Focal Map — TWO DISTINCT TRAJECTORIES */}
        <section className="flex-1 flex flex-col gap-3 min-h-0">
          <GridMap stepTrained={stepTrained} stepRandom={stepRandom} />
        </section>

        {/* Right Information & Timeline Panel */}
        <aside className="w-[340px] flex flex-col gap-3 shrink-0 min-h-0">
          <ReasoningPanel stepData={stepTrained} />
          <TimelinePanel steps={trainedTrajectory?.steps} events={trainedTrajectory?.events} />
          <ChartsPanel
            survivalCurve={trainedTrajectory?.survival_curve}
            severityCurve={trainedTrajectory?.severity_curve}
          />
        </aside>
      </main>

      {/* Post-Mission Debrief Modal */}
      <SummaryModal data={trainedTrajectory} onExport={handleExportReport} />
    </div>
  );
};
