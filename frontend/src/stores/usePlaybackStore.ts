import { create } from 'zustand';

export type HeatmapMode = 'severity' | 'population' | 'resource';
export type ViewMode = 'comparison' | 'single';
export type ExecutionMode = 'replay' | 'live';

interface PlaybackState {
  currentStep: number;
  maxSteps: number;
  isPlaying: boolean;
  playbackSpeed: number;
  activeHeatmap: HeatmapMode;
  viewMode: ViewMode;
  selectedSeed: number;
  executionMode: ExecutionMode;
  isSummaryModalOpen: boolean;

  setStep: (step: number) => void;
  setMaxSteps: (max: number) => void;
  setIsPlaying: (playing: boolean) => void;
  togglePlayPause: () => void;
  setSpeed: (speed: number) => void;
  setHeatmap: (mode: HeatmapMode) => void;
  setViewMode: (mode: ViewMode) => void;
  setSelectedSeed: (seed: number) => void;
  setExecutionMode: (mode: ExecutionMode) => void;
  setSummaryModalOpen: (open: boolean) => void;
  resetPlayback: () => void;
}

export const usePlaybackStore = create<PlaybackState>((set) => ({
  currentStep: 0,
  maxSteps: 50,
  isPlaying: false,
  playbackSpeed: 500,
  activeHeatmap: 'severity',
  viewMode: 'comparison',
  selectedSeed: 123,
  executionMode: 'replay',
  isSummaryModalOpen: false,

  setStep: (step) => set({ currentStep: step }),
  setMaxSteps: (maxSteps) => set({ maxSteps }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  togglePlayPause: () => set((state) => ({ isPlaying: !state.isPlaying })),
  setSpeed: (playbackSpeed) => set({ playbackSpeed }),
  setHeatmap: (activeHeatmap) => set({ activeHeatmap }),
  setViewMode: (viewMode) => set({ viewMode }),
  setSelectedSeed: (selectedSeed) => set({ selectedSeed }),
  setExecutionMode: (executionMode) => set({ executionMode }),
  setSummaryModalOpen: (isSummaryModalOpen) => set({ isSummaryModalOpen }),
  resetPlayback: () => set({ currentStep: 0, isPlaying: false }),
}));
