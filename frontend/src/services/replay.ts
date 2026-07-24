import { api } from './api';
import { SeedsResponse, SimulationResponse } from '../types/simulation';

export const getAvailableSeeds = async (): Promise<number[]> => {
  const response = await api.get<SeedsResponse>('/seeds');
  return response.data.seeds;
};

export const getReplayTrajectory = async (seed: number): Promise<SimulationResponse> => {
  const response = await api.get<SimulationResponse>(`/replay?seed=${seed}`);
  return response.data;
};
