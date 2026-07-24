import { api } from './api';
import { HealthResponse } from '../types/simulation';

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};
