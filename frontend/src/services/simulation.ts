import { api } from './api';
import { SimulationRequest, SimulationResponse, WSFrame } from '../types/simulation';

export const runSimulation = async (payload: SimulationRequest): Promise<SimulationResponse> => {
  const response = await api.post<SimulationResponse>('/simulate', payload);
  return response.data;
};

export const createSimulationWebSocket = (
  seed: number,
  onFrame: (frame: WSFrame) => void,
  onError?: (err: Event) => void,
  onClose?: () => void
): WebSocket => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/api/ws/simulate?seed=${seed}`;
  
  const socket = new WebSocket(wsUrl);
  
  socket.onmessage = (event) => {
    try {
      const frame: WSFrame = JSON.parse(event.data);
      onFrame(frame);
    } catch (e) {
      console.error('Failed to parse WebSocket frame', e);
    }
  };

  if (onError) socket.onerror = onError;
  if (onClose) socket.onclose = onClose;

  return socket;
};
