export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
}

export interface SeedsResponse {
  seeds: number[];
}

export interface SimulationRequest {
  seed: number;
  mode: 'live' | 'replay';
}

export interface MetricsSummary {
  final_survival: number;
  population_saved: number;
  initial_population: number;
  total_reward: number;
  agent_reliability: number;
  active_emergencies: number;
  resource_efficiency: number;
}

export interface EventItem {
  step: number;
  text: string;
  type: 'info' | 'warning' | 'critical' | 'success';
}

export interface DriftStatus {
  status: 'NORMAL' | 'WARNING' | 'RECOVERING' | 'STABLE';
  api_status: 'active' | 'deprecated';
  current_schema_version: number;
  last_error: string | null;
}

export interface StepData {
  step: number;
  grid: number[][][]; // 5x5x4 [population, severity, resources, zone_id]
  cmd_msg: {
    intent: string;
    zone: number;
    resource: string;
    priority: string;
    units: number;
  } | null;
  res_action: {
    action: string;
    zone: number;
    units: number;
    resource: string;
  } | null;
  reward: number;
  total_reward: number;
  survival_rate: number;
  mean_severity: number;
  max_severity: number;
  drift_status: DriftStatus;
  oversight_logs: any[];
  decision_explanation: string;
}

export interface SimulationResponse {
  agent_type: string;
  seed: number;
  mode: 'live' | 'replay';
  steps: StepData[];
  survival_curve: number[];
  severity_curve: number[];
  events: EventItem[];
  metrics: MetricsSummary;
}

export interface ComparisonMeta {
  survival_delta: number;
  population_saved_delta: number;
  policies_match: boolean;
}

export interface ComparisonResponse {
  seed: number;
  mode: string;
  trained: SimulationResponse;
  random: SimulationResponse;
  comparison: ComparisonMeta;
}

export interface WSFrame {
  type: 'init' | 'step' | 'complete' | 'error';
  step?: number;
  data?: any;
  seed?: number;
  total_steps?: number;
  metrics?: MetricsSummary;
  events?: EventItem[];
  message?: string;
}
