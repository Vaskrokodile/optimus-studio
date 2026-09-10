// API client for Optimus Studio backend

export interface HardwareInfo {
  cpu_name: string;
  cpu_cores: number;
  ram_gb: number;
  os: string;
  arch: string;
  gpu: {
    name: string;
    vram_gb: number;
    type: string;
  };
  python_version: string;
  mlx_available: boolean;
  vllm_available: boolean;
  torch_available: boolean;
  recommended_backend: string;
  total_memory_gb: number;
  labels: string[];
}

export interface CompatibilityInfo {
  status: "recommended" | "feasible" | "tight" | "not-recommended";
  best_precision: string;
  best_precision_gb: number;
  reason: string;
  score: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  huggingface_id: string;
  params_b: number;
  fp16_gb: number;
  int8_gb: number;
  int4_gb: number;
  family: string;
  context_length: number;
  backends: string[];
  description: string;
  tags: string[];
  compatibility: CompatibilityInfo;
  local: ModelLocalStatus;
}

export interface ModelLocalStatus {
  model_id: string;
  precision: string;
  repository: string;
  status: "not-downloaded" | "queued" | "downloading" | "downloaded" | "failed";
  path: string;
  bytes_downloaded: number;
  size_gb: number;
  error: string;
}

export interface ContextEstimate {
  context_window: number;
  max_model_context: number;
  max_safe_context: number;
  estimated_tps: number;
  low_tps: number;
  high_tps: number;
  kv_cache_gb: number;
  model_memory_gb: number;
  available_memory_gb: number;
  fits_in_memory: boolean;
  basis: string;
}

export interface PromptSkillInfo {
  id: string;
  name: string;
  description: string;
  source: string;
}

export interface ChatResponse {
  answer: string;
  reasoning: string;
  tokens_per_sec: number;
  context_tokens: number;
  context_window: number;
  context_utilization: number;
  active_skills: PromptSkillInfo[];
  tool_calls: Array<{ name: string; ok: boolean }>;
  uncertainty: { score: number; needs_research: boolean; explicit: boolean; time_sensitive: boolean; reasons: string[] };
  learning_session: LearningSession | null;
}

export interface LearningSession {
  id: string;
  model_id: string;
  query: string;
  initial_answer: string;
  method: string;
  reason: string;
  status: "running" | "completed" | "failed";
  stage: string;
  progress: number;
  sources: Array<{ title: string; url: string }>;
  dataset_path: string;
  environment_id: string;
  run_id: string;
  baseline_artifact_path: string;
  baseline_evaluation: { passed?: boolean; score?: number };
  adapted_artifact_path: string;
  adapted_evaluation: { passed?: boolean; score?: number };
  accepted_adapter_path: string;
  retrieved_skill_ids: string[];
  generated_skill_path: string;
  framework_artifact_path: string;
  framework_evaluation: { passed?: boolean; score?: number };
  final_answer: string;
  error: string;
  events?: LearningEvent[];
}

export interface LearningEvent {
  sequence: number;
  timestamp?: number;
  stage: string;
  message: string;
  progress: number;
  data?: Record<string, unknown>;
}

export interface TasksetInfo {
  id: string;
  name: string;
  package_name: string;
  domain: string;
  description: string;
  num_tasks: number;
  needs_sandbox: boolean;
  tags: string[];
  eval_config: Record<string, number>;
}

export interface EnvironmentTask {
  id?: string;
  name: string;
  prompt: string;
  expected_answer: string;
  criteria: string[];
  difficulty: string;
}

export interface SimulatorAction {
  name: string;
  description: string;
}

export interface SimulatorScenario {
  name: string;
  initial_state: Record<string, string | number | boolean>;
  solution: string[];
}

export interface SimulatorSpec {
  template_id: string;
  observation: string;
  state: Record<string, string | number | boolean>;
  actions: SimulatorAction[];
  max_steps: number;
  scenarios: SimulatorScenario[];
}

export interface SimulationStep {
  session_id?: string;
  observation: string;
  state: Record<string, string | number | boolean>;
  reward: number;
  terminated: boolean;
  success: boolean;
  outcome: string;
  step: number;
  valid: boolean;
  actions: string[];
}

export interface EnvironmentSpec {
  id: string;
  taskset_id: string;
  name: string;
  mode: "IL" | "RL";
  kind?: "task" | "state-machine";
  goal: string;
  description: string;
  domain: string;
  interaction: { observation: string; action: string; max_steps: number };
  reward: { correctness: number; reasoning: number; efficiency: number; method: string };
  tasks: EnvironmentTask[];
  simulator?: SimulatorSpec;
  status: string;
  created_at: number;
  updated_at: number;
}

export interface RunConfig {
  model_id: string;
  taskset_id: string;
  backend: string;
  precision: string;
  sft_iters: number;
  sft_lr: number;
  sft_task_offset: number;
  sft_tasks: number | null;
  grpo_iters: number;
  grpo_group_size: number;
  grpo_lr: number;
  grpo_temperature: number;
  max_seq_length: number;
  benchmark_tasks: number;
  benchmark_batch_size: number;
  rollouts_per_example: number;
  max_reasoning_tokens: number;
  max_answer_tokens: number;
}

export interface RunPreflightCheck {
  id: string;
  label: string;
  status: "pass" | "warn" | "block";
  detail: string;
}

export interface RunPreflight {
  ready: boolean;
  model_id: string;
  taskset_id: string;
  backend: string;
  precision: string;
  checks: RunPreflightCheck[];
}

export interface RunManifest {
  schema_version: number;
  created_at: string;
  git: { revision: string; dirty: boolean | null };
  runtime: { python: string; platform: string; machine: string; packages: Record<string, string | null> };
  config: Record<string, unknown>;
  hardware: Record<string, unknown> | null;
  model: Record<string, unknown> | null;
  taskset: Record<string, unknown> | null;
}

export interface RunState {
  id: string;
  status: string;
  stage: string;
  progress: number;
  started_at: number;
  elapsed_seconds: number;
  manifest: RunManifest;
  metrics: Record<string, number>;
  baseline_accuracy: number;
  post_sft_accuracy: number;
  post_grpo_accuracy: number;
  sft_loss_history: number[];
  grpo_reward_history: number[];
  config: RunConfig;
  artifact_dir: string;
}

export interface LogEvent {
  timestamp: number;
  stage: string;
  level: string;
  message: string;
  data: Record<string, any>;
}

export interface RsiPanel {
  id: string;
  title: string;
  model_id: string;
  workspace: string;
  status: "starting" | "ready" | "running" | "stopped" | "failed";
  session_id: string;
  created_at: number;
  updated_at: number;
  last_error: string;
  pid: number | null;
  events?: RsiEvent[];
}

export interface RsiEvent {
  type: string;
  sequence: number;
  timestamp: number;
  panelId?: string;
  data?: Record<string, unknown>;
  error?: string;
}

const API_BASE = "";

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function getHardware(): Promise<HardwareInfo> {
  return fetchJSON("/api/hardware");
}

export async function getModels(): Promise<ModelInfo[]> {
  return fetchJSON("/api/models");
}

export async function downloadModel(modelId: string, precision?: string): Promise<ModelLocalStatus> {
  const res = await fetch(`/api/models/${modelId}/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ precision }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getModelStatus(modelId: string): Promise<ModelLocalStatus> {
  return fetchJSON(`/api/models/${modelId}/status`);
}

export async function getContextEstimate(modelId: string, contextWindow: number): Promise<ContextEstimate> {
  return fetchJSON(`/api/models/${modelId}/context-estimate?context_window=${contextWindow}`);
}

export async function sendChat(modelId: string, message: string, history: Array<{ role: string; text: string }>, contextWindow: number): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, message, history, context_window: contextWindow, use_tools: true }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getRsiPanels(): Promise<RsiPanel[]> {
  return fetchJSON("/api/rsi/panels");
}

export async function getRsiPanel(panelId: string): Promise<RsiPanel> {
  return fetchJSON(`/api/rsi/panels/${panelId}`);
}

export async function createRsiPanels(modelId: string, count: number, task = ""): Promise<RsiPanel[]> {
  const res = await fetch("/api/rsi/panels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, count, task }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function promptRsiPanel(panelId: string, prompt: string): Promise<RsiPanel> {
  const res = await fetch(`/api/rsi/panels/${panelId}/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stopRsiPanel(panelId: string): Promise<RsiPanel> {
  const res = await fetch(`/api/rsi/panels/${panelId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function streamRsiEvents(panelId: string, after: number, onEvent: (event: RsiEvent) => void): EventSource {
  const source = new EventSource(`/api/rsi/panels/${panelId}/events?after=${after}`);
  source.onmessage = (message) => onEvent(JSON.parse(message.data) as RsiEvent);
  return source;
}

// ---------------------------------------------------------------------------
// RSI Loops — configurable self-improvement loops
// ---------------------------------------------------------------------------

export interface RsiLoopTemplate {
  id: string;
  kind: string;
  name: string;
  objective: string;
  default_minutes: number;
  default_iterations: number;
  needs_sandbox: boolean;
}

export interface RsiLoop {
  id: string;
  name: string;
  kind: string;
  model_id: string;
  objective: string;
  time_budget_minutes: number;
  max_iterations: number;
  status: "draft" | "running" | "completed" | "stopped" | "failed";
  panel_id: string;
  iterations_done: number;
  best_score: number;
  score_history: number[];
  created_at: number;
  updated_at: number;
  last_error: string;
  panel_status?: string;
  panel_events?: RsiEvent[];
}

export interface RunnabilityScore {
  score: number;
  label: string;
  factors: {
    model_fit: number;
    time_feasibility: number;
    memory_headroom: number;
    estimated_tps: number;
    feasible_iterations: number;
    needs_sandbox: boolean;
  };
}

export async function getRsiLoops(): Promise<{ loops: RsiLoop[]; templates: RsiLoopTemplate[]; kinds: string[] }> {
  return fetchJSON("/api/rsi/loops");
}

export async function getRsiLoop(loopId: string): Promise<RsiLoop> {
  return fetchJSON(`/api/rsi/loops/${loopId}`);
}

export async function createRsiLoop(loop: Partial<RsiLoop>): Promise<RsiLoop> {
  const res = await fetch("/api/rsi/loops", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(loop),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stopRsiLoop(loopId: string): Promise<RsiLoop> {
  const res = await fetch(`/api/rsi/loops/${loopId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getLoopRunnability(config: { model_id: string; time_budget_minutes: number; max_iterations: number; needs_sandbox: boolean; context_window?: number }): Promise<RunnabilityScore> {
  const res = await fetch("/api/rsi/loops/runnability", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getLearningSession(sessionId: string): Promise<LearningSession> {
  return fetchJSON(`/api/learning/${sessionId}`);
}

export function streamLearningEvents(sessionId: string, onEvent: (event: LearningEvent) => void): EventSource {
  const source = new EventSource(`/api/learning/${sessionId}/events`);
  source.onmessage = (message) => onEvent(JSON.parse(message.data) as LearningEvent);
  return source;
}

export async function getEnvironments(): Promise<EnvironmentSpec[]> {
  return fetchJSON("/api/environments");
}

export async function getEnvironment(environmentId: string): Promise<EnvironmentSpec> {
  return fetchJSON(`/api/environments/${environmentId}`);
}

export async function saveEnvironment(environment: Partial<EnvironmentSpec>): Promise<EnvironmentSpec> {
  const res = await fetch("/api/environments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(environment),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteEnvironment(environmentId: string): Promise<void> {
  const res = await fetch(`/api/environments/${environmentId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

export async function createEnvironmentFromChat(mode: "IL" | "RL", description: string, modelId: string): Promise<EnvironmentSpec> {
  const res = await fetch("/api/environments/from-chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, description, model_id: modelId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resetSimulation(environmentId: string, scenario = 0): Promise<SimulationStep> {
  const res = await fetch(`/api/environments/${environmentId}/simulate/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stepSimulation(environmentId: string, sessionId: string, action: string): Promise<SimulationStep> {
  const res = await fetch(`/api/environments/${environmentId}/simulate/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, action }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTasksets(): Promise<TasksetInfo[]> {
  return fetchJSON("/api/tasksets");
}

export async function getRuns(): Promise<RunState[]> {
  return fetchJSON("/api/runs");
}

export async function getRun(id: string): Promise<RunState> {
  return fetchJSON(`/api/runs/${id}`);
}

export async function preflightRun(config: Partial<RunConfig>): Promise<RunPreflight> {
  const res = await fetch("/api/runs/preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Preflight failed: ${await res.text()}`);
  return res.json();
}

export async function createRun(config: Partial<RunConfig>): Promise<{ id: string; status: string }> {
  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to create run: ${await res.text()}`);
  return res.json();
}

export function streamRunEvents(
  runId: string,
  onEvent: (event: LogEvent) => void,
  onError?: (err: Event) => void
): EventSource {
  const es = new EventSource(`/api/runs/${runId}/events`);
  es.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data) as LogEvent;
      onEvent(event);
    } catch (err) {
      console.error("Failed to parse event:", err);
    }
  };
  if (onError) es.onerror = onError;
  return es;
}

// ---------------------------------------------------------------------------
// Harness graph — algorithmic harness self-improvement
// ---------------------------------------------------------------------------

export interface ActionNode {
  key: string;
  action_type: string;
  category: "action" | "mistake";
  label: string;
  weight: number;
  observations: number;
  successes: number;
  failures: number;
  ema_success: number;
  last_seen: number;
}

export interface ActionEdge {
  source: string;
  target: string;
  co_occurrences: number;
  joint_successes: number;
  weight: number;
}

export interface TaskNode {
  id: string;
  kind: string;
  success: boolean;
  score: number;
  action_keys: string[];
  created_at: number;
}

export interface HarnessGraph {
  nodes: ActionNode[];
  edges: ActionEdge[];
  tasks: TaskNode[];
  total_tasks: number;
  total_actions: number;
  total_edges: number;
  pending_tasks: number;
}

export interface EfficiencySnapshot {
  timestamp: number;
  efficiency: number;
  total_actions: number;
  total_tasks: number;
  success_rate: number;
}

export async function getHarnessGraph(): Promise<HarnessGraph> {
  return fetchJSON("/api/harness-graph");
}

export async function getHarnessGraphEfficiency(limit = 500): Promise<EfficiencySnapshot[]> {
  return fetchJSON(`/api/harness-graph/efficiency?limit=${limit}`);
}

export async function getHarnessGraphTopActions(limit = 20): Promise<ActionNode[]> {
  return fetchJSON(`/api/harness-graph/top-actions?limit=${limit}`);
}

export async function ingestToolLogs(): Promise<{ ingested: number }> {
  const res = await fetch("/api/harness-graph/ingest-tool-logs", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resetHarnessGraph(): Promise<{ status: string }> {
  const res = await fetch("/api/harness-graph", { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
