import { FormEvent, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BrainCircuit, Code2, Cpu, ExternalLink, Gauge, Hourglass, Infinity as InfinityIcon, Sigma, Repeat, Square, TerminalSquare, Wrench, Zap } from "lucide-react";
import {
  createRsiLoop,
  getHardware,
  getLoopRunnability,
  getModels,
  getRsiLoops,
  stopRsiLoop,
  type HardwareInfo,
  type ModelInfo,
  type RsiLoop,
  type RsiLoopTemplate,
  type RunnabilityScore,
} from "../api/client";
import { refreshWorkspaceTabs } from "../components/WorkspaceTabs";

const KIND_ICONS: Record<string, typeof Code2> = {
  coding: Code2,
  reasoning: BrainCircuit,
  math: Sigma,
  "tool-calling": Wrench,
  agentic: TerminalSquare,
};

function RunnabilityBar({ score, label }: { score: number; label: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  // green -> amber -> red gradient stop based on score
  const hue = Math.round(pct * 1.2); // 0 = red, 120 = green
  return (
    <div className="loop-runnability" title={`Runnability: ${pct}% (${label})`}>
      <div className="loop-runnability-head">
        <span><Gauge /> Runnability</span>
        <strong style={{ color: `hsl(${hue} 70% 55%)` }}>{pct}% · {label}</strong>
      </div>
      <div className="loop-runnability-track">
        <div className="loop-runnability-fill" style={{ width: `${pct}%`, background: `linear-gradient(90deg, hsl(${Math.max(0, hue - 30)} 75% 50%), hsl(${hue} 70% 52%))` }} />
        <div className="loop-runnability-ticks"><i /><i /><i /><i /></div>
      </div>
      <div className="loop-runnability-scale"><span>not runnable</span><span>tight</span><span>good</span><span>excellent</span></div>
    </div>
  );
}

function LoopCard({ loop, onStop, onOpen }: { loop: RsiLoop; onStop: (id: string) => void; onOpen: (panelId: string) => void }) {
  const Icon = KIND_ICONS[loop.kind] || Repeat;
  return (
    <div className="loop-card">
      <div className="loop-card-head">
        <span className="loop-kind-chip"><Icon />{loop.kind}</span>
        <span className={`panel-live-dot ${loop.panel_status || loop.status}`} />
      </div>
      <h3>{loop.name}</h3>
      <p>{loop.objective}</p>
      <div className="loop-card-meta">
        <span><Cpu />{loop.model_id}</span>
        <span><Hourglass />{loop.time_budget_minutes}m</span>
        <span><InfinityIcon />{loop.iterations_done}/{loop.max_iterations} iters</span>
      </div>
      <div className="loop-card-actions">
        {loop.panel_id && <button onClick={() => onOpen(loop.panel_id)}><ExternalLink /> Open agent</button>}
        {loop.status === "running" && <button className="danger" onClick={() => onStop(loop.id)}><Square /> Stop loop</button>}
      </div>
    </div>
  );
}

export default function RsiLoopsPage() {
  const [loops, setLoops] = useState<RsiLoop[]>([]);
  const [templates, setTemplates] = useState<RsiLoopTemplate[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [kindFilter, setKindFilter] = useState<string>("all");
  const [selectedTemplate, setSelectedTemplate] = useState<RsiLoopTemplate | null>(null);
  const [name, setName] = useState("");
  const [modelId, setModelId] = useState("");
  const [objective, setObjective] = useState("");
  const [minutes, setMinutes] = useState(30);
  const [iterations, setIterations] = useState(5);
  const [runnability, setRunnability] = useState<RunnabilityScore | null>(null);
  const [error, setError] = useState("");
  const [launching, setLaunching] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    getRsiLoops().then((data) => { setLoops(data.loops); setTemplates(data.templates); }).catch(() => undefined);
    getModels().then((items) => {
      setModels(items);
      const installed = items.find((item) => item.local.status === "downloaded");
      if (installed) setModelId((current) => current || installed.id);
    }).catch(() => undefined);
    getHardware().then(setHardware).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!modelId) return;
    const timer = window.setTimeout(() => {
      getLoopRunnability({
        model_id: modelId,
        time_budget_minutes: minutes,
        max_iterations: iterations,
        needs_sandbox: selectedTemplate?.needs_sandbox ?? false,
      }).then(setRunnability).catch(() => setRunnability(null));
    }, 200);
    return () => window.clearTimeout(timer);
  }, [modelId, minutes, iterations, selectedTemplate]);

  const visibleLoops = useMemo(
    () => (kindFilter === "all" ? loops : loops.filter((loop) => loop.kind === kindFilter)),
    [loops, kindFilter],
  );
  const visibleTemplates = useMemo(
    () => (kindFilter === "all" ? templates : templates.filter((template) => template.kind === kindFilter)),
    [templates, kindFilter],
  );
  const downloadedModels = models.filter((item) => item.local.status === "downloaded");

  const pickTemplate = (template: RsiLoopTemplate) => {
    setSelectedTemplate(template);
    setName(template.name);
    setObjective(template.objective);
    setMinutes(template.default_minutes);
    setIterations(template.default_iterations);
    setError("");
  };

  const launch = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedTemplate || !modelId || !objective.trim()) return;
    setLaunching(true);
    setError("");
    try {
      const loop = await createRsiLoop({
        name: name.trim() || selectedTemplate.name,
        kind: selectedTemplate.kind,
        model_id: modelId,
        objective: objective.trim(),
        time_budget_minutes: minutes,
        max_iterations: iterations,
      });
      setLoops((current) => [...current, loop]);
      setSelectedTemplate(null);
      refreshWorkspaceTabs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not launch the loop");
    } finally {
      setLaunching(false);
    }
  };

  const stopLoop = async (loopId: string) => {
    try {
      const updated = await stopRsiLoop(loopId);
      setLoops((current) => current.map((item) => (item.id === loopId ? updated : item)));
      refreshWorkspaceTabs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not stop the loop");
    }
  };

  const openPanel = (panelId: string) => navigate(`/rsi/${panelId}`);

  return (
    <section className="product-page loops-page">
      <header className="product-hero">
        <div>
          <span className="product-kicker"><Repeat /> Recursive self-improvement</span>
          <h1>RSI Loops</h1>
          <p>Configure a self-improvement loop: pick a model, give it an objective, a time budget and an iteration cap, then let it run on a persistent agent workspace.</p>
        </div>
        {hardware && (
          <div className="hardware-pill">
            <span className="live-dot" />
            <div>
              <small>This machine</small>
              <strong>{hardware.gpu.name || "CPU only"} · {hardware.ram_gb}GB RAM{hardware.gpu.vram_gb ? ` · ${hardware.gpu.vram_gb}GB VRAM` : ""}</strong>
            </div>
          </div>
        )}
      </header>

      <div className="loop-filter-row">
        {["all", "coding", "reasoning", "math", "tool-calling", "agentic"].map((kind) => (
          <button key={kind} className={kindFilter === kind ? "selected" : ""} onClick={() => setKindFilter(kind)}>
            {kind === "all" ? <Repeat /> : (() => { const Icon = KIND_ICONS[kind] || Repeat; return <Icon />; })()}
            {kind === "all" ? "All loops" : kind}
          </button>
        ))}
      </div>

      <div className="loops-layout">
        <div className="loops-main">
          <h2 className="loops-section-title">Start a loop</h2>
          <div className="loop-template-grid">
            {visibleTemplates.map((template) => {
              const Icon = KIND_ICONS[template.kind] || Repeat;
              return (
                <button key={template.id} className={`loop-template ${selectedTemplate?.id === template.id ? "selected" : ""}`} onClick={() => pickTemplate(template)}>
                  <span className="loop-kind-chip"><Icon />{template.kind}</span>
                  <strong>{template.name}</strong>
                  <small>{template.objective}</small>
                </button>
              );
            })}
          </div>

          {selectedTemplate && (
            <form className="loop-config" onSubmit={launch}>
              <h2 className="loops-section-title">Customize “{selectedTemplate.name}”</h2>
              <div className="loop-config-grid">
                <label>Loop name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Nightly coding fixes" /></label>
                <label>Model
                  <select value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="Loop model">
                    {downloadedModels.length === 0 && <option value="">Download a model first</option>}
                    {downloadedModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
                  </select>
                </label>
              </div>
              <label>Objective<textarea value={objective} onChange={(e) => setObjective(e.target.value)} rows={3} placeholder="What should the loop achieve every iteration?" /></label>
              <div className="loop-config-grid">
                <label>Time budget — <strong>{minutes} min</strong>
                  <input type="range" min={5} max={240} step={5} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} />
                </label>
                <label>Max iterations — <strong>{iterations}</strong>
                  <input type="range" min={1} max={30} value={iterations} onChange={(e) => setIterations(Number(e.target.value))} />
                </label>
              </div>

              {runnability && <RunnabilityBar score={runnability.score} label={runnability.label} />}
              {runnability && (
                <div className="loop-factors">
                  <span><Zap />{runnability.factors.estimated_tps.toFixed(1)} tok/s</span>
                  <span>model fit {Math.round(runnability.factors.model_fit * 100)}%</span>
                  <span>headroom {Math.round(runnability.factors.memory_headroom * 100)}%</span>
                  <span>~{runnability.factors.feasible_iterations.toFixed(1)} feasible iters</span>
                </div>
              )}

              {error && <div className="loop-error">{error}</div>}
              <button className="primary-action" type="submit" disabled={launching || !modelId || !objective.trim()}>
                <Repeat /> {launching ? "Launching…" : "Launch loop"}
              </button>
            </form>
          )}

          {loops.length > 0 && (
            <>
              <h2 className="loops-section-title">Your loops</h2>
              <div className="loop-grid">
                {visibleLoops.map((loop) => <LoopCard key={loop.id} loop={loop} onStop={stopLoop} onOpen={openPanel} />)}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
