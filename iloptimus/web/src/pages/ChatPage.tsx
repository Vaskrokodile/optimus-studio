import { CSSProperties, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowUp, Bot, BrainCircuit, Check, ChevronDown, Copy, Database, ExternalLink, Globe2, Paperclip, RotateCcw, SlidersHorizontal, User } from "lucide-react";
import { createEnvironmentFromChat, createRsiPanels, getContextEstimate, getLearningSession, getModels, sendChat, streamLearningEvents, type ContextEstimate, type LearningSession, type ModelInfo } from "../api/client";
import { refreshWorkspaceTabs } from "../components/WorkspaceTabs";

type Message = { role: "user" | "assistant"; text: string; reasoning?: string; skills?: string[]; tools?: string[]; tps?: number; panelIds?: string[]; learningId?: string };

const starters = [
  "Design an IL pipeline for mathematical reasoning",
  "Which local model fits my hardware?",
  "Explain the difference between IL and RL",
];

const previousChats: Record<string, Message[]> = {
  "0": [
    { role: "user", text: "How should I design a reward for reasoning quality?" },
    { role: "assistant", text: "Use correctness as a hard gate, then reward the qualities you want the model to internalize: verification, efficient decomposition, and calibrated uncertainty. A practical IL reward is correctness × (0.6 + 0.4 × reasoning quality), so wrong answers never receive signal while good reasoning separates correct traces." },
  ],
  "1": [
    { role: "user", text: "Compare Qwen and Llama for a small local training run." },
    { role: "assistant", text: "Qwen is often the stronger starting point for compact reasoning and coding experiments, while Llama offers a broad ecosystem and familiar tooling. In Model Library, choose the best quantization for your detected memory, then send it into Optimus Lab." },
  ],
};

const slashCommands = [
  { command: "/il", title: "Create an IL environment", description: "Describe a capability to teach through ideal demonstrations", kind: "prompt" },
  { command: "/rl", title: "Create an RL environment", description: "Describe a world, its goal, and the reward signal", kind: "prompt" },
  { command: "/rsi", title: "Launch RSI agent panels", description: "Example: /rsi 3 — open three persistent coding agents", kind: "prompt" },
  { command: "/learn", title: "Research and learn", description: "Verify a question, build grounded data, and adapt when appropriate", kind: "prompt" },
  { command: "/models", title: "Open Model Library", description: "Choose which local model to chat with", kind: "navigate", to: "/models" },
  { command: "/lab", title: "Open Optimus Lab", description: "Configure and launch a training run", kind: "navigate", to: "/studio" },
  { command: "/clear", title: "Clear this chat", description: "Start again without changing the selected model", kind: "clear" },
];

export default function ChatPage() {
  const [params] = useSearchParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const savedModel = (() => { try { return JSON.parse(localStorage.getItem("iloptimus-chat-model") || "null"); } catch { return null; } })();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState(savedModel?.id || "qwen2.5-1.5b");
  const [thinking, setThinking] = useState(false);
  const [commandIndex, setCommandIndex] = useState(0);
  const [contextOpen, setContextOpen] = useState(false);
  const [contextWindow, setContextWindow] = useState(() => Number(localStorage.getItem("iloptimus-context-window")) || 4096);
  const [contextEstimate, setContextEstimate] = useState<ContextEstimate | null>(null);
  const [lastContextTokens, setLastContextTokens] = useState(0);
  const [copiedMessage, setCopiedMessage] = useState<number | null>(null);
  const [learningSessions, setLearningSessions] = useState<Record<string, LearningSession>>({});
  const endRef = useRef<HTMLDivElement>(null);
  const learningStreams = useRef<Record<string, EventSource>>({});
  const completedLearning = useRef<Set<string>>(new Set());
  const navigate = useNavigate();
  const model = models.find((item) => item.id === modelId)?.name || savedModel?.name || "Qwen2.5-1.5B";
  const modelInfo = models.find((item) => item.id === modelId);
  const commandQuery = input.startsWith("/") && !input.includes(" ") ? input.slice(1).toLowerCase() : null;
  const visibleCommands = commandQuery === null ? [] : slashCommands.filter((item) => item.command.slice(1).startsWith(commandQuery));
  const commandPaletteOpen = visibleCommands.length > 0;
  const approximateTokens = useMemo(() => Math.ceil((messages.reduce((total, message) => total + message.text.length, 0) + input.length) / 3.5), [messages, input]);
  const usedContextTokens = Math.max(lastContextTokens, approximateTokens);
  const contextRatio = Math.min(1, usedContextTokens / Math.max(contextWindow, 1));
  const maxContext = Math.max(2048, Math.min(modelInfo?.context_length || 32768, contextEstimate?.max_safe_context || modelInfo?.context_length || 32768));

  useEffect(() => {
    const chat = params.get("chat");
    setMessages(chat ? previousChats[chat] || [] : []);
    setLastContextTokens(0);
    if (params.get("action") === "new-rsi") setInput("Launch 1 RSI agent panel");
  }, [params]);

  useEffect(() => { getModels().then((items) => {
    setModels(items);
    const installed = items.filter((item) => item.local.status === "downloaded");
    if (!installed.some((item) => item.id === modelId) && installed[0]) setModelId(installed[0].id);
  }).catch(() => setModels([])); }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => () => Object.values(learningStreams.current).forEach((source) => source.close()), []);

  useEffect(() => {
    if (!modelId) return;
    const timer = window.setTimeout(() => {
      getContextEstimate(modelId, contextWindow).then(setContextEstimate).catch(() => setContextEstimate(null));
    }, 180);
    return () => window.clearTimeout(timer);
  }, [modelId, contextWindow]);

  useEffect(() => {
    if (contextWindow > maxContext) setContextWindow(maxContext);
  }, [contextWindow, maxContext]);

  const send = async (text = input) => {
    const clean = text.trim();
    if (!clean || thinking) return;
    setMessages((current) => [...current, { role: "user", text: clean }]);
    setInput("");
    setThinking(true);
    try {
      const rsiSlash = clean.match(/^\/rsi(?:\s+(\d+))?(?:\s+(.+))?$/i);
      const rsiNatural = clean.match(/\b(?:launch|open|start|create)\s+(?:(\d+)\s+)?(?:parallel\s+)?rsi\s+(?:agent\s+)?panels?\b(?:\s+(?:to|and)\s+(.+))?/i);
      const rsiCommand = rsiSlash || rsiNatural;
      if (rsiCommand) {
        const count = Math.max(1, Math.min(6, Number(rsiCommand[1] || 1)));
        const panels = await createRsiPanels(modelId, count, rsiCommand[2]?.trim() || "");
        refreshWorkspaceTabs();
        setMessages((current) => [...current, {
          role: "assistant",
          text: `${count === 1 ? "Your RSI agent panel is" : `${panels.length} parallel RSI agent panels are`} ready. Each is an isolated, persistent process using ${model}; open ${count === 1 ? "it" : "a panel"} to give it a file or coding task.`,
          panelIds: panels.map((panel) => panel.id),
        }]);
        return;
      }
      const command = clean.match(/^\/(il|rl)\s+(.+)/i);
      if (command) {
        const environment = await createEnvironmentFromChat(command[1].toUpperCase() as "IL" | "RL", command[2], modelId);
        setMessages((current) => [...current, { role: "assistant", text: `${environment.name} is ready. I created ${environment.tasks.length} gradable tasks, a ${environment.mode} reward design, and a taskset you can use in Optimus Lab. You’ll find it in My Environments.` }]);
        return;
      }
      const response = await sendChat(modelId, clean, messages, contextWindow);
      setLastContextTokens(response.context_tokens);
      const learningId = response.learning_session?.id;
      setMessages((current) => [...current, {
        role: "assistant",
        text: response.answer,
        reasoning: response.reasoning || undefined,
        skills: response.active_skills.map((skill) => skill.name),
        tools: response.tool_calls.map((tool) => tool.name),
        tps: response.tokens_per_sec,
        learningId,
      }]);
      if (response.learning_session) {
        const session = response.learning_session;
        setLearningSessions((current) => ({ ...current, [session.id]: session }));
        learningStreams.current[session.id]?.close();
        learningStreams.current[session.id] = streamLearningEvents(session.id, async (event) => {
          setLearningSessions((current) => ({
            ...current,
            [session.id]: { ...current[session.id], stage: event.stage, progress: event.progress },
          }));
          if ((event.stage === "completed" || event.stage === "failed") && !completedLearning.current.has(session.id)) {
            completedLearning.current.add(session.id);
            const final = await getLearningSession(session.id).catch(() => null);
            if (final) {
              setLearningSessions((current) => ({ ...current, [session.id]: final }));
              if (final.final_answer) setMessages((current) => [...current, { role: "assistant", text: final.final_answer, learningId: final.id }]);
            }
            learningStreams.current[session.id]?.close();
          }
        });
      }
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message.replace(/^\{"detail":"?|"\}$/g, "") : "The local request failed";
      setMessages((current) => [...current, { role: "assistant", text: clean.startsWith("/") ? `I could not build that environment: ${detail}` : detail }]);
    } finally {
      setThinking(false);
    }
  };

  const copyText = async (text: string, index: number) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const field = document.createElement("textarea");
      field.value = text;
      field.style.position = "fixed";
      field.style.opacity = "0";
      document.body.appendChild(field);
      field.select();
      document.execCommand("copy");
      field.remove();
    }
    setCopiedMessage(index);
    window.setTimeout(() => setCopiedMessage((current) => current === index ? null : current), 1400);
  };

  const runCommand = (item: typeof slashCommands[number]) => {
    if (item.kind === "prompt") { setInput(`${item.command} `); return; }
    if (item.kind === "navigate" && item.to) { navigate(item.to); return; }
    if (item.kind === "clear") { setMessages([]); setInput(""); setLastContextTokens(0); }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (commandPaletteOpen && event.key === "ArrowDown") { event.preventDefault(); setCommandIndex((index) => (index + 1) % visibleCommands.length); return; }
    if (commandPaletteOpen && event.key === "ArrowUp") { event.preventDefault(); setCommandIndex((index) => (index - 1 + visibleCommands.length) % visibleCommands.length); return; }
    if (commandPaletteOpen && event.key === "Escape") { event.preventDefault(); setInput(""); return; }
    if (commandPaletteOpen && event.key === "Enter") { event.preventDefault(); runCommand(visibleCommands[Math.min(commandIndex, visibleCommands.length - 1)]); setCommandIndex(0); return; }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  return (
    <section className="chat-page">
      <div className="chat-space" aria-hidden="true">
        <video className="chat-space-video" src="/blackhole.mp4" autoPlay loop muted playsInline />
        <div className="chat-space-tint" />
      </div>
      <header className="chat-header">
        <div className="model-status"><span className="status-dot" /><select value={modelId} onChange={(event) => { const next = models.find((item) => item.id === event.target.value); setModelId(event.target.value); if (next) localStorage.setItem("iloptimus-chat-model", JSON.stringify({ id: next.id, name: next.name })); }} aria-label="Active model">{models.some((item) => item.local.status === "downloaded") ? models.filter((item) => item.local.status === "downloaded").map((item) => <option key={item.id} value={item.id}>{item.name}</option>) : <option value="">Download a model first</option>}</select><ChevronDown /></div>
        <button className="header-action" onClick={() => navigate("/models")}><SlidersHorizontal /> Model library</button>
      </header>

      <div className={`chat-thread ${messages.length === 0 ? "empty-thread" : ""}`}>
        {messages.length === 0 ? (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="welcome-state">
            <div className="welcome-title-row">
              <h1>What do you want<br />to do today?</h1>
              <div className="mascot-stage" aria-hidden="true"><img src="/wolf-mascot-v2.png" alt="" /></div>
            </div>
            <p className="welcome-copy">Chat with local models, explore ideas, and turn the best conversations into trainable intuition.</p>
            <div className="starter-grid">
              {starters.map((starter) => <button key={starter} onClick={() => send(starter)}>{starter}<ArrowUp /></button>)}
            </div>
          </motion.div>
        ) : (
          <div className="message-list">
            {messages.map((message, index) => (
              <motion.article key={`${message.role}-${index}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`message ${message.role}`}>
                <div className="message-avatar">{message.role === "assistant" ? <Bot /> : <User />}</div>
              <div><div className="message-meta">{message.role === "assistant" ? model : "You"}</div><p>{message.text}</p>{message.panelIds?.length ? <div className="rsi-panel-links">{message.panelIds.map((panelId, panelIndex) => <button key={panelId} onClick={() => navigate(`/rsi/${panelId}`)}><span><Bot />RSI Agent {panelIndex + 1}</span><ExternalLink /></button>)}</div> : null}{message.learningId && learningSessions[message.learningId] && learningSessions[message.learningId].status === "running" ? <div className="learning-card"><div className="learning-card-head"><span><BrainCircuit />Test-time learning</span><strong>{Math.round(learningSessions[message.learningId].progress * 100)}%</strong></div><div className="learning-track"><span style={{ width: `${learningSessions[message.learningId].progress * 100}%` }} /></div><div className="learning-stage"><span className="learning-orbit"><i /><i /><i /></span><div><strong>{learningSessions[message.learningId].stage.replaceAll("-", " ")}</strong><small>{learningSessions[message.learningId].method === "retrieval" ? "Fresh facts stay source-grounded" : "Scene contract → verifier → curated adaptation only if needed"}</small></div></div><div className="learning-mini-steps"><span><Globe2 />Design</span><span><Database />Verify</span><span><BrainCircuit />Adapt if needed</span></div></div> : null}{message.learningId && learningSessions[message.learningId]?.status === "completed" && (learningSessions[message.learningId].baseline_evaluation?.passed || learningSessions[message.learningId].accepted_adapter_path || learningSessions[message.learningId].framework_evaluation?.passed) ? <div className="learning-artifact"><span><BrainCircuit />{learningSessions[message.learningId].baseline_evaluation?.passed ? "Local model scene accepted" : learningSessions[message.learningId].accepted_adapter_path ? "Adapted artifact accepted" : "Verified framework fallback"}{learningSessions[message.learningId].generated_skill_path ? " · failure skill stored" : ""}</span><a href={`/api/learning/${message.learningId}/artifact/${learningSessions[message.learningId].baseline_evaluation?.passed ? "baseline" : learningSessions[message.learningId].accepted_adapter_path ? "adapted" : "framework"}`} target="_blank" rel="noreferrer">Open artifact <ExternalLink /></a></div> : null}{message.role === "assistant" && <>{(message.skills?.length || message.tools?.length || message.tps) && <div className="agent-activity">{message.skills?.map((skill) => <span key={skill}>Skill · {skill}</span>)}{message.tools?.map((tool) => <span key={tool}>Tool · {tool}</span>)}{message.tps ? <span>{message.tps.toFixed(1)} tok/s</span> : null}</div>}<div className="message-tools"><button type="button" onClick={() => copyText(message.text, index)} aria-label={copiedMessage === index ? "Copied" : "Copy response"} title={copiedMessage === index ? "Copied" : "Copy response"}>{copiedMessage === index ? <Check /> : <Copy />}</button><button type="button" aria-label="Regenerate"><RotateCcw /></button></div></>}</div>
              </motion.article>
            ))}
            {thinking && <div className="thinking"><span /><span /><span /></div>}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <form className="composer" onSubmit={(event: FormEvent) => { event.preventDefault(); send(); }}>
        {commandPaletteOpen && <div className="command-palette" role="listbox" aria-label="Slash commands"><div className="command-palette-label">Commands</div>{visibleCommands.map((item,index)=><button type="button" key={item.command} className={index===commandIndex?"active":""} onMouseDown={(event)=>{event.preventDefault();runCommand(item);setCommandIndex(0)}}><code>{item.command}</code><span><strong>{item.title}</strong><small>{item.description}</small></span><em>↵</em></button>)}</div>}
        <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={onKeyDown} placeholder="Message your model…" rows={1} aria-label="Message" />
        {contextOpen && <div className="context-popover" role="dialog" aria-label="Context window settings">
          <div className="tps-estimator"><span>TPS estimator</span><strong>{contextEstimate ? `~${contextEstimate.estimated_tps.toFixed(1)} tok/s` : "Calculating…"}</strong></div>
          {contextEstimate && <p>{contextEstimate.low_tps.toFixed(1)}–{contextEstimate.high_tps.toFixed(1)} tok/s · {contextEstimate.basis}</p>}
          <input type="range" min={2048} max={maxContext} step={1024} value={Math.min(contextWindow, maxContext)} style={{ "--slider-progress": `${Math.max(0, Math.min(100, ((contextWindow - 2048) / Math.max(1, maxContext - 2048)) * 100))}%` } as CSSProperties} onChange={(event) => { const value = Number(event.target.value); setContextWindow(value); localStorage.setItem("iloptimus-context-window", String(value)); }} aria-label="Context window size" />
          <div className="context-scale"><span>{usedContextTokens.toLocaleString()} used</span><strong>{contextWindow.toLocaleString()} tokens</strong><span>{maxContext.toLocaleString()} max</span></div>
          {contextEstimate && !contextEstimate.fits_in_memory && <small>This selection may use system memory and run much slower.</small>}
        </div>}
        <div className="composer-tools"><button type="button" className="attach" aria-label="Attach file"><Paperclip /></button><button type="button" className={`context-meter ${usedContextTokens ? "active" : ""}`} style={{ "--context-angle": `${contextRatio ? Math.max(12, contextRatio * 360) : 0}deg` } as CSSProperties} onClick={() => setContextOpen((open) => !open)} aria-label={`Context window: ${usedContextTokens} of ${contextWindow} tokens`} aria-expanded={contextOpen}><span /></button><span>Context {Math.round(contextRatio * 100)}%</span><div className="command-hints"><button type="button" onClick={()=>setInput("/il ")}>/il</button><button type="button" onClick={()=>setInput("/rl ")}>/rl</button></div></div>
        <button className="send-button" disabled={!input.trim() || thinking || !models.some((item) => item.id === modelId && item.local.status === "downloaded")} aria-label="Send message"><ArrowUp /></button>
      </form>
      <p className="chat-footnote">Local models can make mistakes. Verify important outputs.</p>
    </section>
  );
}
