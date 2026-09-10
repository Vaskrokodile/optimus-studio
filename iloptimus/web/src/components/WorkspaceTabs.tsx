import { useEffect, useRef, useState } from "react";
import { Bot, ChevronDown, FlaskConical, Layers3, MessageSquare, Plus, Workflow, X } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { getRsiPanels, stopRsiPanel, type RsiPanel } from "../api/client";

export const WORKSPACE_TABS_EVENT = "iloptimus:workspace-tabs";

export function refreshWorkspaceTabs() {
  window.dispatchEvent(new Event(WORKSPACE_TABS_EVENT));
}

export default function WorkspaceTabs() {
  const [panels, setPanels] = useState<RsiPanel[]>([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const menuRef = useRef<HTMLDivElement>(null);

  const refresh = () => getRsiPanels().then(setPanels).catch(() => setPanels([]));

  useEffect(() => {
    refresh();
    window.addEventListener(WORKSPACE_TABS_EVENT, refresh);
    return () => window.removeEventListener(WORKSPACE_TABS_EVENT, refresh);
  }, []);

  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const closePanel = async (event: React.MouseEvent, panel: RsiPanel) => {
    event.stopPropagation();
    if (["starting", "ready", "running"].includes(panel.status)) await stopRsiPanel(panel.id).catch(() => undefined);
    setPanels((current) => current.filter((item) => item.id !== panel.id));
    if (location.pathname === `/rsi/${panel.id}`) navigate("/");
  };

  const createItems = [
    { label: "RSI agent panel", detail: "A persistent local coding agent", icon: Bot, action: () => navigate("/?action=new-rsi") },
    { label: "Training run", detail: "Configure a supervised or RL run", icon: FlaskConical, action: () => navigate("/studio") },
    { label: "IL / RL pipeline", detail: "Build a no-code environment", icon: Workflow, action: () => navigate("/pipelines") },
    { label: "Environment", detail: "Open saved environments", icon: Layers3, action: () => navigate("/environments") },
  ];

  return (
    <div className="workspace-tabs-shell">
      <div className="workspace-tabs" role="tablist" aria-label="Open workspaces">
        <button className={`workspace-tab ${location.pathname === "/" ? "active" : ""}`} onClick={() => navigate("/")} role="tab">
          <MessageSquare /><span>Chat</span>
        </button>
        {panels.map((panel) => (
          <button key={panel.id} className={`workspace-tab ${location.pathname === `/rsi/${panel.id}` ? "active" : ""}`} onClick={() => navigate(`/rsi/${panel.id}`)} role="tab">
            <span className={`panel-live-dot ${panel.status}`} /><Bot /><span>{panel.title}</span>
            <span className="workspace-tab-close" onClick={(event) => closePanel(event, panel)} role="button" aria-label={`Close ${panel.title}`}><X /></span>
          </button>
        ))}
        <div className="workspace-add" ref={menuRef}>
          <button className="workspace-add-button" onClick={() => setMenuOpen((open) => !open)} aria-label="Create workspace" aria-expanded={menuOpen}><Plus /><ChevronDown /></button>
          {menuOpen && <div className="workspace-add-menu">
            <div className="workspace-add-label">Create workspace</div>
            {createItems.map(({ label, detail, icon: Icon, action }) => <button key={label} onClick={() => { setMenuOpen(false); action(); }}><Icon /><span><strong>{label}</strong><small>{detail}</small></span></button>)}
          </div>}
        </div>
      </div>
      <div id="workspace-tabs-trailer" className="workspace-tabs-trailer" />
    </div>
  );
}
