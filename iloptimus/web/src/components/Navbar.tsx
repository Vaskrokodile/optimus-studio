import { NavLink, useNavigate } from "react-router-dom";
import { useState } from "react";
import {
  BookOpen,
  Cpu,
  FlaskConical,
  MessageSquarePlus,
  Network,
  Workflow,
  Layers3,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
  Repeat,
  Share2,
} from "lucide-react";
import ThemeToggle from "./ThemeToggle";

const navItems = [
  { to: "/studio", label: "Optimus Lab", icon: FlaskConical },
  { to: "/models", label: "Model library", icon: Cpu },
  { to: "/pipelines", label: "IL / RL", icon: Workflow },
  { to: "/environments", label: "My environments", icon: Layers3 },
  { to: "/research/sakura-island", label: "Research paper", icon: BookOpen },
  { to: "/research/optimus-map", label: "Optimus map", icon: Network },
  { to: "/harness-graph", label: "Harness graph", icon: Share2 },
  { to: "/rsi-loops", label: "RSI Loops", icon: Repeat },
];

const recentChats = [
  "Designing a reasoning reward",
  "Compare Qwen and Llama",
  "Agentic coding curriculum",
  "Review last training run",
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  const newChat = () => {
    navigate(`/?new=${Date.now()}`);
    setOpen(false);
  };

  return (
    <>
      <button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle menu">
        {open ? <PanelLeftClose /> : <PanelLeftOpen />}
      </button>
      {open && <button className="sidebar-scrim" onClick={() => setOpen(false)} aria-label="Close menu" />}
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="sidebar-top">
          <button onClick={newChat} className="new-chat-btn">
            <MessageSquarePlus /> <span>New chat</span><kbd>⌘ K</kbd>
          </button>
          <nav className="main-nav" aria-label="Workspace">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} onClick={() => setOpen(false)} className={({ isActive }) => `side-link ${isActive ? "active" : ""}`}>
                <Icon /><span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="sidebar-lower">
          <div className="recents-label">Recent chats</div>
          <div className="recent-list">
            {recentChats.map((chat, index) => (
              <NavLink key={chat} to={`/?chat=${index}`} onClick={() => setOpen(false)} className="recent-chat">
                <span>{chat}</span>
              </NavLink>
            ))}
          </div>
          <div className="account-row">
            <div className="avatar">IL</div>
            <div className="account-copy"><strong>Local workspace</strong><span>Optimus account</span></div>
            <ThemeToggle />
            <button className="icon-button" aria-label="Settings"><Settings /></button>
          </div>
        </div>
      </aside>
    </>
  );
}
