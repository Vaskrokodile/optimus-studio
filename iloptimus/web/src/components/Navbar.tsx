import { NavLink, useNavigate } from "react-router-dom";
import { useState } from "react";
import {
  BookOpen,
  Cpu,
  FlaskConical,
  MessageSquarePlus,
  Network,
  Search,
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
  { title: "Designing a reasoning reward", date: "Today" },
  { title: "Compare Qwen and Llama", date: "Yesterday" },
  { title: "Agentic coding curriculum", date: "Sep 10" },
  { title: "Review last training run", date: "Sep 8" },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const newChat = () => {
    navigate(`/?new=${Date.now()}`);
    setOpen(false);
  };

  const filteredChats = recentChats.filter((chat) => chat.title.toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <>
      <button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle menu">
        {open ? <PanelLeftClose /> : <PanelLeftOpen />}
      </button>
      {open && <button className="sidebar-scrim" onClick={() => setOpen(false)} aria-label="Close menu" />}
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="sidebar-top">
          <div className="brand">
            <span className="brand-mark"><i /></span>
            <span><strong>Optimus Studio</strong><small>Local model lab</small></span>
          </div>
          <button onClick={newChat} className="new-chat-btn">
            <MessageSquarePlus /> <span>New chat</span><kbd>⌘ K</kbd>
          </button>
          <label className="chat-search">
            <Search />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search chats…" aria-label="Search chats" />
          </label>
          <div>
            <div className="recents-label">Recent chats</div>
            <div className="recent-list">
              {filteredChats.map((chat) => (
                <NavLink key={chat.title} to={`/?chat=${recentChats.indexOf(chat)}`} onClick={() => setOpen(false)} className="recent-chat">
                  <span>{chat.title}</span>
                  <small>{chat.date}</small>
                </NavLink>
              ))}
            </div>
          </div>
        </div>

        <div className="sidebar-lower">
          <div className="recents-label">Menu</div>
          <nav className="main-nav" aria-label="Workspace">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} onClick={() => setOpen(false)} className={({ isActive }) => `side-link ${isActive ? "active" : ""}`}>
                <Icon /><span>{label}</span>
              </NavLink>
            ))}
          </nav>
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
