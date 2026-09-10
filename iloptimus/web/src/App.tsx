import { Routes, Route, useLocation } from "react-router-dom";
import { ThemeProvider } from "./components/ThemeProvider";
import Navbar from "./components/Navbar";
import ChatPage from "./pages/ChatPage";
import ModelLibraryPage from "./pages/ModelLibraryPage";
import OptimusLabPage from "./pages/OptimusLabPage";
import EnvironmentBuilderPage from "./pages/EnvironmentBuilderPage";
import MyEnvironmentsPage from "./pages/MyEnvironmentsPage";
import EnvironmentPlayPage from "./pages/EnvironmentPlayPage";
import AppErrorBoundary from "./components/AppErrorBoundary";
import WorkspaceTabs from "./components/WorkspaceTabs";
import RsiPanelPage from "./pages/RsiPanelPage";
import RsiLoopsPage from "./pages/RsiLoopsPage";
import ResearchPaperPage from "./pages/ResearchPaperPage";
import OptimusMindMapPage from "./pages/OptimusMindMapPage";
import HarnessGraphPage from "./pages/HarnessGraphPage";

export default function App() {
  const location = useLocation();
  if (location.pathname === "/research/sakura-island") {
    return <ResearchPaperPage />;
  }
  if (location.pathname === "/research/optimus-map") {
    return <OptimusMindMapPage />;
  }
  return (
    <ThemeProvider>
      <div className="app-shell">
        <Navbar />
        <main className="app-content">
          <WorkspaceTabs />
          <AppErrorBoundary key={location.pathname}><Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/models" element={<ModelLibraryPage />} />
            <Route path="/studio" element={<OptimusLabPage />} />
            <Route path="/pipelines" element={<EnvironmentBuilderPage />} />
            <Route path="/environments" element={<MyEnvironmentsPage />} />
            <Route path="/environments/:environmentId/play" element={<EnvironmentPlayPage />} />
            <Route path="/rsi/:panelId" element={<RsiPanelPage />} />
            <Route path="/rsi-loops" element={<RsiLoopsPage />} />
            <Route path="/harness-graph" element={<HarnessGraphPage />} />
          </Routes></AppErrorBoundary>
        </main>
      </div>
    </ThemeProvider>
  );
}
