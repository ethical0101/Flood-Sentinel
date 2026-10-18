import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { BarChart3, BookOpen, ChevronsLeft, ChevronsRight, Cpu, Database, FlaskConical, History, LayoutGrid, Map, ScanLine, ShieldAlert } from "lucide-react";
import { api } from "../api";
import { useApi } from "../lib";
import { StatusDot } from "./ui";

export const NAV = [
  { to: "/", label: "Command", icon: LayoutGrid, end: true, title: "Command overview" },
  { to: "/segmentation", label: "Segmentation", icon: ScanLine, title: "Segmentation studio" },
  { to: "/routes", label: "Route planner", icon: Map, title: "Flood-risk route planner" },
  { to: "/evaluation", label: "Evaluation lab", icon: BarChart3, title: "Model evaluation laboratory" },
  { to: "/training", label: "Training & system", icon: FlaskConical, title: "Training and system monitor" },
  { to: "/datasets", label: "Datasets", icon: Database, title: "Dataset explorer" },
  { to: "/history", label: "History & reports", icon: History, title: "Analysis history and reports" },
  { to: "/about", label: "Docs & settings", icon: BookOpen, title: "Documentation, diagnostics and settings" },
];

function readCollapsed() {
  try {
    return localStorage.getItem("nav-collapsed") === "1";
  } catch {
    return false;
  }
}

function StatusChips() {
  const health = useApi(() => api.health(), [], 10000);
  const sys = useApi(() => api.system(), [], 10000);
  const models = useApi(() => api.modelStatus(), [], 30000);
  const train = useApi(() => api.training(), [], 10000);
  const gpu = sys.data?.gpu;
  const running = train.data?.runs.find((r) => r.state === "running");
  const paused = train.data?.runs.find((r) => r.state === "paused");
  const apiOk = !!health.data && !health.error;
  const chip = "flex h-7 items-center gap-1.5 rounded-md border border-line bg-white px-2 text-[11.5px] text-ink-2";
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className={chip} title="FastAPI backend health (polled every 10 s)">
        <StatusDot state={apiOk ? "ok" : health.loading ? "idle" : "bad"} />{apiOk ? `API v${health.data!.version}` : health.loading ? "Connecting" : "API offline"}
      </span>
      <span className={`${chip} hidden sm:flex`} title={gpu ? `${gpu.name} · ${gpu.utilization_pct}% utilisation · ${gpu.temperature_c} °C` : "No NVIDIA telemetry"}>
        <Cpu className="h-3.5 w-3.5" />{gpu ? <span className="tnum">{gpu.memory_used_mib}/{gpu.memory_total_mib} MiB · {gpu.utilization_pct}%</span> : "GPU n/a"}
      </span>
      <span className={`${chip} hidden md:flex`} title="Default model served by the API">
        <StatusDot state={models.data?.ready ? "ok" : "warn"} /><span className="max-w-40 truncate font-mono">{models.data?.default_model ?? "–"}</span>
      </span>
      <span className={`${chip} hidden md:flex`} title="Training process state (from status.json)">
        <StatusDot state={running ? "busy" : paused ? "warn" : "idle"} />
        {running ? <span className="tnum">Training {running.name} · {running.epoch}/{running.epochs}</span> : paused ? `Paused: ${paused.name}` : "Training idle"}
      </span>
    </div>
  );
}

export default function AppShell() {
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const loc = useLocation();
  const current = NAV.find((n) => (n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to))) ?? NAV[0];
  useEffect(() => {
    try {
      localStorage.setItem("nav-collapsed", collapsed ? "1" : "0");
    } catch {
      /* storage unavailable: keep in memory */
    }
  }, [collapsed]);
  useEffect(() => {
    document.title = `${current.title} · FloodRoute`;
  }, [current]);

  return (
    <div className="flex min-h-screen">
      <aside className={`sticky top-0 hidden h-screen shrink-0 flex-col bg-nav text-slate-300 transition-[width] duration-150 lg:flex ${collapsed ? "w-[60px]" : "w-[216px]"}`}>
        <div className={`flex h-12 items-center gap-2 border-b border-white/5 ${collapsed ? "justify-center" : "px-4"}`}>
          <img src="/favicon.svg" alt="" className="h-7 w-7" />
          {!collapsed && <div className="leading-tight"><div className="text-[14px] font-semibold text-white">FloodRoute</div>
            <div className="text-[9.5px] uppercase tracking-[0.18em] text-slate-500">Flood intelligence</div></div>}
        </div>
        <nav aria-label="Main" className="flex-1 space-y-0.5 overflow-y-auto px-2 py-3 scroll-thin scroll-dark">
          {NAV.map(({ to, label, icon: Icon, end, title }) => (
            <NavLink key={to} to={to} end={end} title={collapsed ? title : undefined}
              className={({ isActive }) => `group relative flex h-9 items-center gap-2.5 rounded-md text-[12.5px] font-medium transition-colors ${
                collapsed ? "justify-center" : "px-2.5"} ${isActive ? "bg-nav-3 text-white" : "hover:bg-nav-2 hover:text-white"}`}>
              {({ isActive }) => (<>
                {isActive && <span className="absolute left-0 top-2 h-5 w-[3px] rounded-r bg-accent-2" />}
                <Icon className="h-4 w-4 shrink-0" aria-hidden />{!collapsed && <span className="truncate">{label}</span>}
              </>)}
            </NavLink>
          ))}
        </nav>
        <div className="space-y-2 border-t border-white/5 p-2">
          {!collapsed && (
            <div className="flex gap-1.5 rounded-md bg-white/5 p-2 text-[10.5px] leading-snug text-slate-400">
              <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-amber-400" aria-hidden />Research prototype. Not for real evacuation or navigation decisions.
            </div>
          )}
          <button onClick={() => setCollapsed((c) => !c)} aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
            className="flex h-8 w-full items-center justify-center rounded-md text-slate-400 hover:bg-nav-2 hover:text-white">
            {collapsed ? <ChevronsRight className="h-4 w-4" /> : <ChevronsLeft className="h-4 w-4" />}
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="sticky top-0 z-[1100] flex gap-1 overflow-x-auto bg-nav px-2 py-1.5 lg:hidden" aria-label="Main (compact)">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => `flex shrink-0 items-center gap-1.5 rounded px-2 py-1.5 text-[11.5px] font-medium ${isActive ? "bg-nav-3 text-white" : "text-slate-300"}`}>
              <Icon className="h-3.5 w-3.5" />{label}
            </NavLink>
          ))}
        </div>
        <header className="sticky top-0 z-[1050] hidden h-12 items-center justify-between gap-3 border-b border-line bg-white/90 px-5 backdrop-blur lg:flex">
          <div className="flex min-w-0 items-center gap-2 text-[12.5px]">
            <span className="text-ink-3">FloodRoute</span><span className="text-ink-3">/</span>
            <span className="truncate font-semibold text-ink">{current.title}</span>
          </div>
          <StatusChips />
        </header>
        <div className="border-b border-line bg-white px-3 py-2 lg:hidden"><StatusChips /></div>
        <main className="mx-auto w-full max-w-[1680px] flex-1 px-3 py-4 sm:px-5 lg:py-5">
          <Outlet />
        </main>
        <footer className="border-t border-line bg-white px-5 py-2 text-[10.5px] text-ink-3">
          Imagery & labels: Sen1Floods11 (Bonafilia et al., 2020); WorldFloods v2 test split (Portalés-Julià et al., 2023, CC BY-NC 4.0) ·
          Roads © OpenStreetMap contributors (ODbL) · Flood extents are model estimates; routes are not verified safe.
        </footer>
      </div>
    </div>
  );
}
