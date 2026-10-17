import { NavLink, Outlet } from "react-router-dom";
import { BarChart3, BookOpen, Cpu, Database, LayoutDashboard, Map, ScanLine, ShieldAlert } from "lucide-react";
import { api } from "../api";
import { useApi } from "../lib";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/segmentation", label: "Segmentation Studio", icon: ScanLine },
  { to: "/routes", label: "Safe-Route Planner", icon: Map },
  { to: "/evaluation", label: "Model Evaluation", icon: BarChart3 },
  { to: "/training", label: "Data & Training", icon: Database },
  { to: "/about", label: "Documentation", icon: BookOpen },
];

function BackendPill() {
  const health = useApi(() => api.health(), [], 10000);
  const sys = useApi(() => api.system(), [], 15000);
  const ok = !!health.data && !health.error;
  const gpu = sys.data?.gpu;
  return (
    <div className="space-y-2 rounded-lg bg-white/5 p-3 text-[11px] text-slate-300 ring-1 ring-white/10">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_6px] shadow-emerald-400" : "bg-red-400"}`} />
        <span className="font-medium text-slate-100">{ok ? "API online" : health.loading ? "Connecting…" : "API offline"}</span>
      </div>
      {gpu ? (
        <div className="flex items-center gap-2">
          <Cpu className="h-3.5 w-3.5 text-slate-400" />
          <span className="truncate" title={gpu.name}>{gpu.name.replace("NVIDIA GeForce ", "")}</span>
          <span className="ml-auto tabular-nums text-slate-400">{gpu.utilization_pct}%</span>
        </div>
      ) : sys.data ? (
        <div className="text-slate-400">No NVIDIA GPU detected</div>
      ) : null}
      {gpu && (
        <div className="h-1 overflow-hidden rounded bg-white/10" title={`VRAM ${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB`}>
          <div className="h-full bg-sky-400" style={{ width: `${(100 * gpu.memory_used_mib) / gpu.memory_total_mib}%` }} />
        </div>
      )}
    </div>
  );
}

export default function Layout() {
  return (
    <div className="flex h-full min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col bg-ink-900 px-4 py-5 text-slate-300 lg:flex">
        <div className="flex items-center gap-2.5 px-2">
          <img src="/favicon.svg" alt="" className="h-8 w-8" />
          <div>
            <div className="text-[15px] font-semibold tracking-tight text-white">FloodRoute</div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-400">Disaster response console</div>
          </div>
        </div>
        <nav className="mt-7 space-y-0.5" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-colors ${
                  isActive ? "bg-white/10 text-white ring-1 ring-white/10" : "hover:bg-white/5 hover:text-white"}`}>
              <Icon className="h-4 w-4" aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto space-y-3">
          <BackendPill />
          <div className="flex gap-2 px-1 text-[10.5px] leading-snug text-slate-400">
            <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-amber-400" />
            Research prototype. Not for real evacuation or navigation decisions.
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* compact nav for small screens */}
        <div className="sticky top-0 z-[1100] flex gap-1 overflow-x-auto bg-ink-900 px-3 py-2 lg:hidden">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => `flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium ${isActive ? "bg-white/15 text-white" : "text-slate-300"}`}>
              <Icon className="h-3.5 w-3.5" />{label}
            </NavLink>
          ))}
        </div>
        <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <Outlet />
        </main>
        <footer className="border-t border-slate-200 px-8 py-3 text-[11px] text-slate-500">
          Imagery & labels: Sen1Floods11 (Bonafilia et al., 2020) · Roads © OpenStreetMap contributors (ODbL) ·
          Flood extents are model estimates; routes are not verified safe.
        </footer>
      </div>
    </div>
  );
}
