import type { ReactNode } from "react";
import { AlertTriangle, Info, Loader2, RefreshCw } from "lucide-react";

export function Card({ title, subtitle, action, children, className = "", bodyClass = "p-5" }: {
  title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; bodyClass?: string;
}) {
  return (
    <section className={`min-w-0 rounded-xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)] animate-fade-in ${className}`}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-3.5">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold tracking-wide text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, tone = "default", icon }: {
  label: string; value: ReactNode; hint?: ReactNode; tone?: "default" | "good" | "warn" | "bad" | "water"; icon?: ReactNode;
}) {
  const toneCls = { default: "text-slate-900", good: "text-emerald-700", warn: "text-amber-700", bad: "text-red-700", water: "text-water-700" }[tone];
  return (
    <div className="min-w-0 rounded-xl border border-slate-200/80 bg-white px-4 py-3.5 animate-fade-in">
      <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-slate-500">
        {icon}
        {label}
      </div>
      <div className={`mt-1.5 text-2xl font-semibold tabular-nums ${toneCls}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

const BADGE = {
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
  green: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  amber: "bg-amber-50 text-amber-800 ring-amber-200",
  red: "bg-red-50 text-red-700 ring-red-200",
  blue: "bg-water-50 text-water-700 ring-water-100",
  violet: "bg-violet-50 text-violet-700 ring-violet-200",
};
export function Badge({ children, color = "slate" }: { children: ReactNode; color?: keyof typeof BADGE }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${BADGE[color]}`}>
      {children}
    </span>
  );
}

export function StateBadge({ state }: { state: string }) {
  const color = ({ completed: "green", running: "blue", interrupted: "amber", stalled: "amber", failed: "red" } as const)[
    state as "completed"] ?? "slate";
  return <Badge color={color}>{state === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-water-600" />}{state}</Badge>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500" role="status">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      {label ?? "Loading…"}
    </div>
  );
}

export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="flex-1">
        <div className="font-medium">Something went wrong</div>
        <div className="mt-0.5 text-red-700">{error.message}</div>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex items-center gap-1 rounded-md bg-white px-2.5 py-1 text-xs font-medium text-red-700 ring-1 ring-red-200 hover:bg-red-100">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50/60 px-6 py-10 text-center">
      {icon && <div className="mb-3 text-slate-400">{icon}</div>}
      <div className="text-sm font-medium text-slate-700">{title}</div>
      {children && <div className="mt-1 max-w-md text-xs leading-relaxed text-slate-500">{children}</div>}
    </div>
  );
}

export function Callout({ tone = "info", title, children }: { tone?: "info" | "warn" | "danger"; title?: string; children: ReactNode }) {
  const cls = {
    info: "border-water-100 bg-water-50/70 text-slate-700",
    warn: "border-amber-200 bg-amber-50 text-amber-900",
    danger: "border-red-200 bg-red-50 text-red-900",
  }[tone];
  const Icon = tone === "info" ? Info : AlertTriangle;
  return (
    <div className={`flex gap-3 rounded-lg border p-3.5 text-[13px] leading-relaxed ${cls}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0 opacity-80" aria-hidden />
      <div>
        {title && <div className="mb-0.5 font-semibold">{title}</div>}
        {children}
      </div>
    </div>
  );
}

export function PageHeader({ eyebrow, title, children, actions }: { eyebrow?: string; title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        {eyebrow && <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-water-600">{eyebrow}</div>}
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {children && <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{children}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3">
      <span>
        <span className="block text-[13px] font-medium text-slate-800">{label}</span>
        {hint && <span className="block text-xs text-slate-500">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? "bg-water-600" : "bg-slate-300"}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-4" : "translate-x-0.5"}`} />
      </button>
    </label>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, hint }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string; hint?: string;
}) {
  return (
    <label className="block">
      <span className="flex items-center justify-between text-[13px] font-medium text-slate-800">
        {label}
        <span className="font-mono text-xs text-slate-600">{format ? format(value) : value}</span>
      </span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1.5 w-full accent-water-600"
      />
      {hint && <span className="block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

export function Select({ label, value, onChange, children, hint }: {
  label: string; value: string; onChange: (v: string) => void; children: ReactNode; hint?: string;
}) {
  return (
    <label className="block">
      <span className="text-[13px] font-medium text-slate-800">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 shadow-sm focus:border-water-500 focus:outline-none"
      >
        {children}
      </select>
      {hint && <span className="mt-0.5 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

export function Button({ children, onClick, disabled, variant = "primary", type = "button", className = "" }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; variant?: "primary" | "secondary" | "ghost"; type?: "button" | "submit"; className?: string;
}) {
  const cls = {
    primary: "bg-water-600 text-white hover:bg-water-700 shadow-sm disabled:bg-slate-300",
    secondary: "bg-white text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50 disabled:text-slate-400",
    ghost: "text-slate-600 hover:bg-slate-100",
  }[variant];
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed ${cls} ${className}`}>
      {children}
    </button>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="divide-y divide-slate-100 text-[13px]">
      {items.map(([k, v], i) => (
        <div key={i} className="flex justify-between gap-4 py-1.5">
          <dt className="shrink-0 text-slate-500">{k}</dt>
          <dd className="min-w-0 break-words text-right font-medium text-slate-800">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
