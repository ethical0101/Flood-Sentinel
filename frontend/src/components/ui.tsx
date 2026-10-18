// Design-system primitives (see docs/ui_redesign_plan.md). Dense, bordered, low-shadow.
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { AlertTriangle, ArrowDown, ArrowUp, CheckCircle2, ChevronsUpDown, Download, Info, Loader2, RefreshCw, Table2, X, XCircle } from "lucide-react";
import { toCsv, downloadText } from "../lib";

// ------------------------------------------------------------------ layout
export function Panel({ title, info, actions, children, className = "", body = "p-3", footer, id }: {
  title?: ReactNode; info?: string; actions?: ReactNode; children: ReactNode; className?: string; body?: string; footer?: ReactNode; id?: string;
}) {
  return (
    <section id={id} className={`min-w-0 rounded-lg border border-line bg-panel animate-fade ${className}`}>
      {(title || actions) && (
        <header className="flex min-h-10 flex-wrap items-center justify-between gap-x-3 gap-y-1.5 border-b border-line-2 px-3 py-2">
          <h2 className="flex min-w-0 items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-2">
            <span className="truncate">{title}</span>
            {info && <InfoTip text={info} />}
          </h2>
          {actions && <div className="flex min-w-0 max-w-full flex-wrap items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={body}>{children}</div>
      {footer && <footer className="border-t border-line-2 px-3 py-2 text-[11px] text-ink-3">{footer}</footer>}
    </section>
  );
}

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0 max-w-3xl">
        <h1 className="text-[20px] font-semibold tracking-tight text-ink">{title}</h1>
        {sub && <p className="mt-0.5 text-[13px] leading-relaxed text-ink-2">{sub}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

// ------------------------------------------------------------------ tooltip
export function InfoTip({ text, label = "More information" }: { text: string; label?: string }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex normal-case tracking-normal">
      <button type="button" aria-label={label} aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full text-ink-3 hover:text-accent">
        <Info className="h-3.5 w-3.5" aria-hidden />
      </button>
      {open && (
        <span id={id} role="tooltip"
          className="absolute left-1/2 top-5 z-[2000] w-64 -translate-x-1/2 rounded-md bg-nav px-2.5 py-2 text-[11.5px] font-normal leading-snug text-slate-100 shadow-lg">
          {text}
        </span>
      )}
    </span>
  );
}

// ------------------------------------------------------------------ metrics
export function Metric({ label, value, unit, hint, tone = "ink", info, source }: {
  label: string; value: ReactNode; unit?: string; hint?: ReactNode; tone?: "ink" | "water" | "ok" | "warn" | "bad" | "truth"; info?: string; source?: string;
}) {
  const c = { ink: "text-ink", water: "text-water", ok: "text-ok", warn: "text-warn", bad: "text-bad", truth: "text-truth" }[tone];
  return (
    <div className="min-w-0 rounded-lg border border-line bg-panel px-3 py-2.5 animate-fade" title={source ? `Source: ${source}` : undefined}>
      <div className="flex items-center gap-1 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-3">
        <span className="truncate">{label}</span>{info && <InfoTip text={info} />}
      </div>
      <div className={`mt-1 truncate text-[22px] font-semibold leading-tight tnum ${c}`}>
        {value}{unit && <span className="ml-1 text-[12px] font-medium text-ink-3">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 truncate text-[11.5px] text-ink-2">{hint}</div>}
    </div>
  );
}

const BADGE = {
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
  ok: "bg-ok-soft text-ok ring-green-200",
  warn: "bg-warn-soft text-warn ring-amber-200",
  bad: "bg-bad-soft text-bad ring-red-200",
  water: "bg-water-soft text-water ring-blue-200",
  truth: "bg-truth-soft text-truth ring-violet-200",
  accent: "bg-accent-soft text-accent ring-cyan-200",
  dark: "bg-nav text-slate-100 ring-nav",
};
export function Badge({ children, tone = "slate", title }: { children: ReactNode; tone?: keyof typeof BADGE; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-[1px] text-[10.5px] font-semibold ring-1 ring-inset ${BADGE[tone]}`}>
      {children}
    </span>
  );
}

export function StatusDot({ state }: { state: "ok" | "warn" | "bad" | "idle" | "busy" }) {
  const c = { ok: "bg-emerald-500", warn: "bg-amber-500", bad: "bg-red-500", idle: "bg-slate-400", busy: "bg-sky-500 animate-pulse" }[state];
  return <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${c}`} aria-hidden />;
}

export function RunState({ state }: { state?: string }) {
  const tone = ({ completed: "ok", running: "water", paused: "warn", stopped: "slate", interrupted: "warn", stalled: "warn", failed: "bad" } as const)[
    (state ?? "") as "completed"] ?? "slate";
  return <Badge tone={tone}>{state === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-water" />}{state ?? "unknown"}</Badge>;
}

// ------------------------------------------------------------------ states
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-2" role="status">
      <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />{label ?? "Loading…"}
    </span>
  );
}

export function ErrorState({ error, onRetry, compact }: { error: Error; onRetry?: () => void; compact?: boolean }) {
  return (
    <div role="alert" className={`flex items-start gap-2.5 rounded-md border border-red-200 bg-bad-soft text-[12.5px] text-bad ${compact ? "p-2" : "p-3"}`}>
      <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1 break-words">{error.message}</div>
      {onRetry && <Button size="xs" variant="secondary" onClick={onRetry}><RefreshCw className="h-3 w-3" />Retry</Button>}
    </div>
  );
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-10 text-center">
      {icon && <div className="mb-2 text-ink-3">{icon}</div>}
      <div className="text-[13px] font-semibold text-ink">{title}</div>
      {children && <div className="mt-1 max-w-md text-[12px] leading-relaxed text-ink-2">{children}</div>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function Notice({ tone = "info", title, children }: { tone?: "info" | "warn" | "bad" | "ok" | "truth"; title?: string; children: ReactNode }) {
  const cls = {
    info: "border-cyan-200 bg-accent-soft text-ink", warn: "border-amber-200 bg-warn-soft text-amber-950",
    bad: "border-red-200 bg-bad-soft text-red-950", ok: "border-green-200 bg-ok-soft text-green-950",
    truth: "border-violet-200 bg-truth-soft text-violet-950",
  }[tone];
  const Icon = tone === "ok" ? CheckCircle2 : tone === "info" || tone === "truth" ? Info : AlertTriangle;
  return (
    <div className={`flex gap-2.5 rounded-md border px-3 py-2.5 text-[12.5px] leading-relaxed ${cls}`} role={tone === "bad" ? "alert" : undefined}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0 opacity-80" aria-hidden />
      <div className="min-w-0">{title && <div className="font-semibold">{title}</div>}{children}</div>
    </div>
  );
}

// ------------------------------------------------------------------ controls
export function Button({ children, onClick, disabled, variant = "primary", size = "sm", type = "button", className = "", title, href, download }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; variant?: "primary" | "secondary" | "ghost" | "danger" | "dark";
  size?: "xs" | "sm" | "md"; type?: "button" | "submit"; className?: string; title?: string; href?: string; download?: boolean;
}) {
  const v = {
    primary: "bg-accent text-white hover:bg-cyan-800 disabled:bg-slate-300 disabled:text-white",
    secondary: "bg-white text-ink ring-1 ring-inset ring-line hover:bg-slate-50 disabled:text-ink-3",
    ghost: "text-ink-2 hover:bg-slate-100 hover:text-ink",
    danger: "bg-bad text-white hover:bg-red-800 disabled:bg-slate-300",
    dark: "bg-nav text-white hover:bg-nav-3",
  }[variant];
  const s = { xs: "h-6 px-2 text-[11px] gap-1", sm: "h-8 px-3 text-[12.5px] gap-1.5", md: "h-9 px-4 text-[13px] gap-2" }[size];
  const cls = `inline-flex items-center justify-center rounded-md font-medium transition-colors disabled:cursor-not-allowed ${v} ${s} ${className}`;
  if (href) return <a href={href} className={cls} title={title} download={download}>{children}</a>;
  return <button type={type} onClick={onClick} disabled={disabled} className={cls} title={title}>{children}</button>;
}

export function Segmented<T extends string>({ value, onChange, options, label, size = "sm" }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[]; label: string; size?: "xs" | "sm";
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap gap-0.5 rounded-md bg-slate-100 p-0.5">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} disabled={o.disabled} aria-description={o.title} data-tip={o.title}
          onClick={() => onChange(o.value)}
          className={`inline-flex items-center gap-1 rounded px-2.5 font-medium transition-colors disabled:opacity-40 ${size === "xs" ? "h-6 text-[11px]" : "h-7 text-[12px]"} ${
            value === o.value ? "bg-white text-ink shadow-[0_1px_2px_rgba(15,23,42,0.08)]" : "text-ink-2 hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, hint, error, children, info }: { label: string; hint?: ReactNode; error?: string; children: ReactNode; info?: string }) {
  return (
    <div className="block">
      <div className="mb-1 flex items-center gap-1 text-[12px] font-medium text-ink">{label}{info && <InfoTip text={info} />}</div>
      {children}
      {error ? <div className="mt-1 text-[11px] text-bad">{error}</div> : hint ? <div className="mt-1 text-[11px] leading-snug text-ink-3">{hint}</div> : null}
    </div>
  );
}

export function Select({ value, onChange, children, label, disabled }: { value: string; onChange: (v: string) => void; children: ReactNode; label: string; disabled?: boolean }) {
  return (
    <select aria-label={label} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}
      className="h-8 w-full rounded-md border border-line bg-white px-2 text-[12.5px] text-ink focus:border-accent-2 focus:outline-none disabled:bg-slate-50">
      {children}
    </select>
  );
}

export function TextInput({ value, onChange, label, placeholder, type = "text" }: { value: string; onChange: (v: string) => void; label: string; placeholder?: string; type?: string }) {
  return (
    <input aria-label={label} type={type} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)}
      className="h-8 w-full rounded-md border border-line bg-white px-2 text-[12.5px] text-ink placeholder:text-ink-3 focus:border-accent-2 focus:outline-none" />
  );
}

export function Slider({ label, value, min, max, step, onChange, format, hint, info }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string; hint?: string; info?: string;
}) {
  return (
    <label className="block">
      <span className="flex items-center justify-between gap-2 text-[12px] font-medium text-ink">
        <span className="flex items-center gap-1">{label}{info && <InfoTip text={info} />}</span>
        <span className="font-mono text-[11px] text-ink-2 tnum">{format ? format(value) : value}</span>
      </span>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-[var(--color-accent)]" />
      {hint && <span className="block text-[11px] leading-snug text-ink-3">{hint}</span>}
    </label>
  );
}

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span><span className="block text-[12px] font-medium text-ink">{label}</span>{hint && <span className="block text-[11px] leading-snug text-ink-3">{hint}</span>}</span>
      <button type="button" role="switch" aria-checked={checked} aria-label={label} onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-[18px] w-8 shrink-0 rounded-full transition-colors ${checked ? "bg-accent" : "bg-slate-300"}`}>
        <span className={`absolute top-[2px] h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-[15px]" : "translate-x-[2px]"}`} />
      </button>
    </div>
  );
}

export function KV({ items, dense }: { items: [ReactNode, ReactNode][]; dense?: boolean }) {
  return (
    <dl className="divide-y divide-line-2">
      {items.map(([k, v], i) => (
        <div key={i} className={`flex justify-between gap-4 ${dense ? "py-1" : "py-1.5"}`}>
          <dt className="shrink-0 text-ink-2">{k}</dt>
          <dd className="min-w-0 break-words text-right font-medium text-ink tnum">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Progress({ value, tone = "accent" }: { value: number; tone?: "accent" | "water" | "warn" | "bad" }) {
  const c = { accent: "bg-accent-2", water: "bg-water", warn: "bg-amber-500", bad: "bg-red-500" }[tone];
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={`h-full rounded-full transition-all ${c}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

// ------------------------------------------------------------------ drawer
export function Drawer({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[3000] flex justify-end bg-slate-900/30" onClick={onClose}>
      <aside role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}
        className={`flex h-full w-full flex-col bg-panel shadow-2xl animate-slide ${wide ? "max-w-3xl" : "max-w-xl"}`}>
        <header className="flex h-12 items-center justify-between border-b border-line px-4">
          <h2 className="text-[13px] font-semibold text-ink">{title}</h2>
          <button aria-label="Close" onClick={onClose} className="rounded p-1 text-ink-2 hover:bg-slate-100"><X className="h-4 w-4" /></button>
        </header>
        <div className="min-h-0 flex-1 overflow-auto p-4 scroll-thin">{children}</div>
      </aside>
    </div>
  );
}

// ------------------------------------------------------------------ data table
export type Column<T> = { key: string; label: string; value: (r: T) => string | number | null | undefined; render?: (r: T) => ReactNode; numeric?: boolean; info?: string };

export function DataTable<T>({ rows, columns, csvName, empty = "No rows.", rowKey, onRowClick, activeKey, maxHeight }: {
  rows: T[]; columns: Column<T>[]; csvName?: string; empty?: string; rowKey: (r: T) => string; onRowClick?: (r: T) => void; activeKey?: string; maxHeight?: number;
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    return [...rows].sort((a, b) => {
      const x = col.value(a), y = col.value(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
    });
  }, [rows, sort, columns]);
  return (
    <div>
      {csvName && (
        <div className="flex justify-end px-3 pt-2">
          <Button size="xs" variant="ghost" onClick={() => downloadText(`${csvName}.csv`, toCsv(columns.map((c) => c.label), sorted.map((r) => columns.map((c) => c.value(r)))), "text/csv")}>
            <Download className="h-3 w-3" />CSV
          </Button>
        </div>
      )}
      <div className="overflow-auto scroll-thin" style={maxHeight ? { maxHeight } : undefined}>
        <table className="w-full text-[12px]">
          <thead className="sticky top-0 z-10 bg-slate-50 text-[10.5px] uppercase tracking-[0.06em] text-ink-3">
            <tr>
              {columns.map((c) => (
                <th key={c.key} scope="col" className={`whitespace-nowrap px-3 py-2 font-semibold ${c.numeric ? "text-right" : "text-left"}`}
                  aria-sort={sort?.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
                  <button className={`inline-flex items-center gap-1 hover:text-ink ${c.numeric ? "flex-row-reverse" : ""}`}
                    onClick={() => setSort((s) => (s?.key === c.key ? (s.dir === 1 ? { key: c.key, dir: -1 } : null) : { key: c.key, dir: 1 }))}>
                    {c.label}
                    {sort?.key === c.key ? (sort.dir === 1 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />) : <ChevronsUpDown className="h-3 w-3 opacity-30" />}
                  </button>
                  {c.info && <span className="ml-1 align-middle"><InfoTip text={c.info} /></span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-line-2">
            {sorted.length === 0 && <tr><td colSpan={columns.length} className="px-3 py-6 text-center text-ink-3">{empty}</td></tr>}
            {sorted.map((r) => (
              <tr key={rowKey(r)} onClick={onRowClick ? () => onRowClick(r) : undefined}
                className={`${onRowClick ? "cursor-pointer hover:bg-slate-50" : ""} ${activeKey === rowKey(r) ? "bg-accent-soft/70" : ""}`}>
                {columns.map((c) => (
                  <td key={c.key} className={`whitespace-nowrap px-3 py-1.5 ${c.numeric ? "text-right tnum" : ""}`}>
                    {c.render ? c.render(r) : fmtCell(c.value(r))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmtCell(v: string | number | null | undefined) {
  if (v == null || (typeof v === "number" && Number.isNaN(v))) return <span className="text-ink-3" title="unavailable">–</span>;
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4);
  return v;
}

// ------------------------------------------------------------------ chart frame with "view data"
export function ChartFrame({ title, info, source, rows, columns, children, height = 240, actions }: {
  title: string; info?: string; source: string; rows: Record<string, unknown>[]; columns: string[]; children: ReactNode; height?: number; actions?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <Panel title={title} info={info} body="p-2"
      actions={<>{actions}<Button size="xs" variant="ghost" onClick={() => setOpen(true)} title="Show the numbers behind this chart"><Table2 className="h-3 w-3" />Data</Button></>}
      footer={<>Source: {source}</>}>
      <div style={{ height }}>{rows.length ? children : <Empty title="No data available" />}</div>
      <Drawer open={open} onClose={() => setOpen(false)} title={`${title} · data`} wide>
        <p className="mb-2 text-[12px] text-ink-2">Source: {source}</p>
        <DataTable rows={rows} rowKey={(r) => JSON.stringify(r)} csvName={title.toLowerCase().replace(/[^a-z0-9]+/g, "_")}
          columns={columns.map((c) => ({ key: c, label: c, value: (r: Record<string, unknown>) => r[c] as number | string, numeric: typeof rows[0]?.[c] === "number" }))} />
      </Drawer>
    </Panel>
  );
}
