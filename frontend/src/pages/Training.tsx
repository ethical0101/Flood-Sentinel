import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, Copy, Database, Terminal, XCircle } from "lucide-react";
import { api } from "../api";
import { Badge, Callout, Card, EmptyState, ErrorState, KV, PageHeader, Skeleton, StateBadge } from "../components/ui";
import { fmtBytes, fmtDuration, fmtInt, fmtPct, useApi } from "../lib";

export default function Training() {
  const data = useApi(() => api.datasets(), []);
  const training = useApi(() => api.training(), [], 5000);
  const [run, setRun] = useState("");
  const runs = training.data?.runs ?? [];
  useEffect(() => {
    if (!runs.length) return;
    if (!run || !runs.some((r) => r.name === run)) setRun((runs.find((r) => r.state === "running") ?? runs[runs.length - 1]).name);
  }, [runs, run]);
  const log = useApi(() => (run ? api.trainingLog(run, 120) : Promise.resolve({ run: "", lines: [] })), [run], 5000);
  const r = runs.find((x) => x.name === run);
  const gpu = training.data?.gpu;
  const [copied, setCopied] = useState(false);

  return (
    <div>
      <PageHeader eyebrow="Data & training" title="Dataset and training monitor">
        Live status of the training runs on the local GPU, plus provenance and integrity checks for the downloaded data.
        This page reads only from disk; it never starts or stops training.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title={<span className="flex items-center gap-1.5"><Database className="h-4 w-4 text-water-600" />Sen1Floods11 (hand-labelled)</span>}
          subtitle="Downloaded from the public gs://sen1floods11 bucket" className="xl:col-span-2">
          {data.error ? <ErrorState error={data.error} onRetry={data.reload} /> : !data.data ? <Skeleton className="h-40" /> : (
            <div className="grid gap-6 md:grid-cols-2">
              <div>
                <KV items={[
                  ["Downloaded", data.data.downloaded_utc ? new Date(data.data.downloaded_utc).toLocaleString() : "no"],
                  ["Files", `${fmtInt(data.data.n_files)} (${fmtBytes(data.data.download_bytes)})`],
                  ["Integrity", data.data.verification ? <span className="inline-flex items-center gap-1">{data.data.verification.passed
                    ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" /> : <XCircle className="h-3.5 w-3.5 text-red-600" />}
                    {data.data.verification.files_ok}/{data.data.verification.files_checked} {data.data.verification.integrity_mode}</span> : "not verified"],
                  ["CRS", data.data.verification ? Object.keys(data.data.verification.crs).join(", ") : "–"],
                  ["Raw on disk", fmtBytes(data.data.raw_bytes_on_disk)],
                  ["Processed (npz)", fmtBytes(data.data.processed_bytes_on_disk)],
                ]} />
                <div className="mt-3 flex flex-wrap gap-1">{data.data.layers.map((l) => <Badge key={l}>{l}</Badge>)}</div>
              </div>
              <div>
                <table className="w-full text-[12.5px]">
                  <thead className="text-left text-[10.5px] uppercase tracking-wider text-slate-500">
                    <tr><th className="py-1 font-medium">Split</th><th className="py-1 text-right font-medium">Chips</th><th className="py-1 text-right font-medium">Mean water</th><th className="py-1 pl-3 font-medium">Events</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {["train", "valid", "test", "bolivia"].filter((s) => data.data!.splits[s]).map((s) => (
                      <tr key={s}>
                        <td className="py-1.5 font-medium">{s}</td>
                        <td className="py-1.5 text-right tabular-nums">{data.data!.splits[s].chips}</td>
                        <td className="py-1.5 text-right tabular-nums">{fmtPct(data.data!.splits[s].mean_water_fraction)}</td>
                        <td className="py-1.5 pl-3 text-[11px] text-slate-500">{data.data!.splits[s].events.length} events</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {data.data.verification && (
                  <div className="mt-3 text-[11px] text-slate-500">Label pixels: {Object.entries(data.data.verification.label_pixel_counts)
                    .map(([k, v]) => `${{ "-1": "no data", "0": "not water", "1": "water" }[k] ?? k} ${(v / 1e6).toFixed(1)}M`).join(" · ")}</div>
                )}
                <p className="mt-3 text-[11px] leading-relaxed text-slate-500">{data.data.citation}</p>
              </div>
            </div>
          )}
        </Card>

        <Card title="GPU (live)" subtitle="NVML readings, refreshed every 5 s">
          {!training.data ? <Skeleton className="h-32" /> : gpu ? (
            <>
              <KV items={[["Device", gpu.name], ["Driver", gpu.driver], ["Utilisation", `${gpu.utilization_pct}%`],
                ["VRAM", `${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB`], ["Temperature", `${gpu.temperature_c} °C`],
                ["Power", gpu.power_w != null ? `${gpu.power_w} W` : "–"]]} />
              <div className="mt-3 h-2 overflow-hidden rounded bg-slate-100"><div className="h-full bg-water-600 transition-all" style={{ width: `${(100 * gpu.memory_used_mib) / gpu.memory_total_mib}%` }} /></div>
            </>
          ) : <EmptyState title="GPU telemetry unavailable">NVML could not be queried (no NVIDIA driver?).</EmptyState>}
        </Card>
      </div>

      <Card title="Training runs" className="mt-6" bodyClass="p-0">
        {training.error ? <div className="p-5"><ErrorState error={training.error} onRetry={training.reload} /></div> : runs.length === 0 ? (
          <div className="p-5"><EmptyState title="No training runs found">Start one with <code>python -m training.train</code>.</EmptyState></div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12.5px]">
              <thead className="bg-slate-50 text-left text-[10.5px] uppercase tracking-wider text-slate-500">
                <tr><th className="px-4 py-2 font-medium">Run</th><th className="px-2 py-2 font-medium">State</th><th className="px-2 py-2 font-medium">Progress</th>
                  <th className="px-2 py-2 text-right font-medium">Best val IoU</th><th className="px-2 py-2 text-right font-medium">Best epoch</th>
                  <th className="px-2 py-2 text-right font-medium">Duration</th><th className="px-4 py-2 text-right font-medium">Checkpoints</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {runs.map((x) => (
                  <tr key={x.name} onClick={() => setRun(x.name)} className={`cursor-pointer hover:bg-slate-50 ${x.name === run ? "bg-water-50/60" : ""}`}>
                    <td className="px-4 py-2 font-mono text-[12px]">{x.name}</td>
                    <td className="px-2 py-2"><StateBadge state={x.state} /></td>
                    <td className="px-2 py-2">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-28 overflow-hidden rounded bg-slate-100"><div className="h-full bg-water-600" style={{ width: `${(100 * x.epoch) / x.epochs}%` }} /></div>
                        <span className="tabular-nums text-slate-600">{x.epoch}/{x.epochs}</span>
                        {x.state === "running" && x.eta_s != null && <span className="text-[11px] text-slate-500">ETA {fmtDuration(x.eta_s)}</span>}
                        {x.state === "completed" && x.epoch < x.epochs && <span className="text-[11px] text-slate-500">early stop</span>}
                      </div>
                    </td>
                    <td className="px-2 py-2 text-right font-semibold tabular-nums">{x.best != null && x.best >= 0 ? x.best.toFixed(4) : "–"}</td>
                    <td className="px-2 py-2 text-right tabular-nums">{x.best_epoch ?? "–"}</td>
                    <td className="px-2 py-2 text-right tabular-nums">{fmtDuration(x.total_time_s ?? x.history.reduce((s, h) => s + h.epoch_time_s, 0))}</td>
                    <td className="px-4 py-2 text-right text-[11px] text-slate-600">{x.checkpoints.map((c) => `${c.file} (${c.mb} MB)`).join(", ") || "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {r && (
        <div className="mt-6 grid gap-6 xl:grid-cols-2">
          <Card title={`Live curves · ${r.name}`} subtitle="Validation IoU and training loss per epoch">
            {r.history.length === 0 ? <EmptyState title="Waiting for the first epoch…" /> : (
              <div className="grid gap-4 md:grid-cols-2">
                <div className="h-48"><ResponsiveContainer>
                  <LineChart data={r.history} margin={{ top: 5, right: 8, bottom: 0, left: -18 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis dataKey="epoch" tick={{ fontSize: 10 }} /><YAxis domain={[0.4, 1]} tick={{ fontSize: 10 }} />
                    <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                    <Line dataKey="val_iou_water" name="val IoU" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} />
                  </LineChart></ResponsiveContainer></div>
                <div className="h-48"><ResponsiveContainer>
                  <AreaChart data={r.history} margin={{ top: 5, right: 8, bottom: 0, left: -18 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis dataKey="epoch" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} />
                    <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                    <Area dataKey="train_loss" name="train loss" stroke="#64748b" fill="#cbd5e1" isAnimationActive={false} />
                  </AreaChart></ResponsiveContainer></div>
              </div>
            )}
            {r.history.length > 0 && (
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-[11px] text-slate-500">
                <div><div className="text-sm font-semibold text-slate-800">{r.history[r.history.length - 1].epoch_time_s}s</div>last epoch</div>
                <div><div className="text-sm font-semibold text-slate-800">{r.history[r.history.length - 1].peak_gpu_mem_mib} MiB</div>peak GPU memory</div>
                <div><div className="text-sm font-semibold text-slate-800">{r.history[r.history.length - 1].lr.toExponential(2)}</div>learning rate</div>
              </div>
            )}
          </Card>
          <Card title={<span className="flex items-center gap-1.5"><Terminal className="h-4 w-4" />train.log</span>} subtitle="Last 120 lines" bodyClass="p-0">
            {log.error ? <div className="p-4"><ErrorState error={log.error} /></div> : (
              <pre className="scroll-thin h-64 overflow-auto whitespace-pre bg-ink-950 p-3 font-mono text-[10.5px] leading-relaxed text-slate-300">
                {(log.data?.lines ?? []).join("\n") || "(empty)"}
              </pre>
            )}
          </Card>
        </div>
      )}

      {r && (
        <div className="mt-6">
          <Callout title="Resume or continue training">
            Training runs in a separate process, not inside the API. If a run is interrupted, resume it from its last checkpoint with:
            <div className="mt-2 flex items-center gap-2">
              <code className="rounded bg-ink-900 px-2 py-1 font-mono text-[11.5px] text-slate-100">{r.resume_command}</code>
              <button onClick={() => { navigator.clipboard?.writeText(r.resume_command); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
                className="inline-flex items-center gap-1 rounded-md bg-white px-2 py-1 text-xs ring-1 ring-slate-300 hover:bg-slate-50"><Copy className="h-3 w-3" />{copied ? "Copied" : "Copy"}</button>
            </div>
            <div className="mt-1.5 text-xs">The full suite can be rerun with <code>.venv\Scripts\python scripts\run_experiments.py</code> (it skips completed runs and resumes interrupted ones).</div>
          </Callout>
        </div>
      )}
    </div>
  );
}
