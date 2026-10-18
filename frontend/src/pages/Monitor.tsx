import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Copy, Cpu, HardDrive, MemoryStick, Play, Square, Terminal } from "lucide-react";
import { api, type TrainingRun } from "../api";
import { Badge, Button, ChartFrame, DataTable, Drawer, Empty, ErrorState, Field, KV, Metric, Notice, PageHead, Panel, Progress, RunState, Select, Skeleton, TextInput } from "../components/ui";
import { fmtDuration, fmtNum, fmtTime, useApi } from "../lib";

type LiveRun = TrainingRun & { step?: number; steps?: number; running_loss?: number; current_epoch?: number; phase?: string };
const AX = { fontSize: 10.5, fill: "#64748b" };

export default function Monitor() {
  const training = useApi(() => api.training(), [], 4000);
  const sys = useApi(() => api.system(), [], 5000);
  const presets = useApi(() => api.presets(), [], 15000);
  const [run, setRun] = useState("");
  const runs = (training.data?.runs ?? []) as LiveRun[];
  const active = runs.find((r) => r.state === "running");
  useEffect(() => {
    if (!runs.length) return;
    if (!run || !runs.some((r) => r.name === run)) setRun((active ?? runs[runs.length - 1]).name);
  }, [runs, run, active]);
  const log = useApi(() => (run ? api.trainingLog(run, 150) : Promise.resolve({ run: "", lines: [] })), [run], 5000);
  const r = runs.find((x) => x.name === run);
  const gpu = sys.data?.gpu;
  const [startOpen, setStartOpen] = useState(false);
  const [startRun, setStartRun] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ tone: "ok" | "bad"; text: string } | null>(null);
  const [copied, setCopied] = useState(false);

  const startable = (presets.data?.presets ?? []).filter((p) => p.state !== "completed");
  const doStart = async () => {
    setBusy(true); setMsg(null);
    try {
      const res = await api.startTraining(startRun, confirm);
      setMsg({ tone: "ok", text: `Started ${startRun} (PID ${res.pid}${res.resume ? ", resuming from last.pt" : ""}).` });
      setStartOpen(false); setConfirm("");
      training.reload(); presets.reload();
    } catch (e) { setMsg({ tone: "bad", text: (e as Error).message }); } finally { setBusy(false); }
  };
  const doStop = async (name: string) => {
    if (!window.confirm(`Request a graceful stop of ${name}? It stops after the current epoch, with the checkpoint saved.`)) return;
    try { const res = await api.stopTraining(name); setMsg({ tone: "ok", text: res.note }); } catch (e) { setMsg({ tone: "bad", text: (e as Error).message }); }
  };

  return (
    <div>
      <PageHead title="Training & system monitor"
        sub="Live state read from each run's status.json, history.json and train.log, with GPU telemetry from NVML. Nothing on this page is simulated; with no job running it shows idle."
        actions={<Button onClick={() => { setStartOpen(true); setStartRun(startable[0]?.run ?? ""); }} disabled={!presets.data}><Play className="h-3.5 w-3.5" />Start / resume training…</Button>} />
      {msg && <div className="mb-3"><Notice tone={msg.tone}>{msg.text}</Notice></div>}

      <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4 xl:grid-cols-6">
        <Metric label="Training state" tone={active ? "water" : "ink"} value={active ? "Running" : runs.some((x) => x.state === "paused") ? "Paused" : "Idle"}
          hint={active ? active.name : "no training process active"} />
        <Metric label="Epoch" value={active ? `${active.current_epoch ?? active.epoch + 1}/${active.epochs}` : "–"} hint={active?.eta_s != null ? `ETA ${fmtDuration(active.eta_s)}` : undefined} />
        <Metric label="Batch" value={active?.step ? `${active.step}/${active.steps}` : "–"} hint={active?.running_loss != null ? `running loss ${active.running_loss.toFixed(4)}` : "updated every 25 steps"} />
        <Metric label="GPU" value={gpu ? `${gpu.utilization_pct}%` : "n/a"} hint={gpu ? `${gpu.memory_used_mib}/${gpu.memory_total_mib} MiB · ${gpu.temperature_c} °C${gpu.power_w != null ? ` · ${gpu.power_w} W` : ""}` : "NVML unavailable"} />
        <Metric label="RAM" tone={sys.data && sys.data.ram.available_gib < 1 ? "warn" : "ink"} value={sys.data ? `${sys.data.ram.available_gib}` : "–"} unit="GiB free" hint={sys.data ? `of ${sys.data.ram.total_gib} GiB · CPU ${sys.data.cpu.utilization_pct}%` : undefined} />
        <Metric label="Disk" value={sys.data ? `${sys.data.disk.free_gb}` : "–"} unit="GB free" hint={sys.data ? `drive ${sys.data.disk.drive}` : undefined} />
      </div>
      {active && active.step != null && active.steps && (
        <Panel className="mt-3" title={`Progress · ${active.name}`}>
          <div className="space-y-2">
            <div><div className="mb-1 flex justify-between text-[11.5px] text-ink-2"><span>Epochs</span><span className="tnum">{active.epoch}/{active.epochs} complete</span></div><Progress value={active.epoch / active.epochs} /></div>
            <div><div className="mb-1 flex justify-between text-[11.5px] text-ink-2"><span>Current epoch batches</span><span className="tnum">{active.step}/{active.steps}</span></div><Progress value={active.step / active.steps} tone="water" /></div>
          </div>
        </Panel>
      )}

      <Panel title="Runs" className="mt-3" body="p-0 overflow-x-auto">
        {training.error ? <div className="p-3"><ErrorState error={training.error} onRetry={training.reload} /></div> : !training.data ? <Skeleton className="m-3 h-32" /> : (
          <DataTable rowKey={(x: LiveRun) => x.name} rows={runs} activeKey={run} onRowClick={(x: LiveRun) => setRun(x.name)} csvName="training_runs"
            columns={[
              { key: "n", label: "Run", value: (x: LiveRun) => x.name, render: (x: LiveRun) => <span className="font-mono text-[11.5px]">{x.name}</span> },
              { key: "s", label: "State", value: (x: LiveRun) => x.state, render: (x: LiveRun) => <span title={x.note}><RunState state={x.state} /></span> },
              { key: "e", label: "Epochs", numeric: true, value: (x: LiveRun) => x.epoch, render: (x: LiveRun) => `${x.epoch}/${x.epochs}` },
              { key: "b", label: "Best val IoU", numeric: true, value: (x: LiveRun) => (x.best != null && x.best >= 0 ? x.best : null) },
              { key: "be", label: "Best epoch", numeric: true, value: (x: LiveRun) => x.best_epoch ?? null },
              { key: "t", label: "Train time", numeric: true, value: (x: LiveRun) => x.total_time_s ?? x.history.reduce((a, h) => a + h.epoch_time_s, 0), render: (x: LiveRun) => fmtDuration(x.total_time_s ?? x.history.reduce((a, h) => a + h.epoch_time_s, 0)) },
              { key: "m", label: "Peak VRAM", numeric: true, value: (x: LiveRun) => Math.max(0, ...x.history.map((h) => h.peak_gpu_mem_mib)) || null },
              { key: "c", label: "Latest checkpoint", value: (x: LiveRun) => x.checkpoints.map((c) => c.modified).sort().pop() ?? "", render: (x: LiveRun) => { const c = [...x.checkpoints].sort((a, b) => a.modified.localeCompare(b.modified)).pop(); return c ? `${c.file} · ${fmtTime(c.modified)}` : "–"; } },
              { key: "x", label: "", value: () => "", render: (x: LiveRun) => x.state === "running" ? <Button size="xs" variant="danger" onClick={() => doStop(x.name)}><Square className="h-3 w-3" />Stop</Button> : null },
            ]} />
        )}
      </Panel>

      {r && (
        <div className="mt-3 grid gap-3 xl:grid-cols-3">
          <ChartFrame title={`Validation IoU · ${r.name}`} source="history.json" rows={r.history.map((h) => ({ epoch: h.epoch, val_iou: h.val_iou_water, val_f1: h.val_f1_water }))} columns={["epoch", "val_iou", "val_f1"]} height={190}>
            <ResponsiveContainer><LineChart data={r.history} margin={{ top: 6, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" /><XAxis dataKey="epoch" tick={AX} /><YAxis domain={[0.4, 1]} tick={AX} /><Tooltip formatter={(v) => Number(v).toFixed(4)} />
              <Line dataKey="val_iou_water" name="val IoU" stroke="#0e7490" dot={false} strokeWidth={2} isAnimationActive={false} /></LineChart></ResponsiveContainer>
          </ChartFrame>
          <ChartFrame title="Loss" source="history.json" rows={r.history.map((h) => ({ epoch: h.epoch, train_loss: h.train_loss, val_loss: h.val_loss }))} columns={["epoch", "train_loss", "val_loss"]} height={190}>
            <ResponsiveContainer><AreaChart data={r.history} margin={{ top: 6, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" /><XAxis dataKey="epoch" tick={AX} /><YAxis tick={AX} /><Tooltip formatter={(v) => Number(v).toFixed(4)} />
              <Area dataKey="train_loss" name="train" stroke="#94a3b8" fill="#e2e8f0" isAnimationActive={false} /><Area dataKey="val_loss" name="val" stroke="#b91c1c" fill="none" isAnimationActive={false} /></AreaChart></ResponsiveContainer>
          </ChartFrame>
          <Panel title="Run status">
            <KV dense items={[["State", <RunState state={r.state} />], ["Started", fmtTime(r.started)], ["Finished", fmtTime(r.finished)],
              ["Best val IoU", r.best != null && r.best >= 0 ? fmtNum(r.best, 4) : "–"], ["Last epoch time", r.history.length ? `${r.history[r.history.length - 1].epoch_time_s} s` : "–"],
              ["Learning rate", r.history.length ? r.history[r.history.length - 1].lr.toExponential(2) : "–"], ["Note", r.note ?? r.error ?? "–"]]} />
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><Terminal className="h-3.5 w-3.5" />train.log</span>} className="xl:col-span-2" body="p-0 overflow-x-auto">
            {log.error ? <div className="p-3"><ErrorState error={log.error} /></div> : (
              <pre className="h-64 overflow-auto whitespace-pre bg-nav p-3 font-mono text-[10.5px] leading-relaxed text-slate-300 scroll-thin scroll-dark">{(log.data?.lines ?? []).join("\n") || "(empty)"}</pre>
            )}
          </Panel>
          <Panel title="Resume & reproduce">
            <p className="text-[12px] text-ink-2">Training runs in its own process, never inside the API. To resume this run from its last checkpoint:</p>
            <div className="mt-2 flex items-center gap-1.5">
              <code className="min-w-0 flex-1 truncate rounded bg-nav px-2 py-1.5 font-mono text-[11px] text-slate-100">{r.resume_command} --polite</code>
              <Button size="xs" variant="secondary" onClick={() => { navigator.clipboard?.writeText(`${r.resume_command} --polite`); setCopied(true); setTimeout(() => setCopied(false), 1400); }}><Copy className="h-3 w-3" />{copied ? "Copied" : "Copy"}</Button>
            </div>
            <p className="mt-2 text-[11.5px] text-ink-3">Full v2 suite: <code>.venv\Scripts\python scripts\run_experiments.py --suite v2 --polite</code>. It skips completed runs, resumes interrupted ones, and waits whenever another project uses the GPU.</p>
          </Panel>
        </div>
      )}

      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <Panel title="GPU sharing & safeguards" info="Training refuses to start while another project uses the GPU, and pauses at epoch boundaries when one appears.">
          {!presets.data ? <Skeleton className="h-28" /> : (
            <ul className="space-y-1 text-[12px]">{presets.data.preflight.checks.map((c) => (
              <li key={c.check} className="flex items-center justify-between gap-2"><span className="flex items-center gap-1.5">{c.ok ? <Badge tone="ok">ok</Badge> : <Badge tone="warn">blocked</Badge>}{c.check}</span><span className="truncate text-right text-ink-2">{c.detail}</span></li>
            ))}</ul>
          )}
        </Panel>
        <Panel title="System">
          {!sys.data ? <Skeleton className="h-28" /> : (
            <KV dense items={[[<span className="flex items-center gap-1"><Cpu className="h-3 w-3" />GPU</span>, gpu ? `${gpu.name} · driver ${gpu.driver}` : "not available"],
              ["PyTorch", `${sys.data.torch.version} · CUDA ${sys.data.torch.cuda_build ?? "–"} · ${sys.data.torch.cuda_available ? "available" : "unavailable"}`],
              [<span className="flex items-center gap-1"><MemoryStick className="h-3 w-3" />RAM</span>, `${sys.data.ram.available_gib} GiB free / ${sys.data.ram.total_gib} GiB (${sys.data.ram.used_pct}% used)`],
              ["CPU", `${sys.data.cpu.physical_cores} cores / ${sys.data.cpu.logical_cores} threads · ${sys.data.cpu.utilization_pct}%`],
              [<span className="flex items-center gap-1"><HardDrive className="h-3 w-3" />Disk {sys.data.disk.drive}</span>, `${sys.data.disk.free_gb} GB free of ${sys.data.disk.total_gb}`], ["OS / Python", `${sys.data.os} · ${sys.data.python}`]]} />
          )}
        </Panel>
      </div>

      <Drawer open={startOpen} onClose={() => setStartOpen(false)} title="Start or resume a training run">
        {!presets.data ? <Skeleton className="h-40" /> : startable.length === 0 ? <Empty title="All presets are completed" /> : (
          <div className="space-y-3">
            <Notice tone="warn">Training uses the GPU for about 15–20 minutes per run. Only predefined experiment presets can be launched, and they run in GPU-courtesy mode.</Notice>
            <Field label="Experiment preset">
              <Select label="Experiment preset" value={startRun} onChange={(v) => { setStartRun(v); setConfirm(""); }}>
                {startable.map((p) => <option key={p.run} value={p.run}>{p.run} ({p.state})</option>)}
              </Select>
            </Field>
            <div className="rounded-md border border-line p-2 font-mono text-[11px] text-ink-2">overrides: {(presets.data.presets.find((p) => p.run === startRun)?.overrides ?? []).join(" ") || "(v1 defaults)"}</div>
            <div>
              <div className="mb-1 text-[12px] font-medium">Preflight checks</div>
              <ul className="space-y-1 text-[12px]">{presets.data.preflight.checks.map((c) => <li key={c.check} className="flex justify-between gap-2"><span>{c.ok ? "✓" : "✗"} {c.check}</span><span className="text-ink-2">{c.detail}</span></li>)}</ul>
            </div>
            <Field label="Type the run name to confirm" error={confirm && confirm !== startRun ? "Does not match the selected run" : undefined}>
              <TextInput label="Confirm run name" value={confirm} onChange={setConfirm} placeholder={startRun} />
            </Field>
            <Button onClick={doStart} disabled={busy || confirm !== startRun || !presets.data.preflight.ok} className="w-full" size="md"><Play className="h-4 w-4" />Start training</Button>
            {!presets.data.preflight.ok && <p className="text-[11.5px] text-warn">Blocked by preflight checks; the server will refuse the request until they pass.</p>}
          </div>
        )}
      </Drawer>
    </div>
  );
}
