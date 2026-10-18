import { Fragment, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3 } from "lucide-react";
import { api, runImage, type EvaluationRun, type Metrics } from "../api";
import { Badge, ChartFrame, DataTable, Empty, ErrorState, KV, Notice, PageHead, Panel, Segmented, Select, Skeleton } from "../components/ui";
import { METRIC_INFO, fmtDist, fmtDuration, fmtInt, fmtMs, fmtNum, fmtPct, modelLabel, useApi } from "../lib";

const AX = { fontSize: 10.5, fill: "#64748b" };

function Confusion({ m }: { m: Metrics }) {
  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  const rows = [[tn, fp], [fn, tp]];
  const tot = [tn + fp, fn + tp];
  const names = ["not water", "water"];
  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center text-[12px]">
        <div />{names.map((n) => <div key={n} className="pb-1 text-[10.5px] font-semibold uppercase text-ink-3">pred. {n}</div>)}
        {rows.map((row, i) => (
          <Fragment key={i}>
            <div className="flex items-center pr-2 text-right text-[10.5px] font-semibold uppercase text-ink-3">true {names[i]}</div>
            {row.map((v, j) => {
              const share = v / Math.max(1, tot[i]);
              return (
                <div key={j} className="rounded px-2 py-2.5" title={`${fmtInt(v)} pixels`}
                  style={{ background: i === j ? `rgba(14,116,144,${0.1 + 0.75 * share})` : `rgba(185,28,28,${0.06 + 0.8 * share})`, color: share > 0.55 ? "white" : "#0f172a" }}>
                  <div className="text-[15px] font-semibold tnum">{fmtPct(share)}</div><div className="text-[10px] opacity-80 tnum">{fmtInt(v)} px</div>
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] text-ink-3">Row-normalised; positive class = water; label −1 excluded.</p>
    </div>
  );
}

export default function Evaluation() {
  const metrics = useApi(() => api.metrics(), []);
  const exps = useApi(() => api.experiments(), []);
  const ext = useApi(() => api.external(), []);
  const [thrMode, setThrMode] = useState<"val" | "0.5">("val");
  const [run, setRun] = useState("");
  const [split, setSplit] = useState<"test" | "bolivia">("test");
  const runs = metrics.data?.runs ?? {};
  const names = Object.keys(runs);
  useEffect(() => { if (!run && names.length) setRun(names.includes("unet_resnet34_s1s2") ? "unet_resnet34_s1s2" : names[0]); }, [names, run]);
  const ev: EvaluationRun | undefined = runs[run];
  const pick = (r: EvaluationRun, s: string) => (thrMode === "val" && r.splits_at_val_threshold ? r.splits_at_val_threshold[s] : r.splits[s]);
  const wf = ext.data?.worldfloods;
  const wfFor = (name: string) => {
    if (!wf) return undefined;
    const t = wf.models[name]?.val_selected_threshold;
    const key = thrMode === "val" && t != null ? `${name}@val${t}` : `${name}@0.5`;
    return wf.totals[key]?.clear;
  };
  const expByName = Object.fromEntries((exps.data?.experiments ?? []).map((e) => [e.experiment_id, e]));

  const rows = useMemo(() => {
    const m = Object.values(runs).map((r) => ({
      key: r.run, kind: "model", name: modelLabel({ arch: r.config.model.arch, encoder: r.config.model.encoder, modality: r.modality }),
      thr: thrMode === "val" ? r.val_selected_threshold ?? 0.5 : 0.5, test: pick(r, "test"), bol: pick(r, "bolivia"), wf: wfFor(r.run),
      params: r.meta.n_params as number | null, lat: r.latency.model_only_ms_mean as number | null, vram: expByName[r.run]?.peak_train_gpu_mem_mib ?? null,
      gen: expByName[r.run]?.generation ?? "v1",
    }));
    const b = Object.entries(metrics.data?.baselines ?? {}).map(([k, v]) => ({
      key: k, kind: "baseline", name: k.replace(/_/g, " "), thr: null as number | null, test: v.splits.test, bol: v.splits.bolivia,
      wf: k === "s2_ndwi_gt0" ? wf?.totals.ndwi_gt0?.clear : k === "s2_mndwi_gt0" ? wf?.totals.mndwi_gt0?.clear : undefined,
      params: null as number | null, lat: null as number | null, vram: null as number | null, gen: "baseline",
    }));
    return [...m, ...b];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs, metrics.data, thrMode, wf, exps.data]);
  type Row = (typeof rows)[number];

  if (metrics.error) return <ErrorState error={metrics.error} onRetry={metrics.reload} />;
  if (!metrics.data) return <div className="space-y-3"><Skeleton className="h-8 w-72" /><Skeleton className="h-64" /></div>;
  if (!names.length) return <Empty icon={<BarChart3 className="h-8 w-8" />} title="No evaluated models">Run python -m training.evaluate --all</Empty>;

  const genData = rows.filter((r) => r.test).map((r) => ({ method: r.kind === "model" ? r.key : r.name, test: +r.test.iou_water.toFixed(4), bolivia: +r.bol.iou_water.toFixed(4), worldfloods: r.wf ? +r.wf.iou_water.toFixed(4) : null }));
  const v1ref = expByName["unet_resnet34_s1s2"];
  const seed = expByName["v2_r0b_unet_s1s2_seed7"];
  const noise = v1ref && seed?.["val_iou@val_thr"] != null && v1ref["val_iou@val_thr"] != null ? Math.abs(seed["val_iou@val_thr"]! - v1ref["val_iou@val_thr"]!) : null;
  const sweep = ev?.threshold_sweep_iou ? Object.keys(ev.threshold_sweep_iou.valid).map((t) => ({ threshold: +t, valid: ev.threshold_sweep_iou!.valid[t], test: ev.threshold_sweep_iou!.test[t], bolivia: ev.threshold_sweep_iou!.bolivia[t] })) : [];
  const evAt = (s: string) => (ev ? pick(ev, s) : undefined);
  const perEvent = ev ? Object.entries((thrMode === "val" && ev.per_event_at_val_threshold ? ev.per_event_at_val_threshold : ev.per_event)[split] ?? {})
    .map(([e, m]) => ({ event: e, IoU: +m.iou_water.toFixed(4), F1: +m.f1_water.toFixed(4), recall: +m.recall_water.toFixed(4), water_share: +m.water_fraction_gt.toFixed(4) })) : [];
  const wa = ext.data?.weakness_analysis;
  const rel = wa ? wa.reliability.bolivia.map((b, i) => ({ bin: b.bin, valid: wa.reliability.valid[i].water_rate, test: wa.reliability.test[i].water_rate, bolivia: b.water_rate, ideal: (i + 0.5) / 10 })) : [];
  const wfCols = wf ? Object.keys(wf.totals).filter((k) => k.startsWith("ndwi") || k.startsWith("mndwi") || k.includes("@val")) : [];

  return (
    <div>
      <PageHead title="Evaluation laboratory"
        sub={<>Protocol: {metrics.data.summary?.protocol ?? "held-out evaluation"}. The test split is spatially close to training; Bolivia and WorldFloods measure generalisation.</>}
        actions={<Segmented label="Threshold" value={thrMode} onChange={setThrMode} options={[{ value: "val", label: "Validation-selected threshold" }, { value: "0.5", label: "Fixed 0.5" }]} />} />

      <Panel title="Models and baselines" info="Pooled water-class metrics. WorldFloods: clear-sky valid pixels; only models that need Sentinel-2 alone can be evaluated there. Click a model row to inspect it below." body="p-0 overflow-x-auto">
        <DataTable rows={rows} rowKey={(r: Row) => r.key} csvName="model_comparison" onRowClick={(r: Row) => r.kind === "model" && setRun(r.key)} activeKey={run}
          columns={[
            { key: "name", label: "Method", value: (r: Row) => r.name, render: (r: Row) => <span className="flex items-center gap-1.5"><span className="font-medium">{r.name}</span>
              {r.kind === "baseline" ? <Badge>baseline</Badge> : <Badge tone={r.gen === "v2" ? "accent" : "slate"}>{r.gen}</Badge>}{r.kind === "model" && <span className="font-mono text-[10.5px] text-ink-3">{r.key}</span>}</span> },
            { key: "thr", label: "Thr.", numeric: true, value: (r: Row) => r.thr, render: (r: Row) => (r.thr == null ? "fixed" : r.thr.toFixed(2)), info: METRIC_INFO.threshold },
            { key: "ti", label: "Test IoU", numeric: true, value: (r: Row) => r.test.iou_water, info: METRIC_INFO.test },
            { key: "tf", label: "Test F1", numeric: true, value: (r: Row) => r.test.f1_water, info: METRIC_INFO.f1 },
            { key: "tp", label: "Precision", numeric: true, value: (r: Row) => r.test.precision_water, info: METRIC_INFO.precision },
            { key: "tr", label: "Recall", numeric: true, value: (r: Row) => r.test.recall_water, info: METRIC_INFO.recall },
            { key: "tm", label: "mIoU", numeric: true, value: (r: Row) => r.test.miou, info: METRIC_INFO.miou },
            { key: "bi", label: "Bolivia IoU", numeric: true, value: (r: Row) => r.bol.iou_water, info: METRIC_INFO.bolivia },
            { key: "br", label: "Bolivia recall", numeric: true, value: (r: Row) => r.bol.recall_water },
            { key: "wi", label: "WorldFloods IoU", numeric: true, value: (r: Row) => r.wf?.iou_water ?? null, info: METRIC_INFO.worldfloods },
            { key: "p", label: "Params (M)", numeric: true, value: (r: Row) => (r.params ? +(r.params / 1e6).toFixed(2) : null) },
            { key: "l", label: "Latency (ms)", numeric: true, value: (r: Row) => (r.lat ? +r.lat.toFixed(1) : null), info: METRIC_INFO.latency },
            { key: "v", label: "Train VRAM (MiB)", numeric: true, value: (r: Row) => r.vram },
          ]} />
      </Panel>

      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <ChartFrame title="Generalisation · IoU on three evaluation sets" source="evaluation.json, baselines.json, worldfloods_evaluation.json" info="A missing WorldFloods bar means the method cannot run there (it needs Sentinel-1)."
          rows={genData} columns={["method", "test", "bolivia", "worldfloods"]} height={290}>
          <ResponsiveContainer>
            <BarChart data={genData} margin={{ top: 6, right: 6, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" vertical={false} />
              <XAxis dataKey="method" tick={{ ...AX, fontSize: 9 }} interval={0} angle={-28} textAnchor="end" height={92} />
              <YAxis domain={[0, 1]} tick={AX} width={30} />
              <Tooltip formatter={(v) => (v == null ? "n/a" : Number(v).toFixed(3))} />
              <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" height={22} />
              <Bar dataKey="test" name="Test" fill="#0e7490" radius={[2, 2, 0, 0]} />
              <Bar dataKey="bolivia" name="Bolivia" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              <Bar dataKey="worldfloods" name="WorldFloods" fill="#7c3aed" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
        <Panel title="v1 → v2 experiments" info="Controlled experiments: one change at a time from the v1 recipe. Seed noise = |validation IoU difference| between v1 (seed 42) and the same recipe with seed 7." body="p-0"
          footer={noise != null ? <>Seed noise on validation IoU: <b className="text-ink tnum">±{noise.toFixed(4)}</b>. Smaller differences are not interpretable.</> : <>The seed-noise run (R0b) has not finished yet.</>}>
          {exps.error ? <div className="p-3"><ErrorState error={exps.error} /></div> : !exps.data ? <Skeleton className="m-3 h-40" /> : (
            <DataTable rowKey={(r) => r.experiment_id} rows={exps.data.experiments.filter((e) => e.generation === "v2" || e.experiment_id === "unet_resnet34_s1s2" || e.experiment_id === "unet_resnet34_s2")} csvName="experiments"
              columns={[
                { key: "id", label: "Experiment", value: (r) => r.experiment_id, render: (r) => <span className="font-mono text-[11px]">{r.experiment_id}</span> },
                { key: "ch", label: "Change vs v1", value: (r) => r.change_vs_v1 },
                { key: "st", label: "State", value: (r) => r.state ?? "", render: (r) => <Badge tone={r.state === "completed" ? "ok" : r.state === "running" ? "water" : "warn"}>{r.state ?? "–"}</Badge> },
                { key: "v", label: "Val IoU", numeric: true, value: (r) => r["val_iou@val_thr"] ?? r["best_val_iou@0.5"] },
                { key: "t", label: "Test IoU", numeric: true, value: (r) => r["test_iou@val_thr"] },
                { key: "b", label: "Bolivia IoU", numeric: true, value: (r) => r["bolivia_iou@val_thr"] },
                { key: "d", label: "Δ val vs v1", numeric: true, value: (r) => (r["val_iou@val_thr"] != null && v1ref?.["val_iou@val_thr"] != null ? +(r["val_iou@val_thr"]! - v1ref["val_iou@val_thr"]!).toFixed(4) : null) },
              ]} />
          )}
        </Panel>
      </div>

      {ext.data?.bootstrap && (
        <Panel title="v2 vs v1 · paired chip bootstrap" className="mt-3" body="p-0 overflow-x-auto"
          info="Difference in pooled water IoU against the v1 main model (unet_resnet34_s1s2), each at its validation-selected threshold. 95% interval from 2,000 chip resamples. Training-seed variance is NOT included: compare with the seed-7 row."
          footer={<>{ext.data.bootstrap.method}. The pre-registered rule selects on validation IoU only; Bolivia differences are reported, not used to choose the default model.</>}>
          <table className="w-full text-[12px] tnum">
            <thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Run</th>
              {Object.keys(ext.data.bootstrap.splits).map((s) => <th key={s} className="px-3 py-1.5 text-right">{s} ΔIoU [95% CI]</th>)}</tr></thead>
            <tbody className="divide-y divide-line-2">
              {Object.keys(ext.data.bootstrap.splits.test?.comparisons ?? {}).map((r) => (
                <tr key={r}><td className="px-3 py-1.5 font-mono text-[11px]">{r}</td>
                  {Object.values(ext.data!.bootstrap!.splits).map((sp, i) => {
                    const c = sp.comparisons[r];
                    const sig = c.ci95[0] > 0 ? "text-ok font-semibold" : c.ci95[1] < 0 ? "text-bad font-semibold" : "text-ink-2";
                    return <td key={i} className={`px-3 py-1.5 text-right ${sig}`}>{c.delta_iou >= 0 ? "+" : ""}{c.delta_iou.toFixed(4)} <span className="text-ink-3">[{c.ci95[0].toFixed(3)}, {c.ci95[1].toFixed(3)}]</span></td>;
                  })}</tr>))}
            </tbody>
          </table>
        </Panel>
      )}

      {ev && (<>
        <div className="mb-2 mt-5 flex flex-wrap items-center gap-2">
          <h2 className="text-[15px] font-semibold text-ink">Run detail</h2>
          <div className="w-72"><Select label="Run" value={run} onChange={setRun}>{names.map((n) => <option key={n} value={n}>{n}</option>)}</Select></div>
          <Badge tone="water">{modelLabel({ arch: ev.config.model.arch, encoder: ev.config.model.encoder, modality: ev.modality })}</Badge>
          {ev.val_selected_threshold != null && <Badge tone="accent">val threshold {ev.val_selected_threshold}</Badge>}
        </div>
        <div className="grid gap-3 xl:grid-cols-3">
          <ChartFrame title="Training curves" source="history.json" rows={ev.history.map((h) => ({ epoch: h.epoch, train_loss: h.train_loss, val_loss: h.val_loss, val_iou: h.val_iou_water, val_f1: h.val_f1_water }))}
            columns={["epoch", "train_loss", "val_loss", "val_iou", "val_f1"]} height={220}>
            <ResponsiveContainer>
              <LineChart data={ev.history} margin={{ top: 6, right: 4, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" /><XAxis dataKey="epoch" tick={AX} /><YAxis yAxisId="l" tick={AX} /><YAxis yAxisId="r" orientation="right" domain={[0.4, 1]} tick={AX} />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} /><Legend wrapperStyle={{ fontSize: 11 }} />
                <Line yAxisId="l" dataKey="train_loss" name="train loss" stroke="#94a3b8" dot={false} strokeWidth={1.8} isAnimationActive={false} />
                <Line yAxisId="l" dataKey="val_loss" name="val loss" stroke="#b91c1c" dot={false} strokeWidth={1.8} isAnimationActive={false} />
                <Line yAxisId="r" dataKey="val_iou_water" name="val IoU" stroke="#0e7490" dot={false} strokeWidth={2} isAnimationActive={false} />
                <ReferenceLine yAxisId="r" x={ev.checkpoint_epoch} stroke="#0e7490" strokeDasharray="4 3" />
              </LineChart>
            </ResponsiveContainer>
          </ChartFrame>
          <ChartFrame title="Threshold sweep" source="evaluation.json (threshold_sweep_iou)" info="IoU against the decision threshold. The marker is the threshold chosen on validation; test and Bolivia curves are shown for transparency only."
            rows={sweep} columns={["threshold", "valid", "test", "bolivia"]} height={220}>
            <ResponsiveContainer>
              <LineChart data={sweep} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" /><XAxis dataKey="threshold" tick={AX} type="number" domain={[0.1, 0.9]} /><YAxis tick={AX} domain={["auto", "auto"]} />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} /><Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="valid" stroke="#64748b" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line dataKey="test" stroke="#0e7490" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line dataKey="bolivia" stroke="#f59e0b" dot={false} strokeWidth={2} isAnimationActive={false} />
                {ev.val_selected_threshold != null && <ReferenceLine x={ev.val_selected_threshold} stroke="#0f172a" strokeDasharray="4 3" />}
              </LineChart>
            </ResponsiveContainer>
          </ChartFrame>
          <Panel title="Confusion matrix" actions={<Segmented size="xs" label="Split" value={split} onChange={setSplit} options={[{ value: "test", label: "Test" }, { value: "bolivia", label: "Bolivia" }]} />}>
            {evAt(split) ? <Confusion m={evAt(split)!} /> : <Empty title="unavailable" />}
          </Panel>
        </div>

        <div className="mt-3 grid gap-3 xl:grid-cols-3">
          <Panel title="Test vs Bolivia (held-out event)" info="Same model, same threshold. Bolivia is ≥1,200 km from every training chip; the test split is not." body="p-0 overflow-x-auto">
            <table className="w-full text-[12px] tnum">
              <thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Metric</th><th className="px-3 py-1.5 text-right">Test</th><th className="px-3 py-1.5 text-right">Bolivia</th><th className="px-3 py-1.5 text-right">Δ</th><th className="px-3 py-1.5 text-right">NDWI Bolivia</th></tr></thead>
              <tbody className="divide-y divide-line-2">
                {(["iou_water", "f1_water", "precision_water", "recall_water", "miou", "pixel_accuracy"] as const).map((k) => {
                  const a = evAt("test")?.[k], b = evAt("bolivia")?.[k], n = metrics.data!.baselines?.s2_ndwi_gt0.splits.bolivia[k];
                  return <tr key={k}><td className="px-3 py-1.5 text-ink-2">{k.replace("_water", "")}</td><td className="px-3 py-1.5 text-right">{fmtNum(a)}</td>
                    <td className="px-3 py-1.5 text-right font-semibold">{fmtNum(b)}</td><td className={`px-3 py-1.5 text-right ${a != null && b != null && b < a ? "text-bad" : "text-ok"}`}>{a != null && b != null ? (b - a >= 0 ? "+" : "") + (b - a).toFixed(3) : "–"}</td>
                    <td className="px-3 py-1.5 text-right text-ink-2">{fmtNum(n)}</td></tr>;
                })}
              </tbody>
            </table>
          </Panel>
          <ChartFrame title={`Per-event IoU · ${split}`} source="evaluation.json (per_event)" rows={perEvent} columns={["event", "IoU", "F1", "recall", "water_share"]} height={210}>
            <ResponsiveContainer>
              <BarChart data={perEvent} margin={{ top: 6, right: 6, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" vertical={false} /><XAxis dataKey="event" tick={{ ...AX, fontSize: 9.5 }} interval={0} /><YAxis domain={[0, 1]} tick={AX} />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 11 }} /><Bar dataKey="IoU" fill="#0e7490" radius={[2, 2, 0, 0]} /><Bar dataKey="recall" fill="#7dd3fc" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
          <Panel title="Configuration & checkpoint">
            <KV dense items={[["Parameters", `${(ev.meta.n_params / 1e6).toFixed(2)} M`], ["Initialisation", ev.meta.pretrained], ["Trained on", ev.meta.trained_on], ["Selected on", ev.meta.selected_on],
              ["Best epoch", ev.checkpoint_epoch], ["Seed", ev.config.seed], ["Crop / batch", `${String(ev.config.data.crop_size)} px / ${String(ev.config.train.batch_size)}`],
              ["LR / weight decay", `${String(ev.config.train.lr)} / ${String(ev.config.train.weight_decay)}`],
              ["Radiometric aug.", (ev.config.data.radiometric as { enabled?: boolean } | undefined)?.enabled ? "on" : "off"],
              ["Training time", fmtDuration(expByName[run]?.training_time_s)], ["Peak VRAM train / infer", `${expByName[run]?.peak_train_gpu_mem_mib ?? "–"} / ${ev.inference_peak_gpu_mem_mib ?? "–"} MiB`],
              ["Latency (model, p95)", `${fmtMs(ev.latency.model_only_ms_mean)} (${fmtMs(ev.latency.model_only_ms_p95)})`], ["End-to-end", fmtMs(ev.latency.end_to_end_ms_mean)],
              ["Checkpoint", <span className="font-mono text-[10.5px]">{expByName[run]?.checkpoint ?? "–"}</span>]]} />
          </Panel>
        </div>

        <Panel title="Representative predictions · test split" className="mt-3" info="A deterministic selection across water fractions, including failure cases. Panels: S2 RGB, S1 false colour, hand label (reference), P(water), errors (TP blue, FP red, FN amber).">
          {ev.sample_images.length === 0 ? <Empty title="No sample images" /> : <div className="space-y-2">{ev.sample_images.map((n) => <img key={n} src={runImage(run, n)} alt={`Prediction sample ${n}`} loading="lazy" className="w-full rounded border border-line" />)}</div>}
        </Panel>
      </>)}

      <h2 className="mb-2 mt-5 text-[15px] font-semibold text-ink">Audit evidence</h2>
      {ext.error ? <ErrorState error={ext.error} /> : !ext.data ? <Skeleton className="h-40" /> : (
        <div className="grid gap-3 xl:grid-cols-3">
          <Panel title="Split independence" info="Distance from each held-out chip to the nearest training chip (chip centres), and footprint overlaps (scripts/audit_reproduce.py)." body="p-0 overflow-x-auto">
            {ext.data.audit ? (
              <table className="w-full text-[12px] tnum">
                <thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Split</th><th className="px-3 py-1.5 text-right">Median km</th><th className="px-3 py-1.5 text-right">≤10 km</th><th className="px-3 py-1.5 text-right">Overlaps</th></tr></thead>
                <tbody className="divide-y divide-line-2">{Object.entries(ext.data.audit.geographic_proximity_to_train).map(([s, v]) => (
                  <tr key={s}><td className="px-3 py-1.5">{s}</td><td className="px-3 py-1.5 text-right">{v.median_km_to_nearest_train_chip.toFixed(1)}</td><td className="px-3 py-1.5 text-right">{fmtPct(v.share_within_10km, 0)}</td>
                    <td className={`px-3 py-1.5 text-right ${v.chips_whose_footprint_overlaps_a_train_chip ? "font-semibold text-warn" : ""}`}>{v.chips_whose_footprint_overlaps_a_train_chip}</td></tr>))}</tbody>
              </table>
            ) : <Empty title="Audit not run" />}
          </Panel>
          <Panel title="Where the v1 model misses water" info="False-negative rate of the v1 main model by pixel stratum (scripts/analyze_weaknesses.py). NDWI>0 = optically obvious water." body="p-0 overflow-x-auto">
            {wa ? (
              <table className="w-full text-[12px] tnum">
                <thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Stratum</th><th className="px-3 py-1.5 text-right">Test FN</th><th className="px-3 py-1.5 text-right">Bolivia FN</th></tr></thead>
                <tbody className="divide-y divide-line-2">{Object.keys(wa.fn_rate_by_stratum.test).map((k) => (
                  <tr key={k}><td className="px-3 py-1.5">{k}</td><td className="px-3 py-1.5 text-right">{fmtPct(wa.fn_rate_by_stratum.test[k].fn_rate)}</td>
                    <td className="px-3 py-1.5 text-right" title={`${wa.fn_rate_by_stratum.bolivia[k].water_px.toLocaleString()} water pixels`}>{fmtPct(wa.fn_rate_by_stratum.bolivia[k].fn_rate)}</td></tr>))}</tbody>
              </table>
            ) : <Empty title="Not analysed" />}
          </Panel>
          <ChartFrame title="Calibration · v1 main model" source="weakness_analysis.json (reliability, 10 bins)" info="Observed water rate per predicted-probability bin; the dashed diagonal is perfect calibration."
            rows={rel} columns={["bin", "valid", "test", "bolivia", "ideal"]} height={200}>
            <ResponsiveContainer>
              <LineChart data={rel} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" /><XAxis dataKey="bin" tick={{ ...AX, fontSize: 9 }} /><YAxis domain={[0, 1]} tick={AX} />
                <Tooltip formatter={(v) => (v == null ? "n/a" : Number(v).toFixed(3))} /><Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="ideal" stroke="#cbd5e1" dot={false} strokeDasharray="4 3" isAnimationActive={false} />
                <Line dataKey="test" stroke="#0e7490" isAnimationActive={false} />
                <Line dataKey="bolivia" stroke="#f59e0b" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartFrame>
          {wa && <div className="xl:col-span-3"><Notice tone="info">ECE (10 bins): valid {wa.ece.valid.toFixed(3)} · test {wa.ece.test.toFixed(3)} · Bolivia {wa.ece.bolivia.toFixed(3)}. Threshold selected on validation for the v1 main model: {wa.valid_selected_threshold}.</Notice></div>}
        </div>
      )}

      {wf && (
        <Panel title="WorldFloods v2 · per scene (clear-sky IoU)" className="mt-3" info="External test: 18 Copernicus EMS flood scenes, Sentinel-2 only, never used for training or selection." body="p-0 overflow-x-auto">
          <DataTable rowKey={(r) => String(r.scene)} csvName="worldfloods_per_scene" maxHeight={360}
            rows={Object.entries(wf.per_scene_iou_water).map(([scene, d]) => ({ scene, date: ((d._meta as { s2_date?: string })?.s2_date ?? "").slice(0, 10),
              ...Object.fromEntries(wfCols.map((k) => [k, (d[k] as { clear: number } | undefined)?.clear ?? null])) })) as Record<string, string | number | null>[]}
            columns={[{ key: "scene", label: "Scene", value: (r) => r.scene as string, render: (r) => <span className="font-mono text-[11px]">{r.scene}</span> },
              { key: "date", label: "S2 date", value: (r) => r.date as string },
              ...wfCols.map((k) => ({ key: k, label: k.replace("_gt0", ">0"), numeric: true, value: (r: Record<string, string | number | null>) => r[k] as number | null }))]} />
        </Panel>
      )}

      {metrics.data.routing && (
        <Panel title="Routing evaluation against hand labels" className="mt-3" info="300 random origin–destination pairs in 6 held-out areas: routes planned on the model prediction, flooded length measured on the hand label." body="p-0"
          footer={<>{metrics.data.routing.overall.pairs_where_model_route_has_more_gt_flood_than_shortest} of {metrics.data.routing.overall.n_pairs} pairs got worse than the shortest route. Mean detour +{metrics.data.routing.overall.mean_detour_pct.toFixed(1)}%.</>}>
          <DataTable rowKey={(r) => r.area} csvName="routing_evaluation" rows={Object.entries(metrics.data.routing.areas).map(([area, a]) => ({ area, ...a }))}
            columns={[{ key: "a", label: "Area", value: (r) => r.area }, { key: "e", label: "Edge agreement", numeric: true, value: (r) => r.edge_agreement.accuracy },
              { key: "s", label: "Labelled flood · shortest", numeric: true, value: (r) => r.mean_gt_flooded_m.shortest, render: (r) => fmtDist(r.mean_gt_flooded_m.shortest) },
              { key: "m", label: "· risk-aware (model)", numeric: true, value: (r) => r.mean_gt_flooded_m.risk_aware_model, render: (r) => fmtDist(r.mean_gt_flooded_m.risk_aware_model) },
              { key: "o", label: "· oracle (label)", numeric: true, value: (r) => r.mean_gt_flooded_m.risk_aware_oracle, render: (r) => fmtDist(r.mean_gt_flooded_m.risk_aware_oracle) }]} />
        </Panel>
      )}
      <div className="mt-3"><Notice tone="truth">Hand labels (violet) are reference data. Every model number on this page is read from files written by training/evaluate.py, scripts/evaluate_worldfloods.py and scripts/evaluate_routing.py.</Notice></div>
    </div>
  );
}
