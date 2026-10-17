import { Fragment, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3 } from "lucide-react";
import { api, runImage, type Metrics, type RoutingEvaluation } from "../api";
import { Badge, Callout, Card, EmptyState, ErrorState, KV, PageHeader, Skeleton } from "../components/ui";
import { fmtDist, fmtInt, fmtMs, fmtNum, fmtPct, modelLabel, useApi } from "../lib";

const SPLIT_LABEL: Record<string, string> = { test: "Test (held-out chips)", bolivia: "Bolivia (unseen event)", valid: "Validation (model selection)" };

function ConfusionMatrix({ m }: { m: Metrics }) {
  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  const rowT = [tn + fp, fn + tp];
  const cells = [[tn, fp], [fn, tp]];
  const names = ["Not water", "Water"];
  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center text-[12px]">
        <div />
        {names.map((n) => <div key={n} className="pb-1 text-[11px] font-medium text-slate-500">Pred. {n.toLowerCase()}</div>)}
        {cells.map((row, i) => (
          <Fragment key={i}>
            <div className="flex items-center pr-2 text-right text-[11px] font-medium text-slate-500">True {names[i].toLowerCase()}</div>
            {row.map((v, j) => {
              const share = v / Math.max(1, rowT[i]);
              const correct = i === j;
              return (
                <div key={`${i}${j}`} className="rounded-md px-2 py-3"
                  style={{ background: correct ? `rgba(37,99,235,${0.12 + 0.75 * share})` : `rgba(220,38,38,${0.08 + 0.8 * share})`, color: share > 0.55 ? "white" : "#0f172a" }}>
                  <div className="text-base font-semibold tabular-nums">{fmtPct(share)}</div>
                  <div className="text-[10px] tabular-nums opacity-80">{fmtInt(v)} px</div>
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-slate-500">Row-normalised. Positive class = water; pixels labelled −1 (no data) excluded.</p>
    </div>
  );
}

export default function Evaluation() {
  const metrics = useApi(() => api.metrics(), []);
  const [run, setRun] = useState("");
  const [split, setSplit] = useState<"test" | "bolivia">("test");
  const runs = metrics.data?.runs ?? {};
  const names = Object.keys(runs);

  useEffect(() => {
    if (!run && names.length) setRun(names.includes("unet_resnet34_s1s2") ? "unet_resnet34_s1s2" : names[0]);
  }, [names, run]);

  const ev = run ? runs[run] : undefined;
  const comparison = useMemo(() => {
    const rows = Object.values(runs).map((r) => ({
      name: modelLabel({ arch: r.config.model.arch, encoder: r.config.model.encoder, modality: r.modality }), key: r.run, kind: "model" as const,
      test: r.splits.test, bolivia: r.splits.bolivia, params: r.meta.n_params, latency: r.latency.model_only_ms_mean,
    }));
    const base = Object.entries(metrics.data?.baselines ?? {}).map(([k, b]) => ({
      name: k.replace(/_/g, " "), key: k, kind: "baseline" as const, test: b.splits.test, bolivia: b.splits.bolivia, params: null, latency: null, desc: b.description,
    }));
    return [...rows.sort((a, b) => b.test.iou_water - a.test.iou_water), ...base.sort((a, b) => b.test.iou_water - a.test.iou_water)];
  }, [runs, metrics.data]);

  if (metrics.error) return <ErrorState error={metrics.error} onRetry={metrics.reload} />;
  if (metrics.loading && !metrics.data) return <div className="space-y-4"><Skeleton className="h-10 w-80" /><Skeleton className="h-72" /></div>;
  if (!names.length)
    return (
      <div>
        <PageHeader eyebrow="Evaluation" title="Model evaluation" />
        <EmptyState icon={<BarChart3 className="h-9 w-9" />} title="No evaluated models yet">
          Run <code>python -m training.evaluate --all</code> after training. Results are read from artifacts/runs/*/evaluation.json.
        </EmptyState>
      </div>
    );

  const eventData = ev ? Object.entries(ev.per_event[split] ?? {}).map(([e, m]) => ({ event: e, IoU: +m.iou_water.toFixed(4), F1: +m.f1_water.toFixed(4), water: m.water_fraction_gt })) : [];
  const chartData = comparison.map((r) => ({ name: r.name.replace(" · ", "\n"), Test: +r.test.iou_water.toFixed(4), Bolivia: +r.bolivia.iou_water.toFixed(4), kind: r.kind }));

  return (
    <div>
      <PageHeader eyebrow="Evaluation" title="Model evaluation">
        {metrics.data?.summary?.protocol ? `Protocol: ${metrics.data.summary.protocol}.` : "Held-out evaluation."} The test split shares flood events with training but uses disjoint chips;
        Bolivia is an entire flood event never seen during training or model selection.
      </PageHeader>

      <Card title="Held-out comparison" subtitle="Trained models (top) and non-learned baselines (bottom) under the same protocol" bodyClass="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead className="bg-slate-50 text-[10.5px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Method</th>
                <th className="px-2 py-2 text-right font-medium">Test IoU</th><th className="px-2 py-2 text-right font-medium">Test F1</th>
                <th className="px-2 py-2 text-right font-medium">Precision</th><th className="px-2 py-2 text-right font-medium">Recall</th>
                <th className="px-2 py-2 text-right font-medium">mIoU</th><th className="px-2 py-2 text-right font-medium">Pixel acc.</th>
                <th className="px-2 py-2 text-right font-medium">Bolivia IoU</th><th className="px-2 py-2 text-right font-medium">Chip-mean IoU</th>
                <th className="px-2 py-2 text-right font-medium">Params</th><th className="px-4 py-2 text-right font-medium">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {comparison.map((r) => (
                <tr key={r.key} className={`${r.kind === "baseline" ? "bg-slate-50/50 text-slate-600" : ""} ${r.key === run ? "bg-water-50/60" : ""}`}>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      {r.kind === "model" ? <button onClick={() => setRun(r.key)} className="text-left font-medium text-slate-900 hover:text-water-700">{r.name}</button>
                        : <span title={"desc" in r ? r.desc : ""}>{r.name}</span>}
                      {r.kind === "baseline" && <Badge>baseline</Badge>}
                    </div>
                  </td>
                  <td className="px-2 py-2 text-right font-semibold tabular-nums">{fmtNum(r.test.iou_water)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{fmtNum(r.test.f1_water)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{fmtNum(r.test.precision_water)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{fmtNum(r.test.recall_water)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{fmtNum(r.test.miou)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{fmtNum(r.test.pixel_accuracy)}</td>
                  <td className="px-2 py-2 text-right font-semibold tabular-nums">{fmtNum(r.bolivia.iou_water)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{fmtNum(r.test.mean_chip_iou_water)}</td>
                  <td className="px-2 py-2 text-right tabular-nums">{r.params ? `${(r.params / 1e6).toFixed(1)} M` : "–"}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{r.latency ? fmtMs(r.latency) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="border-t border-slate-100 px-4 py-2 text-[11px] text-slate-500">
          IoU/F1/precision/recall: water class, pooled over all valid pixels of the split. Chip-mean IoU averages per-chip IoU (chips with no water in label and prediction skipped).
          Latency: model forward pass, 1×C×512×512, fp16 autocast, RTX 3050 Laptop GPU.
        </div>
      </Card>

      <Card title="IoU (water) on held-out data" className="mt-6">
        <div className="h-72">
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 40, left: 28 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => Number(v).toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" />
              <Bar dataKey="Test" fill="#2563eb" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Bolivia" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {metrics.data?.routing && <RoutingEval r={metrics.data.routing} />}

      {ev && (
        <>
          <div className="mt-8 mb-3 flex flex-wrap items-center gap-3">
            <h2 className="text-lg font-semibold text-slate-900">Run details</h2>
            <select value={run} onChange={(e) => setRun(e.target.value)} aria-label="Run"
              className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-[13px]">
              {names.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            <div className="ml-auto flex gap-1 rounded-lg bg-slate-100 p-1 text-xs font-medium">
              {(["test", "bolivia"] as const).map((s) => (
                <button key={s} onClick={() => setSplit(s)} className={`rounded-md px-2.5 py-1 ${split === s ? "bg-white shadow-sm" : "text-slate-600"}`}>{SPLIT_LABEL[s]}</button>
              ))}
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-3">
            <Card title="Training curves" subtitle="Per-epoch, from history.json" className="xl:col-span-2">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="h-60">
                  <ResponsiveContainer>
                    <LineChart data={ev.history} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="train_loss" name="train loss" stroke="#64748b" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="val_loss" name="val loss" stroke="#dc2626" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="h-60">
                  <ResponsiveContainer>
                    <LineChart data={ev.history} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0.4, 1]} tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="val_iou_water" name="val IoU" stroke="#2563eb" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="val_f1_water" name="val F1" stroke="#059669" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <p className="mt-2 text-[11px] text-slate-500">Best checkpoint = epoch {ev.checkpoint_epoch} (highest validation IoU). Early stopping patience: {String(ev.config.train.early_stopping_patience)} epochs.</p>
            </Card>
            <Card title="Confusion matrix" subtitle={SPLIT_LABEL[split]}>
              <ConfusionMatrix m={ev.splits[split]} />
            </Card>
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-3">
            <Card title="Per-event IoU" subtitle={`${SPLIT_LABEL[split]} · water fraction varies widely by event`} className="xl:col-span-2">
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={eventData} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                    <XAxis dataKey="event" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="IoU" fill="#2563eb" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="F1" fill="#93c5fd" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card title="Checkpoint & latency">
              <KV items={[
                ["Architecture", modelLabel({ arch: ev.config.model.arch, encoder: ev.config.model.encoder, modality: ev.modality })],
                ["Parameters", `${(ev.meta.n_params / 1e6).toFixed(2)} M`],
                ["Initialisation", ev.meta.pretrained],
                ["Trained on", ev.meta.trained_on],
                ["Selected on", ev.meta.selected_on],
                ["Best epoch", ev.checkpoint_epoch],
                ["Model latency", `${fmtMs(ev.latency.model_only_ms_mean)} (p95 ${fmtMs(ev.latency.model_only_ms_p95)})`],
                ["End-to-end", fmtMs(ev.latency.end_to_end_ms_mean)],
                ["Throughput", `${ev.latency.throughput_fps_model_only.toFixed(1)} chips/s`],
                ["Hardware", ev.latency.gpu ?? ev.latency.device],
              ]} />
              <p className="mt-2 text-[11px] text-slate-500">End-to-end = {ev.latency.end_to_end_includes}.</p>
            </Card>
          </div>

          <Card title="Sample predictions" subtitle="Deterministic selection of test chips spanning low to high water fraction · panels: S2 RGB, S1 false colour, hand label, P(water), errors" className="mt-6" bodyClass="space-y-3 p-4">
            {ev.sample_images.length === 0 ? <EmptyState title="No sample images" /> : ev.sample_images.map((n) => (
              <img key={n} src={runImage(run, n)} alt={`Prediction sample ${n}`} loading="lazy" className="w-full rounded-lg border border-slate-200" />
            ))}
          </Card>
          <div className="mt-4"><Callout>Ground-truth panels are Sen1Floods11 hand labels. The probability and error panels are this model's predictions on chips it never saw during training.</Callout></div>
        </>
      )}
    </div>
  );
}

function RoutingEval({ r }: { r: RoutingEvaluation }) {
  const o = r.overall;
  return (
    <Card title="Routing evaluation against hand labels" className="mt-6"
      subtitle={`${o.n_pairs} random origin–destination pairs (${r.protocol.pairs_per_area} per study area, seed ${r.protocol.seed}). Routes are planned on the model prediction and scored on the Sen1Floods11 hand label.`}>
      <div className="grid gap-4 sm:grid-cols-4">
        <div><div className="text-[11px] text-slate-500">Labelled-flooded road per trip</div>
          <div className="text-sm"><span className="font-semibold">{fmtDist(o.mean_gt_flooded_m.shortest)}</span> shortest → <span className="font-semibold text-emerald-700">{fmtDist(o.mean_gt_flooded_m.risk_aware_model)}</span> risk-aware</div></div>
        <div><div className="text-[11px] text-slate-500">Reduction vs shortest</div><div className="text-xl font-semibold text-emerald-700">{fmtPct(o.reduction_in_gt_flooded_length_vs_shortest)}</div></div>
        <div><div className="text-[11px] text-slate-500">Oracle (plans on the hand label)</div><div className="text-sm font-semibold">{fmtDist(o.mean_gt_flooded_m.risk_aware_oracle)} per trip</div></div>
        <div><div className="text-[11px] text-slate-500">Mean detour</div><div className="text-xl font-semibold">+{o.mean_detour_pct.toFixed(1)}%</div></div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-[12.5px]">
          <thead className="text-left text-[10.5px] uppercase tracking-wider text-slate-500">
            <tr><th className="py-1.5 font-medium">Area</th><th className="py-1.5 text-right font-medium">Edge agreement</th>
              <th className="py-1.5 text-right font-medium">Flooded road (model / label)</th><th className="py-1.5 text-right font-medium">Labelled flood: shortest</th>
              <th className="py-1.5 text-right font-medium">Risk-aware</th><th className="py-1.5 text-right font-medium">Oracle</th></tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {Object.entries(r.areas).map(([k, a]) => (
              <tr key={k}>
                <td className="py-1.5 font-mono text-[11.5px]">{k}</td>
                <td className="py-1.5 text-right tabular-nums">{fmtPct(a.edge_agreement.accuracy)}</td>
                <td className="py-1.5 text-right tabular-nums">{a.flooded_road_km_model.toFixed(1)} / {a.flooded_road_km_label.toFixed(1)} km</td>
                <td className="py-1.5 text-right tabular-nums">{fmtDist(a.mean_gt_flooded_m.shortest)}</td>
                <td className="py-1.5 text-right tabular-nums">{fmtDist(a.mean_gt_flooded_m.risk_aware_model)}</td>
                <td className="py-1.5 text-right tabular-nums">{fmtDist(a.mean_gt_flooded_m.risk_aware_oracle)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-slate-500">In {o.pairs_where_model_route_has_more_gt_flood_than_shortest} of {o.n_pairs} pairs the model-planned route crossed more labelled flood than the shortest route. Where flooding blocks every alternative, little reduction is possible. Flood penalty 10×, buffer 1 px, threshold 0.5.</p>
    </Card>
  );
}
