import { Link } from "react-router-dom";
import { AlertOctagon, AlertTriangle, ArrowUpRight, Info, Map, ScanLine } from "lucide-react";
import { api, type HistoryItem } from "../api";
import { Badge, Button, DataTable, Empty, ErrorState, KV, Metric, PageHead, Panel, Skeleton } from "../components/ui";
import { ARCH_LABEL, METRIC_INFO, encoderLabel, fmtBytes, fmtDist, fmtMs, fmtNum, fmtPct, fmtTime, useApi } from "../lib";

const SEV = {
  critical: { icon: AlertOctagon, cls: "text-bad" },
  warning: { icon: AlertTriangle, cls: "text-warn" },
  info: { icon: Info, cls: "text-accent" },
};

export default function Command() {
  const models = useApi(() => api.modelStatus(), [], 30000);
  const metrics = useApi(() => api.metrics(), []);
  const ext = useApi(() => api.external(), []);
  const data = useApi(() => api.datasets(), []);
  const sys = useApi(() => api.system(), [], 10000);
  const warn = useApi(() => api.warnings(), [], 30000);
  const seg = useApi(() => api.history({ kind: "segmentation", limit: 6 }), [], 15000);
  const routes = useApi(() => api.history({ kind: "route", limit: 6 }), [], 15000);
  const areas = useApi(() => api.areas(), []);

  const m = models.data;
  const active = m?.models.find((x) => x.name === m.default_model);
  const ev = active ? metrics.data?.runs[active.name] : undefined;
  const thr = ev?.val_selected_threshold;
  const at = (split: string) => (ev?.splits_at_val_threshold?.[split] ?? ev?.splits[split]);
  const ndwi = metrics.data?.baselines?.s2_ndwi_gt0;
  const wf = ext.data?.worldfloods;
  const wfBest = wf ? Object.entries(wf.totals).filter(([k]) => !k.startsWith("ndwi") && !k.startsWith("mndwi") && k.includes("@val"))
    .sort((a, b) => b[1].clear.iou_water - a[1].clear.iou_water)[0] : undefined;
  const routingEval = metrics.data?.routing;
  const gpu = sys.data?.gpu;

  return (
    <div>
      <PageHead title="Command overview"
        sub="System readiness, the active model's held-out performance and recent analyses. Every value on this page comes from the running backend or from recorded evaluation files." />

      <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3 xl:grid-cols-6">
        {!m ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-[78px]" />) : (<>
          <Metric label="Readiness" tone={m.ready ? "ok" : "bad"} value={m.ready ? "Ready" : "Not ready"}
            hint={`${m.models.filter((x) => x.training_state === "completed").length} models · ${m.cuda_available ? "CUDA" : "CPU"} inference`} />
          <Metric label="Active model" value={<span className="text-[15px]">{active ? `${ARCH_LABEL[active.arch] ?? active.arch} · ${encoderLabel(active.encoder)}` : m.default_model}</span>}
            hint={active ? `${active.modality.toUpperCase()} · ${active.name} · epoch ${active.best_epoch ?? "–"}` : "not found"} source="/api/model/status" />
          <Metric label="Test IoU" tone="water" value={fmtNum(at("test")?.iou_water)} info={METRIC_INFO.test}
            hint={thr != null ? `threshold ${thr} (chosen on valid)` : "threshold 0.5"} source="evaluation.json" />
          <Metric label="Bolivia IoU (unseen)" tone="water" value={fmtNum(at("bolivia")?.iou_water)} info={METRIC_INFO.bolivia}
            hint={ndwi ? `NDWI baseline ${fmtNum(ndwi.splits.bolivia.iou_water)}` : undefined} source="evaluation.json, baselines.json" />
          <Metric label="Inference latency" value={ev ? fmtMs(ev.latency.model_only_ms_mean) : "–"} info={METRIC_INFO.latency}
            hint={ev ? `512² · ${ev.latency.precision}` : "not benchmarked"} />
          <Metric label="GPU" value={gpu ? <span className="tnum text-[18px]">{gpu.memory_used_mib}<span className="text-[12px] text-ink-3">/{gpu.memory_total_mib} MiB</span></span> : "n/a"}
            hint={gpu ? `${gpu.name.replace("NVIDIA GeForce ", "")} · ${gpu.utilization_pct}%` : "NVML unavailable"} source="NVML (live)" />
        </>)}
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-3">
        <div className="min-w-0 space-y-3 xl:col-span-2">
          <Panel title="Held-out performance · active model vs strongest baseline" info="Pooled water IoU. Learned model at its validation-selected threshold; NDWI uses its fixed 0 threshold."
            actions={<Link to="/evaluation" className="text-[11.5px] font-medium text-accent hover:underline">Evaluation lab →</Link>} body="p-0 overflow-x-auto">
            {metrics.error ? <div className="p-3"><ErrorState error={metrics.error} onRetry={metrics.reload} /></div> : !metrics.data ? <Skeleton className="m-3 h-24" /> : (
              <table className="w-full text-[12.5px]">
                <thead className="bg-slate-50 text-[10.5px] uppercase tracking-[0.06em] text-ink-3">
                  <tr><th className="px-3 py-2 text-left font-semibold">Evaluation set</th><th className="px-3 py-2 text-right font-semibold">Active model</th>
                    <th className="px-3 py-2 text-right font-semibold">NDWI &gt; 0</th><th className="px-3 py-2 text-left font-semibold">Note</th></tr>
                </thead>
                <tbody className="divide-y divide-line-2 tnum">
                  <tr><td className="px-3 py-2">Sen1Floods11 test (90 chips)</td><td className="px-3 py-2 text-right font-semibold">{fmtNum(at("test")?.iou_water)}</td>
                    <td className="px-3 py-2 text-right">{fmtNum(ndwi?.splits.test.iou_water)}</td><td className="px-3 py-2 text-[11.5px] text-ink-2">in-distribution; spatially close to training</td></tr>
                  <tr><td className="px-3 py-2">Bolivia (15 chips, unseen event)</td><td className="px-3 py-2 text-right font-semibold">{fmtNum(at("bolivia")?.iou_water)}</td>
                    <td className="px-3 py-2 text-right">{fmtNum(ndwi?.splits.bolivia.iou_water)}</td><td className="px-3 py-2 text-[11.5px] text-ink-2">independent region</td></tr>
                  <tr><td className="px-3 py-2">WorldFloods v2 test (18 scenes)</td>
                    <td className="px-3 py-2 text-right font-semibold">{active?.modality.startsWith("s1") ? <span className="font-normal text-ink-3" title="WorldFloods has no Sentinel-1; the active model needs SAR input">n/a</span> : "–"}</td>
                    <td className="px-3 py-2 text-right">{fmtNum(wf?.totals.ndwi_gt0?.clear.iou_water)}</td>
                    <td className="px-3 py-2 text-[11.5px] text-ink-2">{wfBest ? `best S2-only model ${wfBest[0].split("@")[0]}: ${fmtNum(wfBest[1].clear.iou_water)}` : "external; S2-only models only"}</td></tr>
                </tbody>
              </table>
            )}
          </Panel>

          <Panel title="Study areas · flood-risk on roads" info="Held-out test chips with a cached OpenStreetMap drive network. Flooded road length = model prediction (threshold 0.5, 1 px buffer) vs hand label, from scripts/evaluate_routing.py."
            actions={<Link to="/routes" className="text-[11.5px] font-medium text-accent hover:underline">Route planner →</Link>} body="p-0 overflow-x-auto">
            {!areas.data ? <Skeleton className="m-3 h-32" /> : (
              <DataTable rowKey={(r) => r.id} rows={areas.data.areas}
                columns={[
                  { key: "name", label: "Area", value: (r) => r.name, render: (r) => <Link to={`/routes?area=${r.id}`} className="font-medium text-ink hover:text-accent">{r.name}</Link> },
                  { key: "edges", label: "Road edges", numeric: true, value: (r) => routingEval?.areas[r.id]?.edges ?? r.road_meta?.edges ?? null },
                  { key: "fm", label: "Flooded road · model", numeric: true, value: (r) => routingEval?.areas[r.id]?.flooded_road_km_model ?? null, render: (r) => routingEval?.areas[r.id] ? `${routingEval.areas[r.id].flooded_road_km_model.toFixed(1)} km` : "–" },
                  { key: "fl", label: "· hand label", numeric: true, value: (r) => routingEval?.areas[r.id]?.flooded_road_km_label ?? null, render: (r) => routingEval?.areas[r.id] ? <span className="text-truth">{routingEval.areas[r.id].flooded_road_km_label.toFixed(1)} km</span> : "–" },
                  { key: "agree", label: "Edge agreement", numeric: true, value: (r) => routingEval?.areas[r.id]?.edge_agreement.accuracy ?? null, render: (r) => fmtPct(routingEval?.areas[r.id]?.edge_agreement.accuracy) },
                  { key: "cached", label: "Offline", value: (r) => (r.road_graph_cached ? "cached" : "not cached"), render: (r) => r.road_graph_cached ? <Badge tone="ok">cached</Badge> : <Badge tone="warn">needs network</Badge> },
                ]} />
            )}
          </Panel>

          <div className="grid gap-3 lg:grid-cols-2">
            <RecentPanel title="Recent segmentation analyses" icon={<ScanLine className="h-4 w-4" />} items={seg.data?.items} total={seg.data?.total} to="/segmentation"
              line={(i) => `${i.model ?? ""} · ${fmtPct(i.summary.water_fraction as number)} water${i.summary.water_area_km2 != null ? ` · ${(i.summary.water_area_km2 as number).toFixed(2)} km²` : ""}`}
              empty="No segmentation has been run yet." />
            <RecentPanel title="Recent route analyses" icon={<Map className="h-4 w-4" />} items={routes.data?.items} total={routes.data?.total} to="/routes"
              line={(i) => `${fmtDist(i.summary.risk_aware_m as number)} · flooded ${fmtDist(i.summary.risk_aware_flooded_m as number)} (shortest ${fmtDist(i.summary.baseline_flooded_m as number)})`}
              badge={(i) => <Badge tone={i.summary.status === "flood_free_route_found" ? "ok" : "bad"}>{String(i.summary.status).replace(/_/g, " ")}</Badge>}
              empty="No route has been planned yet." />
          </div>
        </div>

        <div className="min-w-0 space-y-3">
          <Panel title="Warnings & data quality" info="Derived from verification reports, run status files, GPU/RAM/disk probes and the API error log." body="p-0"
            actions={warn.data && <Badge tone={warn.data.warnings.some((w) => w.severity === "critical") ? "bad" : warn.data.warnings.length ? "warn" : "ok"}>{warn.data.warnings.length}</Badge>}>
            {warn.error ? <div className="p-3"><ErrorState error={warn.error} /></div> : !warn.data ? <Skeleton className="m-3 h-32" /> :
              warn.data.warnings.length === 0 ? <Empty title="No active warnings" /> : (
                <ul className="max-h-[360px] divide-y divide-line-2 overflow-auto scroll-thin">
                  {warn.data.warnings.map((w, i) => {
                    const S = SEV[w.severity];
                    return (
                      <li key={i} className="flex gap-2 px-3 py-2 text-[12px] leading-snug">
                        <S.icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${S.cls}`} aria-label={w.severity} />
                        <div className="min-w-0"><div className="text-ink">{w.message}</div><div className="text-[10.5px] uppercase tracking-wide text-ink-3">{w.source}</div></div>
                      </li>
                    );
                  })}
                </ul>
              )}
          </Panel>

          <Panel title="Data" actions={<Link to="/datasets" className="text-[11.5px] font-medium text-accent hover:underline">Explorer →</Link>}>
            {!data.data ? <Skeleton className="h-28" /> : (
              <KV dense items={[
                ["Sen1Floods11", data.data.verification?.passed ? <Badge tone="ok">{data.data.verification.files_ok}/{data.data.verification.files_checked} {data.data.verification.integrity_mode}</Badge> : <Badge tone="warn">unverified</Badge>],
                ["Chips (train/valid/test/Bolivia)", ["train", "valid", "test", "bolivia"].map((s) => data.data!.splits[s]?.chips ?? 0).join(" / ")],
                ["WorldFloods v2 test", data.data.external ? <Badge tone={data.data.external.n_failed === 0 ? "ok" : "bad"}>{data.data.external.scenes} scenes · SHA-256</Badge> : <span className="text-ink-3">not downloaded</span>],
                ["Raw + processed", fmtBytes((data.data.storage.raw_sen1floods11 ?? 0) + (data.data.storage.raw_worldfloods_test ?? 0) + (data.data.storage.processed_chips ?? 0))],
                ["Checkpoints", fmtBytes(data.data.storage.checkpoints)],
              ]} />
            )}
          </Panel>

          <Panel title="Routing evaluation" info="300 random origin–destination pairs over 6 held-out areas; routes planned on model predictions, flooded length measured on hand labels.">
            {!routingEval ? <Empty title="Not evaluated">Run scripts/evaluate_routing.py</Empty> : (
              <KV dense items={[
                ["Labelled-flooded road per trip", <span>{fmtDist(routingEval.overall.mean_gt_flooded_m.shortest)} → <b className="text-ok">{fmtDist(routingEval.overall.mean_gt_flooded_m.risk_aware_model)}</b></span>],
                ["Reduction vs shortest", fmtPct(routingEval.overall.reduction_in_gt_flooded_length_vs_shortest)],
                ["Oracle (plans on label)", fmtDist(routingEval.overall.mean_gt_flooded_m.risk_aware_oracle)],
                ["Mean detour", `+${routingEval.overall.mean_detour_pct.toFixed(1)}%`],
              ]} />
            )}
          </Panel>

          <Panel title="Capabilities & limits">
            <ul className="space-y-1.5 text-[12px] leading-relaxed text-ink-2">
              <li>• Water segmentation from Sentinel-1 SAR, Sentinel-2 optical or RGB, using models trained on this machine.</li>
              <li>• Georeferenced predictions are laid over OpenStreetMap roads; Dijkstra compares shortest and flood-penalised routes.</li>
              <li className="text-warn">• "Water" includes permanent water; depth and passability are unknown. Imagery is from 2016–2019, roads are current. Not a safety system.</li>
            </ul>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function RecentPanel({ title, icon, items, total, to, line, badge, empty }: {
  title: string; icon: React.ReactNode; items?: HistoryItem[]; total?: number; to: string; line: (i: HistoryItem) => string;
  badge?: (i: HistoryItem) => React.ReactNode; empty: string;
}) {
  return (
    <Panel title={title} body="p-0" actions={<Link to="/history" className="text-[11.5px] font-medium text-accent hover:underline">{total ?? 0} total →</Link>}>
      {!items ? <Skeleton className="m-3 h-24" /> : items.length === 0 ? (
        <Empty icon={icon} title={empty} action={<Button size="xs" variant="secondary" href={to}>Open <ArrowUpRight className="h-3 w-3" /></Button>} />
      ) : (
        <ul className="divide-y divide-line-2">
          {items.map((i) => (
            <li key={i.id} className="flex items-center justify-between gap-2 px-3 py-2">
              <div className="min-w-0">
                <div className="truncate text-[12.5px] font-medium text-ink">{i.title}</div>
                <div className="truncate text-[11px] text-ink-2">{line(i)}</div>
              </div>
              <div className="shrink-0 text-right">
                {badge?.(i)}
                <div className="text-[10.5px] text-ink-3">{fmtTime(i.created)}</div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
