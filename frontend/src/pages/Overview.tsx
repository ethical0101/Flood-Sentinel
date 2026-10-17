import { Link } from "react-router-dom";
import { Activity, ArrowRight, CheckCircle2, Cpu, Database, Gauge, HardDrive, Map, ScanLine, XCircle } from "lucide-react";
import { api } from "../api";
import { Badge, Callout, Card, ErrorState, KV, PageHeader, Skeleton, Stat, StateBadge } from "../components/ui";
import { fmtBytes, fmtMs, fmtPct, modelLabel, useApi } from "../lib";

export default function Overview() {
  const models = useApi(() => api.modelStatus(), [], 15000);
  const data = useApi(() => api.datasets(), []);
  const sys = useApi(() => api.system(), [], 10000);
  const metrics = useApi(() => api.metrics(), []);
  const recent = useApi(() => api.recent(), [], 8000);

  const main = models.data?.models.find((m) => m.name === models.data?.default_model);
  const mainEval = main ? metrics.data?.runs[main.name] : undefined;
  const bestBaseline = metrics.data?.baselines
    ? Object.entries(metrics.data.baselines).sort((a, b) => b[1].splits.test.iou_water - a[1].splits.test.iou_water)[0]
    : undefined;

  return (
    <div>
      <PageHeader eyebrow="Real-time flood segmentation & safe-route prediction" title="Operations overview">
        A locally trained deep-learning model maps surface water in Sentinel-1/2 satellite imagery; the predicted
        flood extent is laid over the OpenStreetMap road network to compare a shortest route with a flood-risk-aware route.
      </PageHeader>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {models.loading && !models.data ? <Skeleton className="h-24" /> : (
          <Stat label="Model readiness" icon={<ScanLine className="h-3.5 w-3.5" />}
            value={models.data?.ready ? <span className="flex items-center gap-2"><CheckCircle2 className="h-6 w-6 text-emerald-600" />Ready</span> :
              <span className="flex items-center gap-2"><XCircle className="h-6 w-6 text-red-500" />Not ready</span>}
            hint={`${models.data?.models.filter((m) => m.training_state === "completed").length ?? 0} trained model(s) · ${models.data?.cuda_available ? "CUDA inference" : "CPU inference"}`} />
        )}
        <Stat label="Held-out test IoU (water)" tone="water" icon={<Gauge className="h-3.5 w-3.5" />}
          value={mainEval ? fmtPct(mainEval.splits.test.iou_water) : "–"}
          hint={mainEval ? `${main && modelLabel(main)} · best baseline ${bestBaseline ? fmtPct(bestBaseline[1].splits.test.iou_water) : "–"}` : "Evaluation not run yet"} />
        <Stat label="Unseen-region IoU (Bolivia)" tone="water" icon={<Map className="h-3.5 w-3.5" />}
          value={mainEval ? fmtPct(mainEval.splits.bolivia.iou_water) : "–"}
          hint="Event never seen in training or validation" />
        <Stat label="Inference latency (512²)" icon={<Activity className="h-3.5 w-3.5" />}
          value={mainEval ? fmtMs(mainEval.latency.model_only_ms_mean) : "–"}
          hint={mainEval ? `${mainEval.latency.precision}, batch 1, model only` : "Benchmark pending"} />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Dataset" subtitle="Authentic hand-labelled flood imagery" className="xl:col-span-1"
          action={data.data?.verification?.passed ? <Badge color="green">verified</Badge> : <Badge color="amber">unverified</Badge>}>
          {data.error ? <ErrorState error={data.error} onRetry={data.reload} /> : !data.data ? <Skeleton className="h-32" /> : (
            <KV items={[
              ["Dataset", "Sen1Floods11 v1.1 (hand-labelled)"],
              ["Chips", `${data.data.verification?.unique_chips ?? "–"} × 512² px, 10 m`],
              ["Files verified", `${data.data.verification?.files_ok ?? 0}/${data.data.verification?.files_checked ?? 0} (${data.data.verification?.integrity_mode})`],
              ["Splits", Object.entries(data.data.splits).map(([k, v]) => `${k} ${v.chips}`).join(" · ")],
              ["On disk", `${fmtBytes(data.data.raw_bytes_on_disk)} raw · ${fmtBytes(data.data.processed_bytes_on_disk)} processed`],
              ["Water share (train)", fmtPct(data.data.train_water_fraction)],
            ]} />
          )}
        </Card>

        <Card title="Compute" subtitle="Live readings from this machine" className="xl:col-span-1"
          action={<Cpu className="h-4 w-4 text-slate-400" />}>
          {sys.error ? <ErrorState error={sys.error} onRetry={sys.reload} /> : !sys.data ? <Skeleton className="h-32" /> : (
            <>
              <KV items={[
                ["GPU", sys.data.gpu?.name ?? "not available"],
                ["VRAM", sys.data.gpu ? `${sys.data.gpu.memory_used_mib} / ${sys.data.gpu.memory_total_mib} MiB` : "–"],
                ["GPU utilisation", sys.data.gpu ? `${sys.data.gpu.utilization_pct}% · ${sys.data.gpu.temperature_c} °C` : "–"],
                ["PyTorch", `${sys.data.torch.version}${sys.data.torch.cuda_available ? " · CUDA " + sys.data.torch.cuda_build : " · CPU"}`],
                ["RAM", `${sys.data.ram.available_gib} GiB free of ${sys.data.ram.total_gib}`],
                [<span className="flex items-center gap-1"><HardDrive className="h-3 w-3" />Disk {sys.data.disk.drive}</span>, `${sys.data.disk.free_gb} GB free`],
              ]} />
            </>
          )}
        </Card>

        <Card title="Recent inference" subtitle="Predictions served since the API started" className="xl:col-span-1"
          action={<Link to="/segmentation" className="text-xs font-medium text-water-600 hover:underline">Open studio</Link>}>
          {!recent.data ? <Skeleton className="h-32" /> : recent.data.recent.length === 0 ? (
            <div className="py-6 text-center text-sm text-slate-500">No predictions yet in this session.<br />
              <Link to="/segmentation" className="text-water-600 hover:underline">Run your first segmentation →</Link></div>
          ) : (
            <ul className="divide-y divide-slate-100 text-[13px]">
              {recent.data.recent.slice(0, 6).map((r) => (
                <li key={r.id} className="flex items-center justify-between gap-3 py-2">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-slate-800">{r.source.chip_id ?? r.source.filename ?? "upload"}</div>
                    <div className="text-xs text-slate-500">{r.model} · {new Date(r.created).toLocaleTimeString()}</div>
                  </div>
                  <div className="text-right text-xs">
                    <div className="font-semibold text-water-700">{fmtPct(r.stats.water_fraction)} water</div>
                    <div className="text-slate-500">{fmtMs(r.total_ms)}</div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-5">
        <Card title="Trained models" subtitle="All checkpoints trained locally on the Sen1Floods11 train split" className="xl:col-span-3" bodyClass="p-0">
          {models.error ? <div className="p-5"><ErrorState error={models.error} onRetry={models.reload} /></div> : (
            <div className="overflow-x-auto">
              <table className="w-full text-[13px]">
                <thead className="bg-slate-50 text-left text-[11px] uppercase tracking-wider text-slate-500">
                  <tr><th className="px-5 py-2 font-medium">Model</th><th className="px-3 py-2 font-medium">State</th>
                    <th className="px-3 py-2 text-right font-medium">Val IoU</th><th className="px-3 py-2 text-right font-medium">Test IoU</th>
                    <th className="px-5 py-2 text-right font-medium">Bolivia IoU</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {(models.data?.models ?? []).map((m) => (
                    <tr key={m.name}>
                      <td className="px-5 py-2.5"><div className="font-medium text-slate-800">{modelLabel(m)}</div>
                        <div className="font-mono text-[11px] text-slate-500">{m.name}{m.name === models.data?.default_model && " · default"}</div></td>
                      <td className="px-3 py-2.5"><StateBadge state={m.training_state} /></td>
                      <td className="px-3 py-2.5 text-right tabular-nums">{fmtPct(m.best_val_iou)}</td>
                      <td className="px-3 py-2.5 text-right tabular-nums">{fmtPct(m.test_iou)}</td>
                      <td className="px-5 py-2.5 text-right tabular-nums">{fmtPct(m.bolivia_iou)}</td>
                    </tr>
                  ))}
                  {models.data && models.data.models.length === 0 && (
                    <tr><td colSpan={5} className="px-5 py-6 text-center text-slate-500">No trained checkpoints found in artifacts/runs.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <div className="min-w-0 space-y-4 xl:col-span-2">
          <Card title="What the system does">
            <ul className="space-y-2.5 text-[13px] leading-relaxed text-slate-700">
              <li className="flex gap-2"><ScanLine className="mt-0.5 h-4 w-4 shrink-0 text-water-600" />Segments water in Sentinel-1 SAR, Sentinel-2 optical or RGB imagery with U-Net / DeepLabV3+ models trained on this laptop's RTX 3050.</li>
              <li className="flex gap-2"><Map className="mt-0.5 h-4 w-4 shrink-0 text-water-600" />Samples georeferenced predictions along OSM road geometry and runs Dijkstra on flood-penalised edge costs.</li>
              <li className="flex gap-2"><Database className="mt-0.5 h-4 w-4 shrink-0 text-water-600" />Reports held-out metrics against threshold baselines (NDWI, Otsu) honestly — including where baselines are competitive.</li>
            </ul>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link to="/segmentation" className="inline-flex items-center gap-1 rounded-lg bg-water-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-water-700">Segment an image <ArrowRight className="h-3 w-3" /></Link>
              <Link to="/routes" className="inline-flex items-center gap-1 rounded-lg bg-white px-3 py-1.5 text-xs font-medium text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50">Plan a route <ArrowRight className="h-3 w-3" /></Link>
            </div>
          </Card>
          <Callout tone="warn" title="Limitations">
            Predictions describe water visible to the satellite on the acquisition date, not depth or road passability.
            Roads are today's OSM network, imagery is from 2016–2019 flood events. Arbitrary RGB photos carry no
            coordinates and cannot be routed. Never use this prototype in place of official emergency guidance.
          </Callout>
        </div>
      </div>
    </div>
  );
}
