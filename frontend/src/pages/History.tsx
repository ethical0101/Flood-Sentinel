import { useState } from "react";
import { Download, FileText, GitCompare, Map, ScanLine, Search } from "lucide-react";
import { api, historyExportUrl, reportUrl, resultFile, routeExportUrl, type HistoryItem, type PlanResponse, type SegResult } from "../api";
import { Badge, Button, DataTable, Drawer, Empty, ErrorState, KV, Notice, PageHead, Panel, Segmented, Skeleton, TextInput } from "../components/ui";
import { fmtDist, fmtMs, fmtNum, fmtPct, fmtTime, useApi } from "../lib";

const PAGE = 50;

export default function History() {
  const [kind, setKind] = useState<"" | "segmentation" | "route">("");
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const list = useApi(() => api.history({ kind: kind || undefined, q: query || undefined, limit: PAGE, offset: page * PAGE }), [kind, query, page], 20000);
  const detail = useApi(() => (open ? api.historyItem(open) : Promise.resolve(null)), [open]);
  const cmpA = useApi(() => (selected.length === 2 ? api.historyItem(selected[0]) : Promise.resolve(null)), [selected.join()]);
  const cmpB = useApi(() => (selected.length === 2 ? api.historyItem(selected[1]) : Promise.resolve(null)), [selected.join()]);
  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s.slice(-1), id]));
  const items = list.data?.items ?? [];

  return (
    <div>
      <PageHead title="History & reports"
        sub="Every segmentation and route analysis is stored in a local SQLite database (metadata and file references only; images stay on disk). Select two analyses to compare them, or export a report."
        actions={<>
          <Button variant="secondary" href={historyExportUrl("csv", kind || undefined)}><Download className="h-3.5 w-3.5" />CSV</Button>
          <Button variant="secondary" href={historyExportUrl("json", kind || undefined)}><Download className="h-3.5 w-3.5" />JSON</Button></>} />

      <Panel body="p-0" title={`Analyses · ${list.data?.total ?? 0}`} actions={<div className="flex flex-wrap items-center gap-2">
        <Segmented size="xs" label="Kind" value={kind} onChange={(v) => { setKind(v); setPage(0); }} options={[{ value: "", label: "All" }, { value: "segmentation", label: "Segmentation" }, { value: "route", label: "Routes" }]} />
        <form className="flex w-full max-w-56 gap-1" onSubmit={(e) => { e.preventDefault(); setQuery(q); setPage(0); }}>
          <TextInput label="Search history" value={q} onChange={setQ} placeholder="title, model or source" /><Button type="submit" size="sm" variant="secondary"><Search className="h-3.5 w-3.5" /></Button>
        </form></div>}>
        {list.error ? <div className="p-3"><ErrorState error={list.error} onRetry={list.reload} /></div> : !list.data ? <Skeleton className="m-3 h-40" /> : items.length === 0 ? (
          <Empty title="No analyses yet">Run a segmentation or plan a route; results are stored here automatically.</Empty>
        ) : (
          <DataTable rowKey={(r: HistoryItem) => r.id} rows={items} onRowClick={(r: HistoryItem) => setOpen(r.id)}
            columns={[
              { key: "sel", label: "Compare", value: () => "", render: (r: HistoryItem) => <input type="checkbox" aria-label={`Select ${r.title} for comparison`} checked={selected.includes(r.id)} onClick={(e) => e.stopPropagation()} onChange={() => toggle(r.id)} className="accent-[var(--color-accent)]" /> },
              { key: "k", label: "Kind", value: (r: HistoryItem) => r.kind, render: (r: HistoryItem) => r.kind === "segmentation" ? <Badge tone="water"><ScanLine className="h-3 w-3" />segmentation</Badge> : <Badge tone="accent"><Map className="h-3 w-3" />route</Badge> },
              { key: "t", label: "Title", value: (r: HistoryItem) => r.title, render: (r: HistoryItem) => <span className="font-medium">{r.title}</span> },
              { key: "m", label: "Model", value: (r: HistoryItem) => r.model ?? "", render: (r: HistoryItem) => <span className="font-mono text-[11px]">{r.model ?? (r.kind === "route" && r.source === "ground_truth" ? "hand label" : "default")}</span> },
              { key: "s", label: "Result", value: (r: HistoryItem) => summaryText(r) },
              { key: "g", label: "IoU vs label", numeric: true, value: (r: HistoryItem) => r.metrics?.iou_water ?? null },
              { key: "c", label: "Created", value: (r: HistoryItem) => r.created, render: (r: HistoryItem) => fmtTime(r.created) },
              { key: "a", label: "Files", value: (r: HistoryItem) => (r.available ? "available" : "pruned"), render: (r: HistoryItem) => r.available ? <Badge tone="ok">available</Badge> : <Badge title="Result folder removed by retention; metadata kept">pruned</Badge> },
              { key: "r", label: "", value: () => "", render: (r: HistoryItem) => r.available ? <a onClick={(e) => e.stopPropagation()} href={reportUrl(r.id)} className="inline-flex items-center gap-1 text-[11.5px] text-accent hover:underline"><FileText className="h-3 w-3" />Report</a> : null },
            ]} />
        )}
        {list.data && list.data.total > PAGE && (
          <div className="flex items-center justify-end gap-2 border-t border-line-2 px-3 py-2 text-[12px]">
            <Button size="xs" variant="secondary" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Previous</Button>
            <span className="tnum">{page * PAGE + 1}–{Math.min((page + 1) * PAGE, list.data.total)} of {list.data.total}</span>
            <Button size="xs" variant="secondary" disabled={(page + 1) * PAGE >= list.data.total} onClick={() => setPage((p) => p + 1)}>Next</Button>
          </div>
        )}
      </Panel>

      <Panel title={<span className="flex items-center gap-1.5"><GitCompare className="h-3.5 w-3.5" />Comparison</span>} className="mt-3">
        {selected.length < 2 ? <p className="text-[12px] text-ink-3">Tick two analyses above to compare them side by side ({selected.length}/2 selected).</p> :
          !cmpA.data || !cmpB.data ? <Skeleton className="h-40" /> : cmpA.data.kind !== cmpB.data.kind ? <Notice tone="warn">Pick two analyses of the same kind.</Notice> : (
            <div className="grid gap-3 md:grid-cols-2">{[cmpA.data, cmpB.data].map((it) => <CompareCard key={it.id} it={it} />)}</div>
          )}
      </Panel>

      <Drawer open={!!open} onClose={() => setOpen(null)} title="Analysis detail" wide>
        {detail.loading ? <Skeleton className="h-60" /> : detail.error ? <ErrorState error={detail.error} /> : detail.data && (
          <div className="space-y-3">
            <KV dense items={[["Id", <span className="font-mono text-[11px]">{detail.data.id}</span>], ["Kind", detail.data.kind], ["Title", detail.data.title], ["Created", fmtTime(detail.data.created)],
              ["Model", detail.data.model ?? "–"], ["Source", detail.data.source ?? "–"], ["App version", detail.data.app_version], ["Settings", <span className="font-mono text-[10.5px]">{JSON.stringify(detail.data.settings)}</span>]]} />
            {!detail.data.available && <Notice tone="warn">The result files were removed by the retention policy; only this metadata remains.</Notice>}
            {detail.data.available && detail.data.kind === "segmentation" && <SegDetail r={detail.data.detail as SegResult} />}
            {detail.data.available && detail.data.kind === "route" && <RouteDetail r={detail.data.detail as PlanResponse} />}
          </div>
        )}
      </Drawer>
    </div>
  );
}

function summaryText(r: HistoryItem) {
  const s = r.summary;
  if (r.kind === "segmentation") return `${fmtPct(s.water_fraction as number)} water${s.water_area_km2 != null ? ` · ${(s.water_area_km2 as number).toFixed(2)} km²` : ""}`;
  return `${String(s.status).replace(/_/g, " ")} · ${fmtDist(s.risk_aware_m as number)}, flooded ${fmtDist(s.risk_aware_flooded_m as number)}`;
}

function SegDetail({ r }: { r: SegResult }) {
  return (<>
    <div className="grid grid-cols-2 gap-2">
      <img src={resultFile(r.id, "input.png")} alt="Input" className="aspect-square w-full rounded bg-nav object-contain [image-rendering:pixelated]" />
      <img src={resultFile(r.id, "overlay.png")} alt="Prediction overlay" className="aspect-square w-full rounded bg-nav object-contain [image-rendering:pixelated]" />
    </div>
    <KV dense items={[["Water", fmtPct(r.stats.water_fraction, 2)], ["Area", r.area ? `${r.area.water_area_km2.toFixed(3)} km²` : "–"], ["Threshold", r.settings.threshold],
      ["IoU vs hand label", fmtNum(r.ground_truth?.metrics.iou_water)], ["NDWI agreement IoU", fmtNum(r.baseline?.agreement_iou_with_model)], ["Latency", fmtMs(r.timing.total_ms)]]} />
    <div className="flex flex-wrap gap-1.5">
      <Button size="xs" variant="secondary" href={resultFile(r.id, "mask.png", true)}><Download className="h-3 w-3" />Mask</Button>
      <Button size="xs" variant="secondary" href={resultFile(r.id, "probability.tif", true)}><Download className="h-3 w-3" />GeoTIFF</Button>
      <Button size="xs" variant="secondary" href={resultFile(r.id, "result.json", true)}><Download className="h-3 w-3" />JSON</Button>
      <Button size="xs" variant="dark" href={reportUrl(r.id)}><FileText className="h-3 w-3" />HTML report</Button>
    </div>
  </>);
}

function RouteDetail({ r }: { r: PlanResponse }) {
  return (<>
    <Notice tone={r.status === "flood_free_route_found" ? "ok" : "bad"}>{r.status_note}</Notice>
    <KV dense items={[["Area", r.area_name], ["Flood layer", r.flood_layer.is_model_prediction ? "model prediction" : "hand label (reference)"],
      ["Shortest", `${fmtDist(r.baseline.distance_m)} · ${fmtDist(r.baseline.flooded_length_m)} flooded`], ["Risk-aware", `${fmtDist(r.risk_aware.distance_m)} · ${fmtDist(r.risk_aware.flooded_length_m)} flooded`],
      ["Alternatives", r.alternatives.length], ["Penalty", `${r.settings.flood_penalty}×`]]} />
    <ul className="list-disc space-y-1 pl-4 text-[12px]">{r.explanation.map((e) => <li key={e}>{e}</li>)}</ul>
    <div className="flex flex-wrap gap-1.5">
      <Button size="xs" variant="secondary" href={routeExportUrl(r.id, "json")}><Download className="h-3 w-3" />JSON</Button>
      <Button size="xs" variant="secondary" href={routeExportUrl(r.id, "geojson")}><Download className="h-3 w-3" />GeoJSON</Button>
      <Button size="xs" variant="dark" href={routeExportUrl(r.id, "html")}><FileText className="h-3 w-3" />HTML report</Button>
    </div>
  </>);
}

function CompareCard({ it }: { it: HistoryItem }) {
  const s = it.summary;
  return (
    <div className="rounded-md border border-line p-2.5">
      <div className="mb-1.5 flex items-center justify-between gap-2"><span className="truncate text-[12.5px] font-semibold">{it.title}</span><span className="text-[11px] text-ink-3">{fmtTime(it.created)}</span></div>
      {it.kind === "segmentation" && it.available && <img src={resultFile(it.id, "overlay.png")} alt={`Overlay ${it.title}`} className="mb-2 aspect-square w-full rounded bg-nav object-contain [image-rendering:pixelated]" />}
      <KV dense items={it.kind === "segmentation" ? [["Model", it.model ?? "–"], ["Water", fmtPct(s.water_fraction as number, 2)], ["Area", s.water_area_km2 != null ? `${(s.water_area_km2 as number).toFixed(3)} km²` : "–"],
        ["IoU vs label", fmtNum(it.metrics?.iou_water)], ["NDWI agreement", fmtNum(s.ndwi_agreement_iou as number | null)], ["Latency", fmtMs(s.total_ms as number)], ["Threshold", String(it.settings?.threshold ?? "–")]]
        : [["Status", String(s.status).replace(/_/g, " ")], ["Shortest", `${fmtDist(s.baseline_m as number)} · ${fmtDist(s.baseline_flooded_m as number)} flooded`],
          ["Risk-aware", `${fmtDist(s.risk_aware_m as number)} · ${fmtDist(s.risk_aware_flooded_m as number)} flooded`], ["Flood layer", it.source ?? "–"], ["Penalty", String(it.settings?.flood_penalty ?? "–")]]} />
    </div>
  );
}
