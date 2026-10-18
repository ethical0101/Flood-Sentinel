import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import { CheckCircle2, Database, ExternalLink, XCircle } from "lucide-react";
import { api } from "../api";
import { Badge, DataTable, ErrorState, KV, Notice, PageHead, Panel, Skeleton } from "../components/ui";
import { fmtBytes, fmtInt, fmtPct, fmtTime, useApi } from "../lib";

const STORAGE_LABEL: Record<string, [string, string]> = {
  raw_sen1floods11: ["Raw · Sen1Floods11 GeoTIFFs", "#0e7490"],
  raw_worldfloods_test: ["Raw · WorldFloods v2 test", "#7c3aed"],
  processed_chips: ["Processed · per-chip .npz cache", "#38bdf8"],
  checkpoints: ["Model checkpoints", "#f59e0b"],
  osm_cache: ["Cache · OSM road graphs", "#16a34a"],
  analysis_outputs: ["Analysis outputs (predictions, routes, history)", "#94a3b8"],
};
const SPLIT_COLOR: Record<string, string> = { train: "#0e7490", valid: "#64748b", test: "#2563eb", bolivia: "#f59e0b" };

export default function Datasets() {
  const d = useApi(() => api.datasets(), []);
  if (d.error) return <ErrorState error={d.error} onRetry={d.reload} />;
  if (!d.data) return <div className="space-y-3"><Skeleton className="h-8 w-80" /><Skeleton className="h-60" /></div>;
  const s = d.data;
  const v = s.verification;
  const storageTotal = Object.values(s.storage).reduce((a, b) => a + b, 0);

  return (
    <div>
      <PageHead title="Dataset explorer"
        sub="Provenance, licences, integrity, splits, channels, labels and storage for every dataset this project uses. Training uses Sen1Floods11 only; WorldFloods v2 is used only for evaluation." />

      <div className="grid gap-3 xl:grid-cols-2">
        <Panel title={<span className="flex items-center gap-1.5"><Database className="h-3.5 w-3.5" />Sen1Floods11 v1.1 · hand-labelled</span>}
          actions={v?.passed ? <Badge tone="ok"><CheckCircle2 className="h-3 w-3" />verified ({v.integrity_mode})</Badge> : <Badge tone="bad"><XCircle className="h-3 w-3" />not verified</Badge>}>
          <KV dense items={[["Role", <Badge tone="water">training · validation · test · Bolivia</Badge>], ["Source", s.source],
            ["Downloaded", fmtTime(s.downloaded_utc)], ["Files", `${fmtInt(s.n_files)} · ${fmtBytes(s.download_bytes)}`],
            ["Integrity", v ? `${v.files_ok}/${v.files_checked} files match published MD5` : "–"], ["CRS", v ? Object.keys(v.crs).join(", ") : "–"],
            ["Unique chips", v?.unique_chips ?? "–"], ["Resolution", "512 × 512 px at 10 m"]]} />
          <p className="mt-2 text-[11.5px] leading-relaxed text-ink-2"><b>Licence:</b> {s.license}</p>
          <p className="mt-1 text-[11.5px] leading-relaxed text-ink-3">{s.citation}</p>
        </Panel>
        <Panel title={<span className="flex items-center gap-1.5"><Database className="h-3.5 w-3.5" />WorldFloods v2 · test split</span>}
          actions={s.external ? (s.external.n_failed === 0 ? <Badge tone="ok"><CheckCircle2 className="h-3 w-3" />SHA-256 verified</Badge> : <Badge tone="bad">{s.external.n_failed} failed</Badge>) : <Badge tone="warn">not downloaded</Badge>}>
          {!s.external ? <p className="text-[12px] text-ink-2">Optional external evaluation set. Download with <code>scripts/download_worldfloods_test.py</code> (3.1 GB).</p> : (<>
            <KV dense items={[["Role", <Badge tone="truth">external evaluation only</Badge>], ["Scenes", s.external.scenes], ["Files", `${s.external.n_files} · ${fmtBytes(s.external.bytes)}`],
              ["Downloaded", fmtTime(s.external.downloaded_utc)], ["Integrity", s.external.integrity], ["Channels", s.external.channels], ["Labels", s.external.labels]]} />
            <p className="mt-2 text-[11.5px] leading-relaxed text-ink-2"><b>Licence:</b> {s.external.licence} (non-commercial; attribution required).</p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-ink-3">{s.external.citation}
              <a href="https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2" target="_blank" rel="noreferrer" className="ml-1 inline-flex items-center gap-0.5 text-accent hover:underline">source<ExternalLink className="h-3 w-3" /></a></p>
          </>)}
        </Panel>
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-3">
        <Panel title="Splits" body="p-0" info="Official Sen1Floods11 split lists, used unchanged. Bolivia is held out entirely.">
          <table className="w-full text-[12px] tnum">
            <thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Split</th><th className="px-3 py-1.5 text-right">Chips</th><th className="px-3 py-1.5 text-right">Events</th><th className="px-3 py-1.5 text-right">Mean water</th><th className="px-3 py-1.5 text-left">Use</th></tr></thead>
            <tbody className="divide-y divide-line-2">{["train", "valid", "test", "bolivia"].filter((k) => s.splits[k]).map((k) => (
              <tr key={k}><td className="px-3 py-1.5"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: SPLIT_COLOR[k] }} />{k}</td>
                <td className="px-3 py-1.5 text-right">{s.splits[k].chips}</td><td className="px-3 py-1.5 text-right">{s.splits[k].events.length}</td>
                <td className="px-3 py-1.5 text-right">{fmtPct(s.splits[k].mean_water_fraction)}</td>
                <td className="px-3 py-1.5 text-[11px] text-ink-2">{{ train: "weights, normalisation", valid: "checkpoint + threshold selection", test: "final evaluation", bolivia: "final evaluation (unseen event)" }[k]}</td></tr>))}</tbody>
          </table>
          {v && <div className="border-t border-line-2 px-3 py-2 text-[11px] text-ink-3">Label pixels: {Object.entries(v.label_pixel_counts).map(([k, n]) => `${{ "-1": "no data", "0": "not water", "1": "water" }[k] ?? k} ${(n / 1e6).toFixed(1)} M`).join(" · ")}. Water share of valid training pixels: {fmtPct(s.train_water_fraction)}.</div>}
        </Panel>
        <Panel title="Label definitions" body="p-0 overflow-x-auto">
          <table className="w-full text-[12px]"><thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Value</th><th className="px-3 py-1.5 text-left">Meaning</th><th className="px-3 py-1.5 text-left">Use</th></tr></thead>
            <tbody className="divide-y divide-line-2">{s.label_definitions.map((l) => <tr key={l.value}><td className="px-3 py-1.5 font-mono">{l.value}</td><td className="px-3 py-1.5">{l.meaning}</td><td className="px-3 py-1.5 text-ink-2">{l.use}</td></tr>)}</tbody></table>
        </Panel>
        <Panel title="Storage" info="Measured on disk (cached for 5 minutes).">
          <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-100">
            {Object.entries(s.storage).map(([k, b]) => b > 0 && <div key={k} title={`${STORAGE_LABEL[k]?.[0] ?? k}: ${fmtBytes(b)}`} style={{ width: `${(100 * b) / storageTotal}%`, background: STORAGE_LABEL[k]?.[1] ?? "#cbd5e1" }} />)}
          </div>
          <ul className="mt-2 space-y-1 text-[12px]">{Object.entries(s.storage).map(([k, b]) => (
            <li key={k} className="flex items-center justify-between gap-2"><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: STORAGE_LABEL[k]?.[1] }} />{STORAGE_LABEL[k]?.[0] ?? k}</span><span className="tnum font-medium">{fmtBytes(b)}</span></li>))}
            <li className="flex justify-between border-t border-line-2 pt-1 font-semibold"><span>Total</span><span className="tnum">{fmtBytes(storageTotal)}</span></li></ul>
        </Panel>
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-5">
        <Panel title="Geographic & temporal coverage · Sen1Floods11 events" className="xl:col-span-3" body="p-0 overflow-x-auto">
          <div className="h-[300px]">
            <MapContainer center={[15, 20]} zoom={2} className="h-full w-full" scrollWheelZoom={false} worldCopyJump>
              <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
              {s.events.map((e) => {
                const n = Object.values(e.chips).reduce((a, b) => a + b, 0);
                const color = e.chips.bolivia ? SPLIT_COLOR.bolivia : SPLIT_COLOR.train;
                return <CircleMarker key={e.event} center={[e.lat, e.lon]} radius={4 + Math.sqrt(n)} pathOptions={{ color: "#fff", weight: 1.5, fillColor: color, fillOpacity: 0.9 }}>
                  <Tooltip>{e.event}: {n} chips · {e.s2_date ?? "date n/a"}</Tooltip></CircleMarker>;
              })}
            </MapContainer>
          </div>
          <div className="border-t border-line-2 px-3 py-1.5 text-[11px] text-ink-3">Marker size ∝ √chips. Amber = held-out Bolivia event. Basemap © OpenStreetMap contributors (needs internet).</div>
        </Panel>
        <Panel title="Events" className="xl:col-span-2" body="p-0 overflow-x-auto">
          <DataTable rowKey={(e) => e.event} rows={s.events} csvName="sen1floods11_events" maxHeight={330}
            columns={[{ key: "e", label: "Event", value: (e) => e.event }, { key: "d", label: "S2 date", value: (e) => e.s2_date ?? "" },
              { key: "tr", label: "Train", numeric: true, value: (e) => e.chips.train ?? 0 }, { key: "va", label: "Val", numeric: true, value: (e) => e.chips.valid ?? 0 },
              { key: "te", label: "Test", numeric: true, value: (e) => (e.chips.test ?? 0) + (e.chips.bolivia ?? 0) }, { key: "w", label: "Water", numeric: true, value: (e) => e.mean_water_fraction, render: (e) => fmtPct(e.mean_water_fraction) }]} />
        </Panel>
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-5">
        <Panel title="Channels" className="xl:col-span-3" body="p-0 overflow-x-auto">
          <DataTable rowKey={(c) => c.name} rows={s.channels} maxHeight={320}
            columns={[{ key: "n", label: "Channel", value: (c) => c.name, render: (c) => <span className="font-mono">{c.name}</span> }, { key: "s", label: "Sensor", value: (c) => c.sensor },
              { key: "u", label: "Unit", value: (c) => c.unit }, { key: "p", label: "Preprocessing", value: (c) => c.note }]} />
        </Panel>
        <Panel title="Data-quality warnings" className="xl:col-span-2">
          <ul className="space-y-2">{s.quality_warnings.map((w) => <li key={w}><Notice tone="warn">{w}</Notice></li>)}</ul>
          <p className="mt-2 text-[11px] text-ink-3">From the upgrade audit (docs/upgrade_audit.md, artifacts/audit/).</p>
        </Panel>
      </div>
    </div>
  );
}
