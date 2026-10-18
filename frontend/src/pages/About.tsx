import type { ReactNode } from "react";
import { ArrowRight, CheckCircle2, CircleAlert, RefreshCw, WifiOff, XCircle } from "lucide-react";
import { api } from "../api";
import { Badge, Button, ErrorState, KV, Notice, PageHead, Panel, Skeleton } from "../components/ui";
import { fmtTime, useApi } from "../lib";

function Box({ title, children, tone = "slate" }: { title: string; children: ReactNode; tone?: "slate" | "water" | "ok" | "warn" }) {
  const c = { slate: "border-line bg-white", water: "border-blue-200 bg-water-soft", ok: "border-green-200 bg-ok-soft", warn: "border-amber-200 bg-warn-soft" }[tone];
  return <div className={`rounded-md border px-2.5 py-2 ${c}`}><div className="text-[12px] font-semibold text-ink">{title}</div><div className="mt-0.5 text-[11px] leading-snug text-ink-2">{children}</div></div>;
}
const Arrow = () => <div className="hidden items-center justify-center text-ink-3 lg:flex"><ArrowRight className="h-4 w-4" /></div>;

export default function About() {
  const diag = useApi(() => api.diagnostics(), []);
  const formats = useApi(() => api.formats(), []);
  const d = diag.data;

  return (
    <div className="max-w-[1400px]">
      <PageHead title="Documentation, diagnostics & settings"
        sub="Architecture, methodology, supported inputs, offline behaviour, versions and a live system check. The full documentation is in README.md and docs/."
        actions={<Button variant="secondary" onClick={diag.reload} disabled={diag.loading}><RefreshCw className={`h-3.5 w-3.5 ${diag.loading ? "animate-spin" : ""}`} />Run diagnostics</Button>} />

      <Notice tone="bad" title="Safety disclaimer">This is a research prototype. Flood extents are statistical model estimates and the routes are not verified.
        Never use it to make real evacuation or navigation decisions; follow official emergency services and verified road-closure information.</Notice>

      <div className="mt-3 grid gap-3 xl:grid-cols-3">
        <Panel title="System diagnostics" className="xl:col-span-2" body="p-0"
          actions={d && <span className="flex gap-1"><Badge tone="ok">{d.summary.ok} ok</Badge>{d.summary.warn > 0 && <Badge tone="warn">{d.summary.warn} warn</Badge>}{d.summary.fail > 0 && <Badge tone="bad">{d.summary.fail} fail</Badge>}</span>}
          footer={d ? <>Checked {fmtTime(d.checked_at)} in {d.duration_ms} ms</> : undefined}>
          {diag.error ? <div className="p-3"><ErrorState error={diag.error} onRetry={diag.reload} /></div> : !d ? <Skeleton className="m-3 h-60" /> : (
            <table className="w-full text-[12px]">
              <tbody className="divide-y divide-line-2">{d.checks.map((c) => (
                <tr key={c.check}>
                  <td className="w-7 py-1.5 pl-3">{c.status === "ok" ? <CheckCircle2 className="h-4 w-4 text-ok" aria-label="ok" /> : c.status === "warn" ? <CircleAlert className="h-4 w-4 text-warn" aria-label="warning" /> : <XCircle className="h-4 w-4 text-bad" aria-label="failed" />}</td>
                  <td className="py-1.5 pr-2 font-medium">{c.check}</td>
                  <td className="py-1.5 pr-2 text-ink-2">{c.detail}</td>
                  <td className="py-1.5 pr-3 text-right text-[10.5px] uppercase tracking-wide text-ink-3">{c.category}</td>
                </tr>))}</tbody>
            </table>
          )}
        </Panel>
        <div className="min-w-0 space-y-3">
          <Panel title="Versions">
            {!d ? <Skeleton className="h-40" /> : <KV dense items={Object.entries(d.versions).map(([k, v]) => [k, <span className="font-mono text-[11px]">{v ?? "–"}</span>])} />}
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><WifiOff className="h-3.5 w-3.5" />Offline capability</span>}>
            {!d ? <Skeleton className="h-24" /> : (<>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-ok">Works offline</div>
              <ul className="mb-2 mt-1 list-disc space-y-0.5 pl-4 text-[12px]">{d.offline.works_offline.map((x) => <li key={x}>{x}</li>)}</ul>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-warn">Needs network</div>
              <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[12px]">{d.offline.needs_network.map((x) => <li key={x}>{x}</li>)}</ul>
            </>)}
          </Panel>
        </div>
      </div>

      <Panel title="Architecture" className="mt-3">
        <div className="grid items-stretch gap-2 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          <Box title="Sentinel-1/2 imagery">Sen1Floods11 chips (EPSG:4326, 10 m), user GeoTIFF / RGB, WorldFloods scenes (UTM)</Box><Arrow />
          <Box title="Preprocessing" tone="water">Band standardisation (train statistics); optional NDWI/MNDWI channels; 512² tiles with blended overlap</Box><Arrow />
          <Box title="Segmentation CNN" tone="water">U-Net/ResNet-34 or DeepLabV3+/MobileNetV3, trained locally with AMP; validation-selected threshold</Box><Arrow />
          <Box title="Georeferenced P(water)" tone="water">Mask, overlay, uncertainty, NDWI comparison, area (geodesic), GeoTIFF export</Box>
        </div>
        <div className="mt-2 grid items-stretch gap-2 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          <Box title="OpenStreetMap roads" tone="ok">OSMnx drive graph per area, cached as GraphML (offline after the first fetch)</Box><Arrow />
          <Box title="Edge risk scoring" tone="ok">Samples every 5 m along the road, CRS-aware, optional buffer; unknown ≠ dry</Box><Arrow />
          <Box title="Dijkstra routing" tone="ok">Shortest vs flood-penalised cost, alternatives, explicit status, optional closures</Box><Arrow />
          <Box title="FastAPI + SQLite + React" tone="warn">Models loaded once; serialised GPU inference; history/reports; guarded training control</Box>
        </div>
        <p className="mt-2 text-[11.5px] text-ink-3">API reference: <a className="text-accent hover:underline" href="/docs" target="_blank" rel="noreferrer">/docs</a> (OpenAPI). Training runs as a separate, resumable, GPU-courteous process.</p>
      </Panel>

      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <Panel title="Research methodology">
          <ul className="list-disc space-y-1.5 pl-4 text-[12.5px] leading-relaxed">
            <li><b>Task:</b> binary segmentation, positive class = water (flood and permanent). Label −1 is ignored everywhere.</li>
            <li><b>Data:</b> Sen1Floods11 official splits, used unchanged. Bolivia is held out. WorldFloods v2 test split is external evaluation only.</li>
            <li><b>Selection:</b> the checkpoint and decision threshold are chosen on validation only. Held-out sets are evaluated after selection.</li>
            <li><b>v2 experiments:</b> controlled, one factor at a time (radiometric augmentation, index channels, both, S2-only), with a second-seed run to measure noise.</li>
            <li><b>Baselines:</b> NDWI &gt; 0, MNDWI &gt; 0, Sentinel-1 Otsu, JRC permanent water, and a constant predictor.</li>
            <li><b>Routing evaluation:</b> 300 random origin–destination pairs; routes planned on the model are scored against the hand labels.</li>
          </ul>
        </Panel>
        <Panel title="Supported input formats" body="p-0 overflow-x-auto">
          {!formats.data ? <Skeleton className="m-3 h-32" /> : (<>
            <table className="w-full text-[12px]"><thead className="bg-slate-50 text-[10.5px] uppercase text-ink-3"><tr><th className="px-3 py-1.5 text-left">Format</th><th className="px-3 py-1.5 text-left">Interpretation</th><th className="px-3 py-1.5 text-left">Coordinates</th></tr></thead>
              <tbody className="divide-y divide-line-2">{formats.data.formats.map((f) => <tr key={f.format}><td className="px-3 py-1.5 font-medium">{f.format}</td><td className="px-3 py-1.5 text-ink-2">{f.interpretation}</td><td className="px-3 py-1.5">{f.georeferenced}</td></tr>)}</tbody></table>
            <div className="border-t border-line-2 px-3 py-2 text-[11.5px] text-ink-3">Limits: {formats.data.max_upload_mb} MB per file, {(formats.data.max_pixels / 1e6).toFixed(1)} MP, at least {formats.data.min_side_px} px per side.</div>
          </>)}
        </Panel>
        <Panel title="How to run">
          <pre className="overflow-x-auto rounded-md bg-nav p-3 font-mono text-[11px] leading-relaxed text-slate-200 scroll-thin scroll-dark">{`# backend (serves the built dashboard too)
.venv\\Scripts\\python -m uvicorn backend.app.main:app --port 8000
# frontend development server (proxies /api)
cd frontend && npm run dev
# data and evaluation
.venv\\Scripts\\python scripts\\download_datasets.py
.venv\\Scripts\\python scripts\\download_worldfloods_test.py
.venv\\Scripts\\python -m training.prepare_data
.venv\\Scripts\\python scripts\\run_experiments.py --suite v2 --polite
.venv\\Scripts\\python -m training.evaluate --all
.venv\\Scripts\\python scripts\\evaluate_worldfloods.py`}</pre>
        </Panel>
        <Panel title="Sources, licences & citations">
          <ul className="space-y-2 text-[12px] leading-relaxed">
            <li><b>Sen1Floods11</b>: Bonafilia et al. (2020), CVPR Workshops. No explicit licence published; used non-commercially and not redistributed.</li>
            <li><b>WorldFloods v2</b>: Portalés-Julià et al. (2023), <i>Scientific Reports</i> 13, 20316. CC BY-NC 4.0, test split only.</li>
            <li><b>Copernicus Sentinel-1/2</b> data (ESA), as contained in the datasets.</li>
            <li><b>OpenStreetMap</b> © contributors, ODbL 1.0, via OSMnx (Boeing, 2017).</li>
            <li><b>Indices:</b> NDWI (McFeeters, 1996); MNDWI (Xu, 2006).</li>
          </ul>
        </Panel>
      </div>

      <Panel title="Known limitations" className="mt-3">
        <ul className="grid list-disc gap-x-8 gap-y-1 pl-4 text-[12.5px] leading-relaxed md:grid-cols-2">
          <li>Labels mark all surface water, not flood specifically; permanent rivers count as "flooded" for routing.</li>
          <li>A 10 m pixel is about the width of a road; narrow streets and bridges are unreliable.</li>
          <li>The Sen1Floods11 test split is spatially close to training, so generalisation claims rest on Bolivia (15 chips) and WorldFloods (S2 only).</li>
          <li>Imagery is from 2016–2019; roads are current OSM. Travel times are free-flow.</li>
          <li>Optical bands fail under cloud; SAR misses water in vegetation and confuses smooth surfaces with water.</li>
          <li>Routing is limited to each ≈5 × 5 km raster footprint.</li>
        </ul>
      </Panel>
    </div>
  );
}
