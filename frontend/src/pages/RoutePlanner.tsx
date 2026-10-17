import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CircleMarker, GeoJSON, ImageOverlay, MapContainer, Pane, Polyline, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import type { LatLngBoundsExpression, LeafletMouseEvent, PathOptions } from "leaflet";
import { Flag, Info, MapPin, Navigation, Route as RouteIcon, ShieldCheck } from "lucide-react";
import { api, imageryUrl, overlayUrl, type Area, type AreaDetail, type PlanResponse, type RoadFeature, type Route } from "../api";
import { Badge, Button, Callout, Card, ErrorState, PageHeader, Select, Slider, Spinner, Toggle } from "../components/ui";
import { RISK_COLORS, RISK_LABEL, fmtDist, fmtDuration, fmtPct, modelLabel, useApi } from "../lib";

type Pt = { lat: number; lon: number };
const ROUTE_STYLE = {
  baseline: { color: "#0f172a", weight: 5, dashArray: "8 7", opacity: 0.85 },
  risk: { color: "#059669", weight: 7, opacity: 0.95 },
  alt: ["#7c3aed", "#db2777", "#0891b2"],
};

function FitBounds({ bounds }: { bounds: LatLngBoundsExpression | null }) {
  const map = useMap();
  useEffect(() => { if (bounds) map.fitBounds(bounds, { padding: [16, 16] }); }, [bounds, map]);
  return null;
}

function ClickHandler({ onClick }: { onClick: (p: Pt) => void }) {
  useMapEvents({ click: (e: LeafletMouseEvent) => onClick({ lat: e.latlng.lat, lon: e.latlng.lng }) });
  return null;
}

function RouteCard({ title, r, color, dashed, base, badge }: { title: string; r: Route; color: string; dashed?: boolean; base?: Route; badge?: React.ReactNode }) {
  const d = base ? r.distance_m - base.distance_m : 0;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-slate-900">
          <span className="inline-block h-1 w-7 rounded" style={{ background: dashed ? `repeating-linear-gradient(90deg, ${color} 0 6px, transparent 6px 11px)` : color }} />
          {title}
        </div>
        {badge}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2.5 text-[13px]">
        <div><div className="text-[11px] text-slate-500">Distance</div><div className="font-semibold tabular-nums">{fmtDist(r.distance_m)}
          {base && <span className={`ml-1 text-xs font-normal ${d > 0 ? "text-amber-700" : "text-emerald-700"}`}>({d < 0 ? "−" : "+"}{fmtDist(Math.abs(d))})</span>}</div></div>
        <div><div className="text-[11px] text-slate-500">Est. drive time*</div><div className="font-semibold tabular-nums">{fmtDuration(r.travel_time_s)}</div></div>
        <div><div className="text-[11px] text-slate-500">Predicted-flooded road</div>
          <div className={`font-semibold tabular-nums ${r.flooded_length_m > 0 ? "text-red-700" : "text-emerald-700"}`}>{fmtDist(r.flooded_length_m)} <span className="text-xs font-normal text-slate-500">({fmtPct(r.flooded_share)})</span></div></div>
        <div><div className="text-[11px] text-slate-500">Max P(water) on route</div><div className="font-semibold tabular-nums">{r.max_flood_prob.toFixed(2)}</div></div>
        <div><div className="text-[11px] text-slate-500">Flood-affected segments</div><div className="font-semibold tabular-nums">{r.n_flood_affected_segments} / {r.n_segments}</div></div>
        <div><div className="text-[11px] text-slate-500">Without flood data</div><div className="font-semibold tabular-nums">{fmtDist(r.unknown_length_m)}</div></div>
      </div>
    </div>
  );
}

export default function RoutePlanner() {
  const [params] = useSearchParams();
  const areas = useApi(() => api.areas(), []);
  const models = useApi(() => api.modelStatus(), []);
  const [extraArea, setExtraArea] = useState<Area | null>(null);
  const [areaId, setAreaId] = useState("");
  const [source, setSource] = useState<"model" | "ground_truth">("model");
  const [model, setModel] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [bufferPx, setBufferPx] = useState(1);
  const [penalty, setPenalty] = useState(10);
  const [unknownPenalty, setUnknownPenalty] = useState(0.5);
  const [closeFlooded, setCloseFlooded] = useState(false);
  const [closureM, setClosureM] = useState(20);
  const [alts, setAlts] = useState(2);
  const [start, setStart] = useState<Pt | null>(null);
  const [end, setEnd] = useState<Pt | null>(null);
  const [clickMode, setClickMode] = useState<"start" | "end">("start");
  const [layers, setLayers] = useState({ basemap: true, imagery: false, flood: true, roads: true });
  const [detail, setDetail] = useState<AreaDetail | null>(null);
  const [detailErr, setDetailErr] = useState<Error | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [planErr, setPlanErr] = useState<Error | null>(null);
  const [planning, setPlanning] = useState(false);

  const allAreas = useMemo(() => [...(areas.data?.areas ?? []), ...(extraArea ? [extraArea] : [])], [areas.data, extraArea]);
  const area = allAreas.find((a) => a.id === areaId);

  // initial selection from query string: ?area=, ?chip=, ?result=
  useEffect(() => {
    if (!areas.data || areaId) return;
    const chip = params.get("chip");
    const result = params.get("result");
    const qArea = params.get("area");
    const curated = areas.data.areas.find((a) => a.id === qArea || (chip && a.chip_id === chip));
    if (curated) { setAreaId(curated.id); return; }
    const rid = result ? `r_${result}` : qArea?.startsWith("r_") ? qArea : null;
    if (rid) {
      api.area(rid, { source: "model", threshold: 0.5, buffer_px: 1 })
        .then((d) => { setExtraArea(d.area); setAreaId(rid); })
        .catch((e) => { setDetailErr(e); setAreaId(areas.data!.areas[0]?.id ?? ""); });
      return;
    }
    setAreaId(areas.data.areas[0]?.id ?? "");
  }, [areas.data, areaId, params]);

  // reset points when area changes
  useEffect(() => {
    if (!area) return;
    setPlan(null);
    setPlanErr(null);
    setStart(area.default_start ? { lat: area.default_start[0], lon: area.default_start[1] } : null);
    setEnd(area.default_end ? { lat: area.default_end[0], lon: area.default_end[1] } : null);
    if (!area.sources.includes(source)) setSource("model");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [areaId]);

  // load annotated road network
  useEffect(() => {
    if (!areaId) return;
    let cancelled = false;
    setDetailLoading(true);
    setDetailErr(null);
    api.area(areaId, { source, threshold, buffer_px: bufferPx, model: model || undefined })
      .then((d) => { if (!cancelled) setDetail(d); })
      .catch((e) => { if (!cancelled) { setDetail(null); setDetailErr(e); } })
      .finally(() => { if (!cancelled) setDetailLoading(false); });
    return () => { cancelled = true; };
  }, [areaId, source, threshold, bufferPx, model]);

  const runPlan = async () => {
    if (!start || !end || !areaId) return;
    setPlanning(true);
    setPlanErr(null);
    try {
      setPlan(await api.plan({
        area_id: areaId, source, model: model || null, start, end, threshold, buffer_px: bufferPx,
        flood_penalty: penalty, unknown_penalty: unknownPenalty, close_flooded: closeFlooded,
        closure_min_flooded_m: closureM, alternatives: alts,
      }));
    } catch (e) {
      setPlan(null);
      setPlanErr(e as Error);
    } finally {
      setPlanning(false);
    }
  };

  // automatically re-plan when settings change after a first plan
  useEffect(() => {
    if (plan && start && end) runPlan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, threshold, bufferPx, penalty, unknownPenalty, closeFlooded, closureM, alts, model]);

  const bounds: LatLngBoundsExpression | null = area ? [[area.bounds[1], area.bounds[0]], [area.bounds[3], area.bounds[2]]] : null;
  const layerBounds: LatLngBoundsExpression | null = detail
    ? [[detail.layer.bounds_wgs84[1], detail.layer.bounds_wgs84[0]], [detail.layer.bounds_wgs84[3], detail.layer.bounds_wgs84[2]]] : bounds;
  const roadStyle = (f?: RoadFeature): PathOptions => ({
    color: RISK_COLORS[f?.properties.risk_class ?? "unknown"], weight: f?.properties.risk_class === "clear" ? 2 : 3.5, opacity: 0.9,
  });
  const completedModels = (models.data?.models ?? []).filter((m) => m.training_state === "completed");
  const isPrediction = source === "model";

  return (
    <div>
      <PageHeader eyebrow="Capability B" title="Flood-risk-aware route planner">
        Roads are scored against a georeferenced flood raster, then Dijkstra's algorithm compares the shortest route
        with one that penalises predicted-flooded road. Click the map to place the start and destination.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
        <div className="min-w-0 space-y-4">
          <Card title="Study area & flood layer" bodyClass="space-y-3.5 p-4">
            {areas.error ? <ErrorState error={areas.error} onRetry={areas.reload} /> : (
              <Select label="Study area" value={areaId} onChange={setAreaId}>
                {allAreas.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
              </Select>
            )}
            {area && (
              <div className="rounded-lg bg-slate-50 p-3 text-xs leading-relaxed text-slate-600">
                <div className="font-medium text-slate-800">{area.region ?? area.description}</div>
                {area.event && <div>{area.event}</div>}
                {area.chip_id && <div>Chip <span className="font-mono">{area.chip_id}</span> · <b>{area.split}</b> split (unseen in training)</div>}
                {area.road_meta && <div>OSM graph: {area.road_meta.nodes} nodes, {area.road_meta.edges} edges · fetched {area.road_meta.fetched_utc.slice(0, 10)}</div>}
              </div>
            )}
            <div>
              <div className="mb-1 text-[13px] font-medium text-slate-800">Flood layer</div>
              <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1 text-xs font-medium">
                <button onClick={() => setSource("model")} className={`rounded-md py-1.5 ${source === "model" ? "bg-white shadow-sm" : "text-slate-600"}`}>Model prediction</button>
                <button onClick={() => setSource("ground_truth")} disabled={!area?.sources.includes("ground_truth")}
                  className={`rounded-md py-1.5 disabled:opacity-40 ${source === "ground_truth" ? "bg-white shadow-sm" : "text-slate-600"}`}>Hand label (reference)</button>
              </div>
            </div>
            {source === "model" && area?.kind === "curated" && (
              <Select label="Segmentation model" value={model} onChange={setModel}>
                <option value="">Default model</option>
                {completedModels.map((m) => <option key={m.name} value={m.name}>{modelLabel(m)}</option>)}
              </Select>
            )}
            <Slider label="Flood threshold P(water)" value={threshold} min={0.1} max={0.9} step={0.05} onChange={setThreshold} format={(v) => v.toFixed(2)} />
            <Slider label="Road buffer" value={bufferPx} min={0} max={3} step={1} onChange={setBufferPx} format={(v) => `${v} px (${v * 10} m)`}
              hint="Max-filters the raster to allow for road width and OSM/imagery misalignment." />
          </Card>

          <Card title="Routing assumptions" bodyClass="space-y-3.5 p-4">
            <Slider label="Flood penalty" value={penalty} min={0} max={50} step={1} onChange={setPenalty} format={(v) => `${v}× (+${v} m per flooded m)`} />
            <Slider label="Unknown-data penalty" value={unknownPenalty} min={0} max={5} step={0.5} onChange={setUnknownPenalty} format={(v) => `${v}×`} />
            <Toggle checked={closeFlooded} onChange={setCloseFlooded} label="Treat heavily flooded roads as impassable"
              hint="A conservative assumption chosen by the user, not a verified closure." />
            {closeFlooded && <Slider label="Impassable if flooded length ≥" value={closureM} min={10} max={200} step={10} onChange={setClosureM} format={(v) => `${v} m`} />}
            <Slider label="Alternative routes" value={alts} min={0} max={3} step={1} onChange={setAlts} />
          </Card>

          <Card title="Start & destination" bodyClass="space-y-3 p-4">
            <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1 text-xs font-medium">
              <button onClick={() => setClickMode("start")} className={`flex items-center justify-center gap-1 rounded-md py-1.5 ${clickMode === "start" ? "bg-white shadow-sm" : "text-slate-600"}`}><MapPin className="h-3 w-3 text-emerald-600" />Set start</button>
              <button onClick={() => setClickMode("end")} className={`flex items-center justify-center gap-1 rounded-md py-1.5 ${clickMode === "end" ? "bg-white shadow-sm" : "text-slate-600"}`}><Flag className="h-3 w-3 text-red-600" />Set destination</button>
            </div>
            <div className="space-y-1 font-mono text-[11px] text-slate-600">
              <div>Start: {start ? `${start.lat.toFixed(5)}, ${start.lon.toFixed(5)}` : "click the map"}</div>
              <div>Dest.: {end ? `${end.lat.toFixed(5)}, ${end.lon.toFixed(5)}` : "click the map"}</div>
            </div>
            <Button onClick={runPlan} disabled={!start || !end || planning || !detail} className="w-full">
              {planning ? <Spinner label="Computing routes…" /> : <><Navigation className="h-4 w-4" />Plan routes</>}
            </Button>
          </Card>
        </div>

        <div className="min-w-0 space-y-4">
          <div className={`flex items-start gap-2 rounded-lg border px-3.5 py-2.5 text-[13px] ${isPrediction ? "border-water-100 bg-water-50 text-water-900" : "border-violet-200 bg-violet-50 text-violet-900"}`}>
            {isPrediction ? <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" /> : <Info className="mt-0.5 h-4 w-4 shrink-0" />}
            <div>
              <b>{isPrediction ? "Real geospatial model prediction." : "Reference mode: hand-labelled ground truth."}</b>{" "}
              {detail?.layer.description ?? (isPrediction ? "The flood layer is the trained model's output on this area's imagery." : "Use it to compare routing on the model output against routing on the true labels.")}
            </div>
          </div>

          <Card bodyClass="p-0" className="overflow-hidden">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-slate-100 px-4 py-2 text-xs">
              {(["basemap", "imagery", "flood", "roads"] as const).map((k) => (
                <label key={k} className="flex cursor-pointer items-center gap-1.5 text-slate-700">
                  <input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} className="accent-water-600" />
                  {{ basemap: "OSM basemap", imagery: "Sentinel-2 image", flood: "Flood layer", roads: "Road risk" }[k]}
                </label>
              ))}
              <span className="ml-auto text-slate-500">Next click sets the <b className={clickMode === "start" ? "text-emerald-700" : "text-red-700"}>{clickMode === "start" ? "start" : "destination"}</b></span>
              {detailLoading && <Spinner label="Scoring roads…" />}
            </div>
            <div className="relative h-[560px]">
              <MapContainer center={[20, 0]} zoom={3} className="h-full w-full" scrollWheelZoom>
                {layers.basemap && <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19}
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />}
                <FitBounds bounds={bounds} />
                <ClickHandler onClick={(p) => { if (clickMode === "start") { setStart(p); setClickMode("end"); } else { setEnd(p); setClickMode("start"); } }} />
                {area && layerBounds && layers.imagery && <ImageOverlay url={imageryUrl(area.id)} bounds={layerBounds} opacity={0.85} />}
                {area && layerBounds && layers.flood && detail && (
                  <ImageOverlay key={`${area.id}-${source}-${threshold}-${model}`} url={overlayUrl(area.id, source, threshold, model || undefined)} bounds={layerBounds} opacity={0.8} />
                )}
                {area && bounds && <Polyline positions={[[area.bounds[1], area.bounds[0]], [area.bounds[1], area.bounds[2]], [area.bounds[3], area.bounds[2]], [area.bounds[3], area.bounds[0]], [area.bounds[1], area.bounds[0]]]}
                  pathOptions={{ color: "#0f172a", weight: 1, dashArray: "4 4", opacity: 0.6 }} />}
                {layers.roads && detail && (
                  <GeoJSON key={`${detail.area.id}-${source}-${threshold}-${bufferPx}-${model}`} data={detail.roads as GeoJSON.FeatureCollection}
                    style={(f) => roadStyle(f as unknown as RoadFeature)}
                    onEachFeature={(f, l) => {
                      const p = (f as unknown as RoadFeature).properties;
                      l.bindTooltip(`${p.name ?? p.highway ?? "road"} · ${RISK_LABEL[p.risk_class]} · ${Math.round(p.flooded_len_m)} m flooded of ${Math.round(p.length_m)} m`, { sticky: true });
                    }} />
                )}
                <Pane name="routes" style={{ zIndex: 450 }}>
                  {plan?.alternatives.map((a, i) => (
                    <Polyline key={`alt${i}`} positions={a.coordinates} pathOptions={{ color: ROUTE_STYLE.alt[i], weight: 4, opacity: 0.8 }}>
                      <Tooltip sticky>Alternative {i + 1}: {fmtDist(a.distance_m)}, {fmtDist(a.flooded_length_m)} flooded</Tooltip>
                    </Polyline>
                  ))}
                  {plan && <Polyline positions={plan.baseline.coordinates} pathOptions={ROUTE_STYLE.baseline}><Tooltip sticky>Shortest route (ignores flooding)</Tooltip></Polyline>}
                  {plan && <Polyline positions={plan.risk_aware.coordinates} pathOptions={ROUTE_STYLE.risk}><Tooltip sticky>Risk-aware route</Tooltip></Polyline>}
                  {plan?.risk_aware.segments.filter((s) => s.risk_class === "flooded" || s.risk_class === "partial").map((s, i) => (
                    <Polyline key={`hz${i}`} positions={s.coords} pathOptions={{ color: "#dc2626", weight: 9, opacity: 0.5 }} />
                  ))}
                </Pane>
                {start && <CircleMarker center={[start.lat, start.lon]} radius={8} pathOptions={{ color: "#fff", weight: 3, fillColor: "#059669", fillOpacity: 1 }}><Tooltip>Start</Tooltip></CircleMarker>}
                {end && <CircleMarker center={[end.lat, end.lon]} radius={8} pathOptions={{ color: "#fff", weight: 3, fillColor: "#dc2626", fillOpacity: 1 }}><Tooltip>Destination</Tooltip></CircleMarker>}
              </MapContainer>
              <div className="pointer-events-none absolute bottom-6 left-3 z-[1000] rounded-lg bg-white/95 p-2.5 text-[11px] shadow-md ring-1 ring-slate-200">
                <div className="mb-1 font-semibold text-slate-800">Road risk</div>
                {Object.entries(RISK_LABEL).map(([k, l]) => (
                  <div key={k} className="flex items-center gap-1.5"><span className="h-1 w-4 rounded" style={{ background: RISK_COLORS[k] }} />{l}</div>
                ))}
                <div className="mt-1.5 border-t border-slate-200 pt-1.5 font-semibold text-slate-800">Routes</div>
                <div className="flex items-center gap-1.5"><span className="h-1 w-4 rounded" style={{ background: ROUTE_STYLE.risk.color }} />Risk-aware</div>
                <div className="flex items-center gap-1.5"><span className="h-1 w-4" style={{ background: "repeating-linear-gradient(90deg,#0f172a 0 3px,transparent 3px 5px)" }} />Shortest (baseline)</div>
                <div className="flex items-center gap-1.5"><span className="h-1 w-4 rounded" style={{ background: ROUTE_STYLE.alt[0] }} />Alternatives</div>
                <div className="flex items-center gap-1.5"><span className="h-2 w-4 rounded" style={{ background: "#1d4ed8", opacity: 0.6 }} />Predicted water</div>
              </div>
            </div>
          </Card>

          {detailErr && <ErrorState error={detailErr} />}
          {detail && (
            <div className="grid gap-3 sm:grid-cols-4">
              <MiniStat label="Road edges scored" value={detail.risk_stats.edges.toLocaleString()} />
              <MiniStat label="Flooded road (network)" value={`${detail.risk_stats.flooded_road_km.toFixed(2)} km`} />
              <MiniStat label="Flood-affected edges" value={(detail.risk_stats.risk_class_counts.flooded + detail.risk_stats.risk_class_counts.partial).toLocaleString()} />
              <MiniStat label="Raster covers graph" value={fmtPct(detail.alignment.graph_fraction_covered, 0)} hint={`${detail.alignment.raster_crs} ↔ ${detail.alignment.graph_crs}`} />
            </div>
          )}

          {planErr && <Callout tone="danger" title="No route">{planErr.message}</Callout>}
          {plan && (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <RouteCard title="Shortest route (baseline)" r={plan.baseline} color={ROUTE_STYLE.baseline.color} dashed badge={<Badge>ignores flooding</Badge>} />
                <RouteCard title="Risk-aware route" r={plan.risk_aware} color={ROUTE_STYLE.risk.color} base={plan.baseline}
                  badge={plan.same_as_baseline ? <Badge color="slate">same as shortest</Badge> : <Badge color="green">lowest flood-weighted cost</Badge>} />
                {plan.alternatives.map((a, i) => (
                  <RouteCard key={i} title={`Alternative ${i + 1}`} r={a} color={ROUTE_STYLE.alt[i]} base={plan.baseline} />
                ))}
              </div>
              <Card title={<span className="flex items-center gap-1.5"><RouteIcon className="h-4 w-4 text-emerald-600" />Why this route?</span>} bodyClass="p-4">
                <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-slate-700">
                  {plan.explanation.map((e) => <li key={e}>{e}</li>)}
                </ul>
                <div className="mt-3 text-[11px] text-slate-500">
                  Start snapped {Math.round(plan.start_snap_m)} m and destination {Math.round(plan.destination_snap_m)} m to the nearest road node · computed in {Math.round(plan.compute_ms)} ms · *{plan.travel_time_note}
                </div>
              </Card>
              {plan.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
            </>
          )}
          <Callout tone="warn" title="Not a safety guarantee">
            <span>
              A route marked low-risk only means the model saw no surface water on those roads on the image date. Depth,
              current, bridge status, debris and road elevation are unknown. Follow official emergency services and
              verified road-closure information.
            </span>
          </Callout>
          {areas.data && <div className="text-[11px] text-slate-500">{areas.data.attribution}. Basemap tiles © OpenStreetMap contributors, loaded on demand under the OSMF tile usage policy (turn the basemap off to work fully offline).</div>}
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2.5">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="text-lg font-semibold tabular-nums text-slate-900">{value}</div>
      {hint && <div className="text-[10px] text-slate-400">{hint}</div>}
    </div>
  );
}
