import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CircleMarker, GeoJSON, ImageOverlay, MapContainer, Pane, Polyline, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import type { LatLngBoundsExpression, LeafletMouseEvent, PathOptions } from "leaflet";
import { AlertOctagon, ArrowLeftRight, CheckCircle2, Download, FileText, Flag, List, MapPin, Navigation, Search, ShieldAlert } from "lucide-react";
import { api, imageryUrl, overlayUrl, routeExportUrl, type Area, type AreaDetail, type PlanResponse, type RoadFeature, type Route } from "../api";
import { Badge, Button, DataTable, Drawer, ErrorState, Field, InfoTip, KV, Notice, PageHead, Panel, Segmented, Select, Skeleton, Slider, Spinner, TextInput, Toggle } from "../components/ui";
import { RISK_COLORS, RISK_LABEL, fmtDist, fmtDuration, fmtPct, fmtTime, modelLabel, useApi } from "../lib";

type Pt = { lat: number; lon: number };
const STYLE = { baseline: "#0f172a", risk: "#059669", alt: ["#7c3aed", "#db2777", "#0891b2"] };
const CLASS_ORDER = ["clear", "low", "partial", "flooded", "unknown"];

function FitBounds({ bounds }: { bounds: LatLngBoundsExpression | null }) {
  const map = useMap();
  useEffect(() => { if (bounds) map.fitBounds(bounds, { padding: [12, 12] }); }, [bounds, map]);
  return null;
}
function Clicks({ onClick }: { onClick: (p: Pt) => void }) {
  useMapEvents({ click: (e: LeafletMouseEvent) => onClick({ lat: e.latlng.lat, lon: e.latlng.lng }) });
  return null;
}

function RiskBar({ r }: { r: Route }) {
  const total = Object.values(r.length_by_risk_class_m).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="flex h-2 w-full overflow-hidden rounded-full bg-slate-100" role="img"
      aria-label={CLASS_ORDER.filter((k) => r.length_by_risk_class_m[k]).map((k) => `${k} ${Math.round(r.length_by_risk_class_m[k])} m`).join(", ")}>
      {CLASS_ORDER.map((k) => r.length_by_risk_class_m[k] ? (
        <div key={k} title={`${RISK_LABEL[k]}: ${fmtDist(r.length_by_risk_class_m[k])}`} style={{ width: `${(100 * r.length_by_risk_class_m[k]) / total}%`, background: RISK_COLORS[k] }} />
      ) : null)}
    </div>
  );
}

const STATUS = {
  flood_free_route_found: { tone: "ok" as const, icon: CheckCircle2, title: "Flood-free route found (by the model's prediction)" },
  no_flood_free_route: { tone: "bad" as const, icon: AlertOctagon, title: "No flood-free route exists between these points" },
  no_route_avoids_closures: { tone: "bad" as const, icon: AlertOctagon, title: "No route avoids the impassable segments" },
};

export default function RoutesPage() {
  const [params] = useSearchParams();
  const areas = useApi(() => api.areas(), []);
  const models = useApi(() => api.modelStatus(), []);
  const [extra, setExtra] = useState<Area | null>(null);
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
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<{ name: string; lat: number; lon: number; segments: number }[]>([]);
  const [segOpen, setSegOpen] = useState(false);

  const all = useMemo(() => [...(areas.data?.areas ?? []), ...(extra ? [extra] : [])], [areas.data, extra]);
  const area = all.find((a) => a.id === areaId);

  useEffect(() => {
    if (!areas.data || areaId) return;
    const chip = params.get("chip"), result = params.get("result"), qa = params.get("area");
    const cur = areas.data.areas.find((a) => a.id === qa || (chip && a.chip_id === chip));
    if (cur) return setAreaId(cur.id);
    const rid = result ? `r_${result}` : qa?.startsWith("r_") ? qa : null;
    if (rid) {
      api.area(rid, { source: "model", threshold: 0.5, buffer_px: 1 }).then((d) => { setExtra(d.area); setAreaId(rid); })
        .catch((e) => { setDetailErr(e); setAreaId(areas.data!.areas[0]?.id ?? ""); });
      return;
    }
    setAreaId(areas.data.areas[0]?.id ?? "");
  }, [areas.data, areaId, params]);

  useEffect(() => {
    if (!area) return;
    setPlan(null); setPlanErr(null); setHits([]); setQ("");
    setStart(area.default_start ? { lat: area.default_start[0], lon: area.default_start[1] } : null);
    setEnd(area.default_end ? { lat: area.default_end[0], lon: area.default_end[1] } : null);
    if (!area.sources.includes(source)) setSource("model");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [areaId]);

  useEffect(() => {
    if (!areaId) return;
    let off = false;
    setDetailLoading(true); setDetailErr(null);
    api.area(areaId, { source, threshold, buffer_px: bufferPx, model: model || undefined })
      .then((d) => !off && setDetail(d)).catch((e) => { if (!off) { setDetail(null); setDetailErr(e); } })
      .finally(() => !off && setDetailLoading(false));
    return () => { off = true; };
  }, [areaId, source, threshold, bufferPx, model]);

  const runPlan = async () => {
    if (!start || !end || !areaId) return;
    setPlanning(true); setPlanErr(null);
    try {
      setPlan(await api.plan({ area_id: areaId, source, model: model || null, start, end, threshold, buffer_px: bufferPx,
        flood_penalty: penalty, unknown_penalty: unknownPenalty, close_flooded: closeFlooded, closure_min_flooded_m: closureM, alternatives: alts }));
    } catch (e) { setPlan(null); setPlanErr(e as Error); } finally { setPlanning(false); }
  };
  useEffect(() => { if (plan && start && end) runPlan(); /* re-plan on assumption changes */ // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, threshold, bufferPx, penalty, unknownPenalty, closeFlooded, closureM, alts, model]);

  const search = async () => {
    if (!areaId || q.trim().length < 2) return;
    try { setHits((await api.searchArea(areaId, q.trim())).results); } catch { setHits([]); }
  };

  const bounds: LatLngBoundsExpression | null = area ? [[area.bounds[1], area.bounds[0]], [area.bounds[3], area.bounds[2]]] : null;
  const lb: LatLngBoundsExpression | null = detail ? [[detail.layer.bounds_wgs84[1], detail.layer.bounds_wgs84[0]], [detail.layer.bounds_wgs84[3], detail.layer.bounds_wgs84[2]]] : bounds;
  const roadStyle = (f?: RoadFeature): PathOptions => ({ color: RISK_COLORS[f?.properties.risk_class ?? "unknown"], weight: f?.properties.risk_class === "clear" ? 2 : 3.5, opacity: 0.9 });
  const done = (models.data?.models ?? []).filter((m) => m.training_state === "completed");
  const routes: [string, Route, string, boolean][] = plan ? [["Shortest (baseline)", plan.baseline, STYLE.baseline, true], ["Risk-aware", plan.risk_aware, STYLE.risk, false],
    ...plan.alternatives.map((a, i) => [`Alternative ${i + 1}`, a, STYLE.alt[i], false] as [string, Route, string, boolean])] : [];
  const S = plan ? STATUS[plan.status] : null;

  return (
    <div>
      <PageHead title="Flood-risk route planner"
        sub="Roads are scored against a georeferenced flood raster, and Dijkstra's algorithm compares the shortest route with a flood-penalised one. Click the map or search a road name to set the start and destination." />
      <div className="grid gap-3 2xl:grid-cols-[290px_minmax(0,1fr)_340px] xl:grid-cols-[290px_minmax(0,1fr)]">
        {/* controls */}
        <div className="min-w-0 space-y-3">
          <Panel title="Area & flood layer">
            <div className="space-y-3">
              {areas.error ? <ErrorState error={areas.error} onRetry={areas.reload} compact /> : (
                <Field label="Study area"><Select label="Study area" value={areaId} onChange={setAreaId}>{all.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}</Select></Field>
              )}
              <Field label="Flood layer" info="Model prediction = the trained network's output on the area's Sentinel imagery. Hand label = Sen1Floods11 reference annotation, for comparison only.">
                <Segmented label="Flood layer" value={source} onChange={setSource} options={[{ value: "model", label: "Model prediction" },
                  { value: "ground_truth", label: "Hand label", disabled: !area?.sources.includes("ground_truth") }]} />
              </Field>
              {source === "model" && area?.kind === "curated" && (
                <Field label="Segmentation model"><Select label="Segmentation model" value={model} onChange={setModel}>
                  <option value="">Default model</option>{done.map((m) => <option key={m.name} value={m.name}>{modelLabel(m)} · {m.name}</option>)}</Select></Field>
              )}
              <Slider label="Flood threshold" value={threshold} min={0.1} max={0.9} step={0.05} onChange={setThreshold} format={(v) => v.toFixed(2)} info="Pixels with P(water) at or above this count as flooded when scoring roads." />
              <Slider label="Road buffer" value={bufferPx} min={0} max={3} step={1} onChange={setBufferPx} format={(v) => `${v} px · ${v * 10} m`} info="Max-filter on the raster: allows for road width and ±10 m misalignment between OSM and the imagery." />
            </div>
          </Panel>
          <Panel title="Risk assumptions" info="These are user-configurable policy choices, not validated parameters. Cost = length + penalty × flooded length + unknown penalty × length without data + length × mean P(water).">
            <div className="space-y-3">
              <Slider label="Flood penalty" value={penalty} min={0} max={50} step={1} onChange={setPenalty} format={(v) => `${v}× · +${v} m per flooded m`} />
              <Slider label="Unknown-data penalty" value={unknownPenalty} min={0} max={5} step={0.5} onChange={setUnknownPenalty} format={(v) => `${v}×`} />
              <Toggle checked={closeFlooded} onChange={setCloseFlooded} label="Treat heavily flooded roads as impassable" hint="A conservative assumption, not a verified closure" />
              {closeFlooded && <Slider label="Impassable if flooded ≥" value={closureM} min={10} max={200} step={10} onChange={setClosureM} format={(v) => `${v} m`} />}
              <Slider label="Alternative routes" value={alts} min={0} max={3} step={1} onChange={setAlts} />
            </div>
          </Panel>
          <Panel title="Start & destination">
            <div className="space-y-2.5">
              <Segmented label="Next map click sets" value={clickMode} onChange={setClickMode} options={[
                { value: "start", label: <><MapPin className="h-3 w-3 text-ok" />Start</> }, { value: "end", label: <><Flag className="h-3 w-3 text-bad" />Destination</> }]} />
              <form onSubmit={(e) => { e.preventDefault(); search(); }} className="flex gap-1.5">
                <TextInput label="Search road name in this area" value={q} onChange={setQ} placeholder="Road name (offline OSM search)" />
                <Button type="submit" variant="secondary" disabled={q.trim().length < 2}><Search className="h-3.5 w-3.5" /></Button>
              </form>
              {hits.length > 0 && (
                <ul className="max-h-36 divide-y divide-line-2 overflow-auto rounded-md border border-line text-[12px] scroll-thin">
                  {hits.map((h) => (
                    <li key={h.name} className="flex items-center justify-between gap-2 px-2 py-1">
                      <span className="truncate">{h.name}</span>
                      <span className="flex shrink-0 gap-1">
                        <Button size="xs" variant="ghost" onClick={() => setStart({ lat: h.lat, lon: h.lon })}>Start</Button>
                        <Button size="xs" variant="ghost" onClick={() => setEnd({ lat: h.lat, lon: h.lon })}>End</Button>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              <div className="space-y-0.5 font-mono text-[11px] text-ink-2">
                <div>Start {start ? `${start.lat.toFixed(5)}, ${start.lon.toFixed(5)}` : "—"}</div>
                <div>Dest. {end ? `${end.lat.toFixed(5)}, ${end.lon.toFixed(5)}` : "—"}</div>
              </div>
              <div className="flex gap-1.5">
                <Button className="flex-1" size="md" onClick={runPlan} disabled={!start || !end || planning || !detail}>
                  {planning ? <Spinner label="Routing…" /> : detailLoading ? <Spinner label="Scoring roads…" /> : <><Navigation className="h-4 w-4" />Plan routes</>}
                </Button>
                <Button variant="secondary" size="md" title="Swap start and destination" onClick={() => { setStart(end); setEnd(start); }} disabled={!start || !end}><ArrowLeftRight className="h-4 w-4" /></Button>
              </div>
            </div>
          </Panel>
        </div>

        {/* map */}
        <div className="min-w-0 space-y-3">
          <div className={`flex items-start gap-2 rounded-md border px-3 py-2 text-[12px] ${source === "model" ? "border-blue-200 bg-water-soft text-blue-950" : "border-violet-200 bg-truth-soft text-violet-950"}`}>
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <div><b>{source === "model" ? "Model-predicted flood layer." : "Reference layer: hand-labelled ground truth (not a prediction)."}</b> {detail?.layer.description}
              {area?.event && <> · imagery {area.event}</>}{area?.road_meta && <> · roads fetched {fmtTime(area.road_meta.fetched_utc)}</>}</div>
          </div>
          <Panel body="p-0" title="Map" actions={<div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] normal-case tracking-normal">
              {(["basemap", "imagery", "flood", "roads"] as const).map((k) => (
                <label key={k} className="flex cursor-pointer items-center gap-1 text-ink-2">
                  <input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} className="accent-[var(--color-accent)]" />
                  {{ basemap: "OSM basemap", imagery: "Sentinel-2", flood: "Flood layer", roads: "Road risk" }[k]}
                </label>
              ))}
              {detailLoading && <Spinner label="Scoring roads" />}
            </div>}>
            <div className="relative h-[calc(100vh-230px)] min-h-[460px]">
              <MapContainer center={[20, 0]} zoom={3} className="h-full w-full" scrollWheelZoom>
                {layers.basemap && <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19} attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />}
                <FitBounds bounds={bounds} />
                <Clicks onClick={(p) => { if (clickMode === "start") { setStart(p); setClickMode("end"); } else { setEnd(p); setClickMode("start"); } }} />
                {area && lb && layers.imagery && <ImageOverlay url={imageryUrl(area.id)} bounds={lb} opacity={0.85} />}
                {area && lb && layers.flood && detail && <ImageOverlay key={`${area.id}-${source}-${threshold}-${model}`} url={overlayUrl(area.id, source, threshold, model || undefined)} bounds={lb} opacity={0.8} />}
                {area && <Polyline positions={[[area.bounds[1], area.bounds[0]], [area.bounds[1], area.bounds[2]], [area.bounds[3], area.bounds[2]], [area.bounds[3], area.bounds[0]], [area.bounds[1], area.bounds[0]]]}
                  pathOptions={{ color: "#0f172a", weight: 1, dashArray: "4 4", opacity: 0.6 }} />}
                {layers.roads && detail && <GeoJSON key={`${detail.area.id}-${source}-${threshold}-${bufferPx}-${model}`} data={detail.roads as GeoJSON.FeatureCollection}
                  style={(f) => roadStyle(f as unknown as RoadFeature)}
                  onEachFeature={(f, l) => { const p = (f as unknown as RoadFeature).properties; l.bindTooltip(`${p.name ?? p.highway ?? "road"} · ${RISK_LABEL[p.risk_class]} · ${Math.round(p.flooded_len_m)} of ${Math.round(p.length_m)} m flooded`, { sticky: true }); }} />}
                <Pane name="routes" style={{ zIndex: 450 }}>
                  {plan?.alternatives.map((a, i) => <Polyline key={i} positions={a.coordinates} pathOptions={{ color: STYLE.alt[i], weight: 4, opacity: 0.8 }}><Tooltip sticky>Alternative {i + 1}: {fmtDist(a.distance_m)}, {fmtDist(a.flooded_length_m)} flooded</Tooltip></Polyline>)}
                  {plan && <Polyline positions={plan.baseline.coordinates} pathOptions={{ color: STYLE.baseline, weight: 5, dashArray: "8 7", opacity: 0.85 }}><Tooltip sticky>Shortest route (ignores flooding)</Tooltip></Polyline>}
                  {plan && <Polyline positions={plan.risk_aware.coordinates} pathOptions={{ color: STYLE.risk, weight: 7, opacity: 0.95 }}><Tooltip sticky>Risk-aware route</Tooltip></Polyline>}
                  {plan?.risk_aware.segments.filter((s) => s.risk_class === "flooded" || s.risk_class === "partial").map((s, i) => <Polyline key={`h${i}`} positions={s.coords} pathOptions={{ color: "#dc2626", weight: 10, opacity: 0.45 }} />)}
                </Pane>
                {start && <CircleMarker center={[start.lat, start.lon]} radius={8} pathOptions={{ color: "#fff", weight: 3, fillColor: "#059669", fillOpacity: 1 }}><Tooltip>Start</Tooltip></CircleMarker>}
                {end && <CircleMarker center={[end.lat, end.lon]} radius={8} pathOptions={{ color: "#fff", weight: 3, fillColor: "#dc2626", fillOpacity: 1 }}><Tooltip>Destination</Tooltip></CircleMarker>}
              </MapContainer>
              <div className="pointer-events-none absolute bottom-5 left-2 z-[1000] rounded-md border border-line bg-white/95 p-2 text-[10.5px] shadow-sm">
                <div className="mb-0.5 font-semibold text-ink">Road risk</div>
                {CLASS_ORDER.map((k) => <div key={k} className="flex items-center gap-1.5"><span className="h-1 w-4 rounded" style={{ background: RISK_COLORS[k] }} />{RISK_LABEL[k]}</div>)}
                <div className="mt-1 border-t border-line pt-1 font-semibold text-ink">Routes</div>
                <div className="flex items-center gap-1.5"><span className="h-1 w-4 rounded" style={{ background: STYLE.risk }} />Risk-aware</div>
                <div className="flex items-center gap-1.5"><span className="h-1 w-4" style={{ background: `repeating-linear-gradient(90deg,${STYLE.baseline} 0 3px,transparent 3px 5px)` }} />Shortest</div>
                <div className="flex items-center gap-1.5"><span className="h-1 w-4 rounded" style={{ background: STYLE.alt[0] }} />Alternatives</div>
                <div className="flex items-center gap-1.5"><span className="h-2 w-4 rounded" style={{ background: "#1d4ed8", opacity: 0.6 }} />{source === "model" ? "Predicted water" : "Labelled water"}</div>
              </div>
            </div>
          </Panel>
          {detailErr && <ErrorState error={detailErr} />}
          {detail && (
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {[["Road edges scored", detail.risk_stats.edges.toLocaleString()], ["Flooded road (network)", `${detail.risk_stats.flooded_road_km.toFixed(2)} km`],
                ["Flood-affected edges", (detail.risk_stats.risk_class_counts.flooded + detail.risk_stats.risk_class_counts.partial).toLocaleString()],
                ["Raster covers graph", `${fmtPct(detail.alignment.graph_fraction_covered, 0)} · ${detail.alignment.raster_crs}`]].map(([k, v]) => (
                <div key={k} className="rounded-md border border-line bg-panel px-2.5 py-2"><div className="text-[10.5px] uppercase tracking-wide text-ink-3">{k}</div><div className="text-[14px] font-semibold tnum">{v}</div></div>
              ))}
            </div>
          )}
        </div>

        {/* analysis */}
        <div className="min-w-0 space-y-3 xl:col-span-2 2xl:col-span-1">
          {planErr && <Notice tone="bad" title="No route">{planErr.message}</Notice>}
          {!plan && !planErr && <Panel title="Route analysis"><p className="text-[12px] text-ink-3">Set a start and destination, then plan routes. The comparison, risk breakdown and explanation will appear here.</p></Panel>}
          {plan && S && (<>
            <div className={`rounded-md border px-3 py-2.5 text-[12.5px] ${S.tone === "ok" ? "border-green-200 bg-ok-soft text-green-950" : "border-red-300 bg-bad-soft text-red-950"}`} role={S.tone === "bad" ? "alert" : "status"}>
              <div className="flex items-center gap-1.5 font-semibold"><S.icon className="h-4 w-4" />{S.title}</div>
              <div className="mt-0.5 leading-snug">{plan.status_note}</div>
            </div>
            <Panel title="Route comparison" body="p-0" actions={<InfoTip text={plan.risk_aware.risk_level_rule} />}>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px] tnum">
                  <thead className="bg-slate-50 text-[10px] uppercase tracking-[0.06em] text-ink-3">
                    <tr><th className="px-2.5 py-1.5 text-left">Route</th><th className="px-2 py-1.5 text-right">Dist.</th><th className="px-2 py-1.5 text-right">Time*</th><th className="px-2.5 py-1.5 text-right">Flooded</th></tr>
                  </thead>
                  <tbody className="divide-y divide-line-2">
                    {routes.map(([n, r, c, dashed]) => (
                      <tr key={n}>
                        <td className="px-2.5 py-1.5">
                          <div className="flex items-center gap-1.5 font-medium text-ink"><span className="inline-block h-1 w-5 shrink-0 rounded" style={{ background: dashed ? `repeating-linear-gradient(90deg,${c} 0 4px,transparent 4px 7px)` : c }} />{n}</div>
                          <div className="mt-1"><RiskBar r={r} /></div>
                        </td>
                        <td className="px-2 py-1.5 text-right align-top">{fmtDist(r.distance_m)}{n !== "Shortest (baseline)" && <div className={`text-[10.5px] ${r.distance_m > plan.baseline.distance_m ? "text-warn" : "text-ok"}`}>{r.distance_m >= plan.baseline.distance_m ? "+" : "−"}{fmtDist(Math.abs(r.distance_m - plan.baseline.distance_m))}</div>}</td>
                        <td className="px-2 py-1.5 text-right align-top">{fmtDuration(r.travel_time_s)}</td>
                        <td className={`px-2.5 py-1.5 text-right align-top font-semibold ${r.flooded_length_m > 0 ? "text-bad" : "text-ok"}`}>{fmtDist(r.flooded_length_m)}<div className="text-[10.5px] font-normal text-ink-3">{fmtPct(r.flooded_share)} · max P {r.max_flood_prob.toFixed(2)}</div></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap gap-2 border-t border-line-2 px-2.5 py-1.5 text-[10.5px] text-ink-3">
                {CLASS_ORDER.map((k) => <span key={k} className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm" style={{ background: RISK_COLORS[k] }} />{k}</span>)}
              </div>
            </Panel>
            <Panel title="Why this route?">
              <ul className="list-disc space-y-1 pl-4 text-[12px] leading-relaxed text-ink">{plan.explanation.map((e) => <li key={e}>{e}</li>)}</ul>
              {plan.warnings.map((w) => <div key={w} className="mt-2"><Notice tone="warn">{w}</Notice></div>)}
            </Panel>
            <Panel title="Risk-aware route · flood-affected roads" actions={<Button size="xs" variant="ghost" onClick={() => setSegOpen(true)}><List className="h-3 w-3" />All segments</Button>}>
              {plan.risk_aware.flood_affected_roads.length === 0 ? <p className="text-[12px] text-ok">No road on this route is predicted as flooded.</p> : (
                <ul className="space-y-1 text-[12px]">{plan.risk_aware.flood_affected_roads.map((a) => <li key={a.road} className="flex justify-between gap-2"><span className="truncate">{a.road}</span><span className="shrink-0 font-semibold text-bad tnum">{fmtDist(a.flooded_m)}</span></li>)}</ul>
              )}
              <p className="mt-2 text-[11px] text-ink-3">Risk level: <b className="text-ink">{plan.risk_aware.risk_level}</b></p>
            </Panel>
            <Panel title="Data freshness & export">
              <KV dense items={[["Flood layer", plan.flood_layer.is_model_prediction ? <Badge tone="water">model prediction</Badge> : <Badge tone="truth">hand label</Badge>],
                ["Imagery", plan.imagery_event ?? "uploaded scene"], ["Roads", plan.road_meta ? `OSM, fetched ${fmtTime(plan.road_meta.fetched_utc)}` : "OSM"],
                ["Snap distance", `${Math.round(plan.start_snap_m)} m / ${Math.round(plan.destination_snap_m)} m`], ["Computed", `${Math.round(plan.compute_ms)} ms`]]} />
              <p className="mt-1.5 text-[10.5px] leading-snug text-ink-3">*{plan.travel_time_note} {plan.attribution}.</p>
              <div className="mt-2 grid grid-cols-3 gap-1.5">
                <Button size="xs" variant="secondary" href={routeExportUrl(plan.id, "json")}><Download className="h-3 w-3" />JSON</Button>
                <Button size="xs" variant="secondary" href={routeExportUrl(plan.id, "geojson")}><Download className="h-3 w-3" />GeoJSON</Button>
                <Button size="xs" variant="dark" href={routeExportUrl(plan.id, "html")}><FileText className="h-3 w-3" />Report</Button>
              </div>
            </Panel>
          </>)}
          <Notice tone="warn" title="Not a safety guarantee">A route with no predicted flooding only means the model saw no surface water on those roads on the image date. Depth, current, bridges and road elevation are unknown. Follow official guidance.</Notice>
          {!areas.data && <Skeleton className="h-20" />}
        </div>
      </div>

      <Drawer open={segOpen} onClose={() => setSegOpen(false)} title="Risk-aware route · road segments" wide>
        {plan && <DataTable rowKey={(r) => `${r.coords[0]}-${r.coords[r.coords.length - 1]}`} rows={plan.risk_aware.segments} csvName="route_segments"
          columns={[{ key: "n", label: "Road", value: (r) => r.name ?? "(unnamed)" }, { key: "h", label: "Type", value: (r) => r.highway },
            { key: "c", label: "Risk class", value: (r) => r.risk_class, render: (r) => <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: RISK_COLORS[r.risk_class] }} />{r.risk_class}</span> },
            { key: "l", label: "Length (m)", numeric: true, value: (r) => r.length_m }, { key: "f", label: "Flooded (m)", numeric: true, value: (r) => r.flooded_len_m }]} />}
      </Drawer>
    </div>
  );
}
