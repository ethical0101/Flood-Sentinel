import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Download, FileImage, FileText, Globe2, HelpCircle, Layers, MapPinned, Play, Satellite, Upload, X } from "lucide-react";
import { api, reportUrl, resultFile, type SegResult } from "../api";
import { Badge, Button, Drawer, Empty, ErrorState, Field, KV, Notice, PageHead, Panel, Segmented, Select, Skeleton, Slider, Spinner, Toggle } from "../components/ui";
import { METRIC_INFO, MODALITY_LABEL, fmtInt, fmtMs, fmtNum, fmtPct, fmtTime, modelLabel, useApi } from "../lib";

type View = "compare" | "overlay" | "mask" | "confidence" | "uncertainty" | "ndwi" | "truth";
const ACCEPT = ".tif,.tiff,.png,.jpg,.jpeg";
const MAX_FRAMES = 20;
const COMPAT: Record<string, string[]> = { s1s2: ["s1s2", "s1s2i", "s2", "s2i", "s1", "rgb"], s2: ["s2", "s2i", "rgb"], s1: ["s1"], rgb: ["rgb"] };

const LEGENDS: Record<View, [string, string][]> = {
  compare: [["#008cff", "Water predicted by the model (above threshold)"]],
  overlay: [["#008cff", "Water predicted by the model (above threshold)"]],
  mask: [["#ffffff", "Water (model)"], ["#000000", "Not water"]],
  confidence: [],
  uncertainty: [],
  ndwi: [["#2563eb", "Both model and NDWI"], ["#06b6d4", "Model only"], ["#f59e0b", "NDWI only"]],
  truth: [["#2563eb", "True positive"], ["#ef4444", "False positive"], ["#f59e0b", "False negative"], ["#94a3b8", "Unlabelled (ignored)"]],
};

function Ramp({ name, label }: { name: "viridis" | "magma"; label: string }) {
  const g = name === "viridis" ? "#440154,#3b528b,#21918c,#5ec962,#fde725" : "#000004,#3b0f70,#8c2981,#de4968,#fe9f6d,#fcfdbf";
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-2"><span>0</span>
      <span className="h-2 w-40 rounded" style={{ background: `linear-gradient(90deg,${g})` }} /><span>1</span><span className="text-ink-3">{label}</span></div>
  );
}

function CompareSlider({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="relative mx-auto aspect-square w-full max-w-[640px] select-none overflow-hidden rounded-md bg-nav">
      <img src={before} alt="Input image" className="absolute inset-0 h-full w-full object-contain [image-rendering:pixelated]" />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img src={after} alt="Model prediction overlay" className="absolute inset-0 h-full w-full object-contain [image-rendering:pixelated]" />
      </div>
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_6px_rgba(0,0,0,.6)]" style={{ left: `${pos}%` }} />
      <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] font-semibold text-white">INPUT</span>
      <span className="absolute right-2 top-2 rounded bg-water/90 px-1.5 py-0.5 text-[10px] font-semibold text-white">MODEL PREDICTION</span>
      <input type="range" min={0} max={100} value={pos} onChange={(e) => setPos(Number(e.target.value))} aria-label="Comparison divider position"
        className="absolute inset-x-0 bottom-2 mx-auto w-2/3 accent-white" />
    </div>
  );
}

function Img({ src, alt, checker }: { src: string; alt: string; checker?: boolean }) {
  return (
    <div className={`mx-auto aspect-square w-full max-w-[640px] overflow-hidden rounded-md ${checker ? "bg-[repeating-conic-gradient(#e3e8ef_0_25%,#f4f6f9_0_50%)] bg-[length:16px_16px]" : "bg-nav"}`}>
      <img src={src} alt={alt} className="h-full w-full object-contain [image-rendering:pixelated]" />
    </div>
  );
}

export default function Studio() {
  const models = useApi(() => api.modelStatus(), []);
  const samples = useApi(() => api.samples(), []);
  const formats = useApi(() => api.formats(), []);
  const [mode, setMode] = useState<"sample" | "upload">("sample");
  const [files, setFiles] = useState<File[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [chip, setChip] = useState("");
  const [sampleInput, setSampleInput] = useState("s1s2");
  const [eventFilter, setEventFilter] = useState("all");
  const [model, setModel] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [tta, setTta] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<SegResult | null>(null);
  const [batch, setBatch] = useState<{ name: string; result?: SegResult; error?: string }[]>([]);
  const [view, setView] = useState<View>("compare");
  const [opacity, setOpacity] = useState(0.65);
  const [helpOpen, setHelpOpen] = useState(false);
  const [compareId, setCompareId] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const prev = useApi(() => api.history({ kind: "segmentation", limit: 30 }), [result?.id]);
  const compare = useApi(() => (compareId ? api.result(compareId) : Promise.resolve(null)), [compareId]);
  const file = files[0] ?? null;

  useEffect(() => {
    if (!chip && samples.data?.samples.length) setChip((samples.data.samples.find((x) => x.id === "Spain_7370579") ?? samples.data.samples[0]).id);
  }, [samples.data, chip]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const t = setInterval(() => setElapsed((Date.now() - t0) / 1000), 100);
    return () => clearInterval(t);
  }, [busy]);

  const events = useMemo(() => Array.from(new Set(samples.data?.samples.map((s) => s.event) ?? [])).sort(), [samples.data]);
  const shown = (samples.data?.samples ?? []).filter((s) => eventFilter === "all" || s.event === eventFilter);
  const completed = (models.data?.models ?? []).filter((m) => m.training_state === "completed");
  const compatible = mode === "sample" ? completed.filter((m) => COMPAT[sampleInput].includes(m.modality)) : completed;
  useEffect(() => { if (model && !compatible.some((m) => m.name === model)) setModel(""); }, [compatible, model]);

  const maxMb = formats.data?.max_upload_mb ?? 40;
  const pick = (list: FileList | File[] | null | undefined) => {
    setError(null);
    const all = Array.from(list ?? []);
    if (!all.length) return;
    const bad = all.filter((f) => !/\.(tiff?|png|jpe?g)$/i.test(f.name));
    const big = all.filter((f) => f.size > maxMb * 1e6 * 1.048576);
    const ok = all.filter((f) => !bad.includes(f) && !big.includes(f)).slice(0, MAX_FRAMES);
    const msgs = [];
    if (bad.length) msgs.push(`${bad.length} file(s) skipped: only GeoTIFF, PNG and JPEG are supported`);
    if (big.length) msgs.push(`${big.length} file(s) over the ${maxMb} MB limit`);
    if (all.length - bad.length - big.length > MAX_FRAMES) msgs.push(`only the first ${MAX_FRAMES} frames are kept`);
    if (msgs.length) setError(new Error(msgs.join("; ") + "."));
    if (!ok.length) return;
    setFiles(ok);
    setBatch([]);
    setPreview(/\.(png|jpe?g)$/i.test(ok[0].name) ? URL.createObjectURL(ok[0]) : null);
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    setElapsed(0);
    try {
      if (mode === "upload" && files.length > 1) {
        const out: typeof batch = [];
        for (let i = 0; i < files.length; i++) {
          setProgress(`Frame ${i + 1}/${files.length}`);
          try {
            const r = await api.predictUpload(files[i], { model: model || undefined, threshold, tta });
            out.push({ name: files[i].name, result: r });
            setResult(r);
          } catch (e) {
            out.push({ name: files[i].name, error: (e as Error).message });
          }
          setBatch([...out]);
        }
      } else {
        setProgress(mode === "upload" ? "Uploading & running model" : "Running model");
        const r = mode === "upload" && file
          ? await api.predictUpload(file, { model: model || undefined, threshold, tta })
          : await api.predictSample({ chip_id: chip, model: model || undefined, threshold, tta, input: sampleInput });
        setResult(r);
        setBatch([]);
      }
      setView("compare");
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const canRun = !busy && (mode === "upload" ? !!file : !!chip) && completed.length > 0;
  const img = (n: string) => (result ? resultFile(result.id, n) : "");
  const has = (n: string) => !!result?.files.includes(n);
  const views: { value: View; label: string; disabled?: boolean; title?: string }[] = [
    { value: "compare", label: "Compare" }, { value: "overlay", label: "Overlay" }, { value: "mask", label: "Mask" },
    { value: "confidence", label: "Confidence" }, { value: "uncertainty", label: "Uncertainty", disabled: !has("uncertainty.png") },
    { value: "ndwi", label: "vs NDWI", disabled: !has("model_vs_ndwi.png"), title: "Needs Sentinel-2 bands" },
    { value: "truth", label: "vs hand label", disabled: !result?.ground_truth, title: "Only for held-out samples" },
  ];

  return (
    <div>
      <PageHead title="Segmentation studio"
        sub="Every mask on this page is produced by the trained model through the API. Hand labels appear only for held-out samples, in violet, and are marked as reference data."
        actions={<Button variant="secondary" onClick={() => setHelpOpen(true)}><HelpCircle className="h-3.5 w-3.5" />Supported formats</Button>} />

      <div className="grid gap-3 xl:grid-cols-[300px_minmax(0,1fr)_300px]">
        {/* ---------------- input + settings */}
        <div className="min-w-0 space-y-3">
          <Panel title="1 · Input">
            <Segmented label="Input source" value={mode} onChange={setMode} options={[
              { value: "sample", label: <><Satellite className="h-3.5 w-3.5" />Held-out chip</> },
              { value: "upload", label: <><Upload className="h-3.5 w-3.5" />Upload</> }]} />
            <div className="mt-3">
              {mode === "upload" ? (
                <>
                  <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
                    onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
                    onClick={() => inputRef.current?.click()} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
                    role="button" tabIndex={0} aria-label="Choose or drop image files"
                    className={`flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-3 py-6 text-center transition-colors ${
                      drag ? "border-accent-2 bg-accent-soft" : "border-line hover:border-accent-2"}`}>
                    {preview ? <img src={preview} alt="Selected file preview" className="mb-2 max-h-32 rounded object-contain" /> : <FileImage className="mb-2 h-7 w-7 text-ink-3" />}
                    <div className="text-[12.5px] font-medium text-ink">{files.length > 1 ? `${files.length} frames selected` : file ? file.name : "Drop images or click to browse"}</div>
                    <div className="mt-0.5 text-[11px] text-ink-3">{file ? `${(files.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(2)} MB` : `GeoTIFF · PNG · JPEG · ≤ ${maxMb} MB each · up to ${MAX_FRAMES} frames`}</div>
                    <input ref={inputRef} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => pick(e.target.files)} />
                  </div>
                  {file && <button onClick={() => { setFiles([]); setBatch([]); setPreview(null); }} className="mt-1.5 inline-flex items-center gap-1 text-[11.5px] text-ink-2 hover:text-ink"><X className="h-3 w-3" />Clear</button>}
                  <p className="mt-2 text-[11px] leading-relaxed text-ink-3">The band count sets the interpretation (2 = S1, 13 = S2, 15 = S1+S2, 3 = RGB). Only GeoTIFFs keep coordinates.
                    {chip && <> Test file: <a className="text-accent hover:underline" href={`/api/samples/${chip}/download`}>{chip} (15-band GeoTIFF)</a>.</>}</p>
                </>
              ) : (
                <>
                  <div className="mb-2 grid grid-cols-[1fr_92px] gap-1.5">
                    <Select label="Filter by event" value={eventFilter} onChange={setEventFilter}>
                      <option value="all">All events ({samples.data?.samples.length ?? 0})</option>
                      {events.map((e) => <option key={e} value={e}>{e}</option>)}
                    </Select>
                    <Select label="Bands to use" value={sampleInput} onChange={setSampleInput}>
                      <option value="s1s2">S1+S2</option><option value="s2">S2</option><option value="s1">S1</option><option value="rgb">RGB</option>
                    </Select>
                  </div>
                  {samples.error ? <ErrorState error={samples.error} onRetry={samples.reload} compact /> : !samples.data ? <Skeleton className="h-40" /> : (
                    <div className="grid max-h-60 grid-cols-4 gap-1 overflow-y-auto pr-0.5 scroll-thin" role="listbox" aria-label="Held-out chips">
                      {shown.map((s) => (
                        <button key={s.id} role="option" aria-selected={chip === s.id} onClick={() => setChip(s.id)}
                          title={`${s.id} · ${s.split} · ${(s.water_fraction * 100).toFixed(0)}% water (label)`}
                          className={`relative overflow-hidden rounded ring-2 ${chip === s.id ? "ring-accent-2" : "ring-transparent hover:ring-line"}`}>
                          <img src={`/api/samples/${s.id}/preview.png`} alt={s.id} loading="lazy" className="aspect-square w-full object-cover" />
                          <span className="absolute inset-x-0 bottom-0 truncate bg-black/55 px-0.5 text-[8.5px] text-white">{s.event}{s.split === "bolivia" ? " ★" : ""}</span>
                        </button>
                      ))}
                    </div>
                  )}
                  <p className="mt-1.5 text-[11px] text-ink-3">Selected: <span className="font-mono text-ink">{chip || "–"}</span>. Test-split and Bolivia ★ chips, unseen in training.</p>
                </>
              )}
            </div>
          </Panel>

          <Panel title="2 · Model & inference">
            <div className="space-y-3">
              <Field label="Model" hint={mode === "sample" ? `Compatible with ${MODALITY_LABEL[sampleInput]}` : "Auto = default model, or the best compatible one for the uploaded bands"}>
                <Select label="Model" value={model} onChange={setModel}>
                  <option value="">Auto (recommended)</option>
                  {compatible.map((m) => <option key={m.name} value={m.name}>{modelLabel(m)} · {m.name}</option>)}
                </Select>
              </Field>
              <Slider label="Decision threshold" value={threshold} min={0.1} max={0.9} step={0.05} onChange={setThreshold} format={(v) => v.toFixed(2)} info={METRIC_INFO.threshold} />
              <Toggle checked={tta} onChange={setTta} label="Test-time augmentation" hint="Averages 4 flips (about 4× slower)" />
              <Button onClick={run} disabled={!canRun} size="md" className="w-full">
                {busy ? <Spinner label={`${progress ?? "Working"} · ${elapsed.toFixed(1)} s`} /> : <><Play className="h-4 w-4" />{files.length > 1 && mode === "upload" ? `Run on ${files.length} frames` : "Run segmentation"}</>}
              </Button>
              {completed.length === 0 && models.data && <Notice tone="warn">No trained model is available.</Notice>}
            </div>
          </Panel>
        </div>

        {/* ---------------- workspace */}
        <div className="min-w-0 space-y-3">
          {error && <ErrorState error={error} />}
          <Panel title={result ? <span className="flex items-center gap-2 normal-case tracking-normal">
              <span className="font-semibold text-ink">{result.source.chip_id ?? result.source.filename}</span>
              <Badge tone="water">model prediction</Badge>
              {result.input.georeferenced ? <Badge tone="ok"><Globe2 className="h-3 w-3" />{result.input.crs}</Badge> : <Badge tone="warn">no coordinates</Badge>}
            </span> : "Workspace"}
            actions={result && <Segmented size="xs" label="View" value={view} onChange={setView} options={views} />} body="p-3">
            {!result ? (busy ? <div className="flex h-[420px] items-center justify-center"><Spinner label={`${progress ?? "Running"} · ${elapsed.toFixed(1)} s`} /></div> :
              <Empty icon={<Layers className="h-8 w-8" />} title="No prediction yet">Choose a held-out chip or upload an image, then run the model.</Empty>) : (
              <div>
                {view === "compare" && <CompareSlider before={img("input.png")} after={img("overlay.png")} />}
                {view === "overlay" && (
                  <div>
                    <div className="relative mx-auto aspect-square w-full max-w-[640px] overflow-hidden rounded-md bg-nav">
                      <img src={img("input.png")} alt="Input image" className="absolute inset-0 h-full w-full object-contain [image-rendering:pixelated]" />
                      {has("mask_rgba.png") && <img src={img("mask_rgba.png")} alt="Predicted water layer" style={{ opacity }} className="absolute inset-0 h-full w-full object-contain [image-rendering:pixelated]" />}
                    </div>
                    <div className="mx-auto mt-2 max-w-sm"><Slider label="Overlay opacity" value={opacity} min={0} max={1} step={0.05} onChange={setOpacity} format={(v) => `${Math.round(v * 100)}%`} /></div>
                  </div>
                )}
                {view === "mask" && <Img src={img("mask.png")} alt="Binary water mask" />}
                {view === "confidence" && <Img src={img("probability.png")} alt="Model water probability" checker />}
                {view === "uncertainty" && <Img src={img("uncertainty.png")} alt="Decision uncertainty map" />}
                {view === "ndwi" && <Img src={img("model_vs_ndwi.png")} alt="Model versus NDWI agreement" />}
                {view === "truth" && (
                  <div className="grid gap-2 md:grid-cols-2">
                    <figure><Img src={img("ground_truth.png")} alt="Hand label" /><figcaption className="mt-1 text-[11px] text-truth">Hand label: reference, not a model output</figcaption></figure>
                    <figure><Img src={img("errors.png")} alt="Errors versus hand label" /><figcaption className="mt-1 text-[11px] text-ink-2">Model errors vs hand label</figcaption></figure>
                  </div>
                )}
                <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap gap-3 text-[11px] text-ink-2">
                    {view === "confidence" && <Ramp name="viridis" label="P(water); transparent below 0.05" />}
                    {view === "uncertainty" && <Ramp name="magma" label="1 − |2P − 1|; bright = uncertain" />}
                    {LEGENDS[view].map(([c, l]) => <span key={l} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm ring-1 ring-line" style={{ background: c }} />{l}</span>)}
                  </div>
                  <span className="text-[11px] text-ink-3">{result.input.width}×{result.input.height} px · {MODALITY_LABEL[result.model.modality]}</span>
                </div>
              </div>
            )}
          </Panel>

          {batch.length > 0 && (
            <Panel title={`Frame sequence · ${batch.filter((b) => b.result).length}/${batch.length} segmented`} body="p-2">
              <div className="flex gap-1.5 overflow-x-auto pb-1 scroll-thin">
                {batch.map((b, i) => (
                  <button key={i} disabled={!b.result} onClick={() => b.result && setResult(b.result)} title={b.error ?? b.name}
                    className={`w-24 shrink-0 overflow-hidden rounded text-left ring-2 ${b.result && result?.id === b.result.id ? "ring-accent-2" : "ring-line"}`}>
                    {b.result ? <img src={resultFile(b.result.id, "overlay.png")} alt={b.name} className="aspect-square w-full object-cover" />
                      : <div className="flex aspect-square items-center justify-center bg-bad-soft px-1 text-center text-[10px] text-bad">failed</div>}
                    <div className="truncate px-1 py-0.5 text-[10px] text-ink-2">{i + 1}. {b.result ? `${(b.result.stats.water_fraction * 100).toFixed(1)}% · ${Math.round(b.result.timing.total_ms)} ms` : b.name}</div>
                  </button>
                ))}
              </div>
            </Panel>
          )}

          {result && (
            <Panel title="Compare with a previous analysis" info="Analyses are stored in the local history database; pick one to compare side by side."
              actions={<div className="w-64"><Select label="Previous analysis" value={compareId} onChange={setCompareId}>
                <option value="">Select an analysis…</option>
                {(prev.data?.items ?? []).filter((i) => i.id !== result.id && i.available).map((i) => <option key={i.id} value={i.id}>{i.title} · {i.model} · {fmtTime(i.created)}</option>)}
              </Select></div>} body="p-3">
              {!compareId ? <p className="text-[12px] text-ink-3">Choose an analysis to compare its overlay and metrics with the current one.</p> :
                compare.loading ? <Skeleton className="h-40" /> : compare.error ? <ErrorState error={compare.error} /> : compare.data && (
                  <div className="grid gap-3 md:grid-cols-2">
                    {[result, compare.data].map((r, i) => (
                      <div key={r.id}>
                        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">{i === 0 ? "Current" : "Previous"} · {r.source.chip_id ?? r.source.filename}</div>
                        <img src={resultFile(r.id, "overlay.png")} alt={`Overlay ${i === 0 ? "current" : "previous"}`} className="aspect-square w-full rounded object-contain bg-nav [image-rendering:pixelated]" />
                        <KV dense items={[["Model", r.model.name], ["Threshold", r.settings.threshold], ["Water", fmtPct(r.stats.water_fraction, 2)],
                          ["Area", r.area ? `${r.area.water_area_km2.toFixed(3)} km²` : "–"], ["IoU vs label", fmtNum(r.ground_truth?.metrics.iou_water)], ["Latency", fmtMs(r.timing.total_ms)]]} />
                      </div>
                    ))}
                  </div>
                )}
            </Panel>
          )}
        </div>

        {/* ---------------- analysis */}
        <div className="min-w-0 space-y-3">
          {!result ? (
            <Panel title="Analysis"><p className="text-[12px] text-ink-3">Statistics, baseline comparison, latency and exports appear here after a run.</p></Panel>
          ) : (<>
            {result.warnings.map((w) => <Notice key={w} tone="warn">{w}</Notice>)}
            <Panel title="Prediction">
              <KV dense items={[
                ["Water pixels", fmtPct(result.stats.water_fraction, 2)],
                ["Water area", result.area ? `${result.area.water_area_km2.toFixed(3)} km²` : <span className="text-ink-3" title="Needs a CRS and geotransform">needs georeferencing</span>],
                ["Scene area", result.area ? `${result.area.scene_area_km2.toFixed(2)} km²` : "–"],
                ["Pixel size", result.area ? `${result.area.pixel_area_m2.toFixed(1)} m²` : "–"],
                ["Confident (P<0.1 or >0.9)", fmtPct(result.stats.confident_pixels_fraction)],
                ["Uncertain (1−|2P−1| > 0.6)", fmtPct(result.uncertainty?.uncertain_pixels_fraction)],
              ]} />
            </Panel>
            {result.baseline && (
              <Panel title="Model vs NDWI baseline" info="NDWI > 0 from the same Sentinel-2 bands. Agreement is not accuracy; scores against the hand label appear only for held-out samples.">
                <KV dense items={[["NDWI water", fmtPct(result.baseline.water_fraction, 2)], ["Agreement IoU", fmtNum(result.baseline.agreement_iou_with_model)],
                  ["Model only", fmtPct(result.baseline.model_only_fraction, 2)], ["NDWI only", fmtPct(result.baseline.ndwi_only_fraction, 2)]]} />
                {result.ground_truth && result.baseline.metrics_vs_ground_truth && (
                  <table className="mt-2 w-full text-[12px] tnum">
                    <thead className="text-[10.5px] uppercase text-ink-3"><tr><th className="py-1 text-left">vs hand label</th><th className="py-1 text-right">Model</th><th className="py-1 text-right">NDWI</th></tr></thead>
                    <tbody className="divide-y divide-line-2">
                      {(["iou_water", "f1_water", "precision_water", "recall_water"] as const).map((k) => {
                        const a = result.ground_truth!.metrics[k], b = result.baseline!.metrics_vs_ground_truth![k];
                        return <tr key={k}><td className="py-1 text-ink-2">{k.replace("_water", "")}</td>
                          <td className={`py-1 text-right ${a > b ? "font-semibold text-ok" : ""}`}>{fmtNum(a)}</td>
                          <td className={`py-1 text-right ${b > a ? "font-semibold text-ok" : ""}`}>{fmtNum(b)}</td></tr>;
                      })}
                    </tbody>
                  </table>
                )}
              </Panel>
            )}
            {result.ground_truth && !result.baseline && (
              <Panel title="Agreement with hand label"><KV dense items={[["IoU", fmtNum(result.ground_truth.metrics.iou_water)], ["F1", fmtNum(result.ground_truth.metrics.f1_water)],
                ["Valid pixels", fmtInt(result.ground_truth.metrics.valid_pixels)]]} /></Panel>
            )}
            <Panel title="Provenance & latency">
              <KV dense items={[["Model", result.model.name], ["Architecture", modelLabel(result.model)], ["Checkpoint epoch", result.model.checkpoint_epoch],
                ["Threshold / TTA", `${result.settings.threshold} / ${result.settings.tta ? "on" : "off"}`],
                ["Preprocess", fmtMs(result.timing.preprocess_ms)], ["Inference", fmtMs(result.timing.model_ms)], ["Total", fmtMs(result.timing.total_ms)], ["Device", result.timing.device]]} />
              {result.input.notes.length > 0 && <p className="mt-2 text-[11px] text-ink-3">{result.input.notes.join(" · ")}</p>}
            </Panel>
            <Panel title="Export">
              <div className="grid grid-cols-2 gap-1.5">
                <Button size="xs" variant="secondary" href={resultFile(result.id, "overlay.png", true)}><Download className="h-3 w-3" />Overlay PNG</Button>
                <Button size="xs" variant="secondary" href={resultFile(result.id, "mask.png", true)}><Download className="h-3 w-3" />Mask PNG</Button>
                <Button size="xs" variant="secondary" href={resultFile(result.id, "probability.tif", true)}><Download className="h-3 w-3" />P(water) GeoTIFF</Button>
                <Button size="xs" variant="secondary" href={resultFile(result.id, "result.json", true)}><Download className="h-3 w-3" />Metadata JSON</Button>
                <Button size="xs" variant="dark" className="col-span-2" href={reportUrl(result.id)}><FileText className="h-3 w-3" />Download HTML report</Button>
              </div>
            </Panel>
            {result.input.georeferenced && (
              <Link to={`/routes?result=${result.id}${result.source.chip_id ? `&chip=${result.source.chip_id}` : ""}`}
                className="flex items-center justify-center gap-1.5 rounded-md bg-nav px-3 py-2 text-[12.5px] font-medium text-white hover:bg-nav-3">
                <MapPinned className="h-4 w-4" />Use this prediction for routing</Link>
            )}
          </>)}
        </div>
      </div>

      <Drawer open={helpOpen} onClose={() => setHelpOpen(false)} title="Supported input formats">
        {!formats.data ? <Skeleton className="h-40" /> : (<>
          <table className="w-full text-[12px]">
            <thead className="text-left text-[10.5px] uppercase text-ink-3"><tr><th className="py-1">Format</th><th className="py-1">Interpreted as</th><th className="py-1">Models</th><th className="py-1">Coordinates</th></tr></thead>
            <tbody className="divide-y divide-line-2">{formats.data.formats.map((f) => <tr key={f.format}><td className="py-1.5 pr-2 font-medium">{f.format}</td><td className="py-1.5 pr-2">{f.interpretation}</td><td className="py-1.5 pr-2 font-mono text-[11px]">{f.models.join(", ")}</td><td className="py-1.5">{f.georeferenced}</td></tr>)}</tbody>
          </table>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-[12px] text-ink-2">{formats.data.notes.map((n) => <li key={n}>{n}</li>)}</ul>
          <p className="mt-3 text-[12px] text-ink-2">Limits: {formats.data.max_upload_mb} MB per file; {(formats.data.max_pixels / 1e6).toFixed(1)} MP; at least {formats.data.min_side_px} px per side.</p>
        </>)}
      </Drawer>
    </div>
  );
}
