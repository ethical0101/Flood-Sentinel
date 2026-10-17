import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Download, FileImage, Globe2, ImageUp, Layers, MapPinned, Play, Satellite, Timer, X } from "lucide-react";
import { api, resultFile, type SegResult } from "../api";
import { Badge, Button, Callout, Card, EmptyState, ErrorState, KV, PageHeader, Select, Slider, Spinner, Toggle } from "../components/ui";
import { MODALITY_LABEL, fmtInt, fmtMs, fmtNum, fmtPct, modelLabel, useApi } from "../lib";

type View = "compare" | "overlay" | "probability" | "mask" | "truth";
const ACCEPT = ".tif,.tiff,.png,.jpg,.jpeg";

function CompareSlider({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="relative aspect-square w-full select-none overflow-hidden rounded-lg bg-slate-900">
      <img src={before} alt="Input image" className="absolute inset-0 h-full w-full object-contain [image-rendering:pixelated]" />
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img src={after} alt="Model prediction overlay" className="absolute inset-0 h-full w-full object-contain [image-rendering:pixelated]" />
      </div>
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.6)]" style={{ left: `${pos}%` }} />
      <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white">Input</span>
      <span className="absolute right-2 top-2 rounded bg-water-700/90 px-1.5 py-0.5 text-[10px] font-medium text-white">Model prediction</span>
      <input type="range" min={0} max={100} value={pos} onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Comparison position" className="absolute inset-x-0 bottom-2 mx-auto w-2/3 accent-white" />
    </div>
  );
}

function Legend({ view }: { view: View }) {
  if (view === "probability")
    return (
      <div className="flex items-center gap-2 text-[11px] text-slate-600">
        <span>P(water) 0</span>
        <span className="h-2 w-40 rounded" style={{ background: "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)" }} />
        <span>1</span><span className="text-slate-400">· transparent below 0.05</span>
      </div>
    );
  const items = view === "truth"
    ? [["#2563eb", "True positive"], ["#ef4444", "False positive"], ["#f59e0b", "False negative"], ["#94a3b8", "No label (ignored)"]]
    : view === "mask" ? [["#ffffff", "Water (predicted)"], ["#000000", "Not water"]]
    : [["#008cff", "Water (predicted, above threshold)"]];
  return (
    <div className="flex flex-wrap gap-3 text-[11px] text-slate-600">
      {items.map(([c, l]) => (
        <span key={l} className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm ring-1 ring-slate-300" style={{ background: c }} />{l}</span>
      ))}
    </div>
  );
}

export default function Segmentation() {
  const models = useApi(() => api.modelStatus(), []);
  const samples = useApi(() => api.samples(), []);
  const [mode, setMode] = useState<"upload" | "sample">("sample");
  const [files, setFiles] = useState<File[]>([]);
  const file = files[0] ?? null;
  const [batch, setBatch] = useState<{ name: string; result?: SegResult; error?: string }[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [chip, setChip] = useState<string>("");
  const [sampleInput, setSampleInput] = useState("s1s2");
  const [eventFilter, setEventFilter] = useState("all");
  const [model, setModel] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [tta, setTta] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<SegResult | null>(null);
  const [view, setView] = useState<View>("compare");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!chip && samples.data?.samples.length) {
      const s = samples.data.samples.find((x) => x.id === "Spain_7370579") ?? samples.data.samples[0];
      setChip(s.id);
    }
  }, [samples.data, chip]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const events = useMemo(() => Array.from(new Set(samples.data?.samples.map((s) => s.event) ?? [])).sort(), [samples.data]);
  const shownSamples = (samples.data?.samples ?? []).filter((s) => eventFilter === "all" || s.event === eventFilter);
  const completed = (models.data?.models ?? []).filter((m) => m.training_state === "completed");
  const compatible = mode === "sample"
    ? completed.filter((m) => ({ s1s2: ["s1s2", "s2", "s1", "rgb"], s2: ["s2", "rgb"], s1: ["s1"], rgb: ["rgb"] } as Record<string, string[]>)[sampleInput].includes(m.modality))
    : completed;

  useEffect(() => { if (model && !compatible.some((m) => m.name === model)) setModel(""); }, [compatible, model]);

  const MAX_FRAMES = 20;
  const pick = (list: FileList | File[] | null | undefined) => {
    setError(null);
    const all = Array.from(list ?? []);
    if (!all.length) return;
    const ok = all.filter((f) => /\.(tiff?|png|jpe?g)$/i.test(f.name)).slice(0, MAX_FRAMES);
    if (!ok.length) { setError(new Error("Unsupported file type. Use GeoTIFF, PNG or JPEG.")); return; }
    if (ok.length < all.length) setError(new Error(`Skipped ${all.length - ok.length} file(s): unsupported type or more than ${MAX_FRAMES} frames.`));
    setFiles(ok);
    setBatch([]);
    setPreview(/\.(png|jpe?g)$/i.test(ok[0].name) ? URL.createObjectURL(ok[0]) : null);
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      if (mode === "upload" && files.length > 1) {
        // sequence of frames: repeated inference with the already-loaded model, one request per frame
        const out: { name: string; result?: SegResult; error?: string }[] = [];
        setBatch([]);
        for (let i = 0; i < files.length; i++) {
          setProgress(`Frame ${i + 1} / ${files.length}`);
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

  return (
    <div>
      <PageHeader eyebrow="Capability A" title="Flood segmentation studio">
        Run the trained models on a held-out Sentinel chip or your own file. Each output on this page comes from the
        model; ground-truth labels only appear for held-out samples, and are marked as such.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
        {/* ------------------------------------------------------------ controls */}
        <div className="min-w-0 space-y-4">
          <Card title="1 · Input" bodyClass="p-4">
            <div className="mb-4 grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1 text-[13px] font-medium" role="tablist">
              {(["sample", "upload"] as const).map((m) => (
                <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)}
                  className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 ${mode === m ? "bg-white text-slate-900 shadow-sm" : "text-slate-600"}`}>
                  {m === "sample" ? <><Satellite className="h-3.5 w-3.5" />Held-out sample</> : <><ImageUp className="h-3.5 w-3.5" />Upload file</>}
                </button>
              ))}
            </div>

            {mode === "upload" ? (
              <div>
                <div
                  onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
                  onClick={() => inputRef.current?.click()}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
                  role="button" tabIndex={0} aria-label="Upload image"
                  className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-7 text-center transition-colors ${
                    drag ? "border-water-500 bg-water-50" : "border-slate-300 hover:border-water-500 hover:bg-slate-50"}`}>
                  {preview ? <img src={preview} alt="Selected file preview" className="mb-2 max-h-40 rounded object-contain" /> :
                    <FileImage className="mb-2 h-8 w-8 text-slate-400" />}
                  <div className="text-[13px] font-medium text-slate-700">{files.length > 1 ? `${files.length} frames selected` : file ? file.name : "Drop images or click to browse"}</div>
                  <div className="mt-1 text-xs text-slate-500">{file ? `${(files.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(2)} MB` : `GeoTIFF (2 / 13 / 15 bands or RGB) · PNG · JPEG · max 40 MB each · up to ${MAX_FRAMES} frames`}</div>
                  <input ref={inputRef} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => pick(e.target.files)} />
                </div>
                {file && <button onClick={() => { setFiles([]); setBatch([]); setPreview(null); }} className="mt-2 inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800"><X className="h-3 w-3" />Clear</button>}
                <div className="mt-3 space-y-1.5 text-xs leading-relaxed text-slate-500">
                  <p><b className="text-slate-700">GeoTIFF</b>: 2 bands = Sentinel-1 VV/VH (dB), 13 = Sentinel-2 L1C, 15 = both. CRS and transform are kept, so results can be routed.</p>
                  <p><b className="text-slate-700">PNG/JPEG</b>: treated as RGB and segmented by the RGB model. They have no coordinates, so they cannot be used for routing.</p>
                  {chip && <p>Need a test file? <a className="text-water-600 hover:underline" href={`/api/samples/${chip}/download`}>Download {chip} as a 15-band GeoTIFF</a>.</p>}
                </div>
              </div>
            ) : (
              <div>
                <div className="mb-2 flex gap-2">
                  <select value={eventFilter} onChange={(e) => setEventFilter(e.target.value)} aria-label="Filter by event"
                    className="flex-1 rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-[13px]">
                    <option value="all">All events ({samples.data?.samples.length ?? 0})</option>
                    {events.map((e) => <option key={e} value={e}>{e}</option>)}
                  </select>
                  <select value={sampleInput} onChange={(e) => setSampleInput(e.target.value)} aria-label="Bands to use"
                    className="w-28 rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-[13px]">
                    <option value="s1s2">S1+S2</option><option value="s2">S2 only</option><option value="s1">S1 only</option><option value="rgb">RGB only</option>
                  </select>
                </div>
                {samples.error ? <ErrorState error={samples.error} onRetry={samples.reload} /> : !samples.data ? <Spinner /> : (
                  <div className="grid max-h-72 grid-cols-4 gap-1.5 overflow-y-auto pr-1">
                    {shownSamples.map((s) => (
                      <button key={s.id} onClick={() => setChip(s.id)} title={`${s.id} · ${s.split} · ${(s.water_fraction * 100).toFixed(0)}% water (label)`}
                        className={`group relative overflow-hidden rounded-md ring-2 transition ${chip === s.id ? "ring-water-600" : "ring-transparent hover:ring-slate-300"}`}>
                        <img src={`/api/samples/${s.id}/preview.png`} alt={s.id} loading="lazy" className="aspect-square w-full object-cover" />
                        <span className="absolute inset-x-0 bottom-0 truncate bg-black/55 px-1 text-[9px] text-white">{s.event}{s.split === "bolivia" ? " ★" : ""}</span>
                      </button>
                    ))}
                  </div>
                )}
                <p className="mt-2 text-xs text-slate-500">Chips from the official <b>test</b> split and the <b>Bolivia ★</b> event, which no model saw during training or checkpoint selection.</p>
              </div>
            )}
          </Card>

          <Card title="2 · Inference settings" bodyClass="space-y-4 p-4">
            <Select label="Model" value={model} onChange={setModel}
              hint={mode === "upload" ? "Auto picks the default model, or the best one for the uploaded bands." : `Compatible with ${MODALITY_LABEL[sampleInput]}`}>
              <option value="">Auto (recommended)</option>
              {compatible.map((m) => <option key={m.name} value={m.name}>{modelLabel(m)}</option>)}
            </Select>
            <Slider label="Decision threshold" value={threshold} min={0.1} max={0.9} step={0.05} onChange={setThreshold}
              format={(v) => v.toFixed(2)} hint="Pixels with P(water) above this are mapped as water. Models were evaluated at 0.50." />
            <Toggle checked={tta} onChange={setTta} label="Test-time augmentation" hint="Averages 4 flipped predictions (about 4× slower)." />
            <Button onClick={run} disabled={!canRun} className="w-full">
              {busy ? <Spinner label={progress ?? "Running model…"} /> : <><Play className="h-4 w-4" />{files.length > 1 && mode === "upload" ? `Run on ${files.length} frames` : "Run segmentation"}</>}
            </Button>
            {completed.length === 0 && models.data && <Callout tone="warn">No trained model is available yet. Train one first (Data & Training page).</Callout>}
          </Card>
        </div>

        {/* ------------------------------------------------------------ results */}
        <div className="min-w-0 space-y-4">
          {error && <ErrorState error={error} />}
          {!result && !busy && !error && (
            <Card><EmptyState icon={<Layers className="h-9 w-9" />} title="No prediction yet">
              Choose a held-out chip or upload an image, then run the model. The overlay, probability map and exportable mask will appear here.
            </EmptyState></Card>
          )}
          {busy && !result && <Card><div className="flex h-80 items-center justify-center"><Spinner label="Running model on GPU…" /></div></Card>}
          {batch.length > 0 && (
            <Card title="Frame sequence" subtitle={`${batch.filter((b) => b.result).length} of ${batch.length} frames segmented · click a frame to inspect it`} bodyClass="p-3">
              <div className="flex gap-2 overflow-x-auto pb-1">
                {batch.map((b, i) => (
                  <button key={i} disabled={!b.result} onClick={() => b.result && setResult(b.result)} title={b.error ?? b.name}
                    className={`w-24 shrink-0 overflow-hidden rounded-md text-left ring-2 ${b.result && result?.id === b.result.id ? "ring-water-600" : "ring-slate-200"}`}>
                    {b.result ? <img src={resultFile(b.result.id, "overlay.png")} alt={b.name} className="aspect-square w-full object-cover" />
                      : <div className="flex aspect-square items-center justify-center bg-red-50 px-1 text-center text-[10px] text-red-700">failed</div>}
                    <div className="truncate px-1 py-0.5 text-[10px] text-slate-600">{i + 1}. {b.result ? `${(b.result.stats.water_fraction * 100).toFixed(1)}% · ${Math.round(b.result.timing.total_ms)} ms` : b.name}</div>
                  </button>
                ))}
              </div>
            </Card>
          )}
          {result && (
            <>
              <Card bodyClass="p-4"
                title={<span className="flex flex-wrap items-center gap-2">Model prediction <Badge color="blue">{result.model.name}</Badge>
                  {result.input.georeferenced ? <Badge color="green"><Globe2 className="h-3 w-3" />georeferenced</Badge> : <Badge color="amber">no coordinates</Badge>}</span>}
                subtitle={`${result.source.chip_id ?? result.source.filename} · ${result.input.width}×${result.input.height} px · ${MODALITY_LABEL[result.model.modality]}`}
                action={
                  <div className="flex flex-wrap gap-1.5">
                    <a href={resultFile(result.id, "overlay.png", true)} className="inline-flex items-center gap-1 rounded-md bg-white px-2 py-1 text-xs font-medium ring-1 ring-slate-300 hover:bg-slate-50"><Download className="h-3 w-3" />Overlay</a>
                    <a href={resultFile(result.id, "mask.png", true)} className="inline-flex items-center gap-1 rounded-md bg-white px-2 py-1 text-xs font-medium ring-1 ring-slate-300 hover:bg-slate-50"><Download className="h-3 w-3" />Mask</a>
                    <a href={resultFile(result.id, "probability.tif", true)} className="inline-flex items-center gap-1 rounded-md bg-white px-2 py-1 text-xs font-medium ring-1 ring-slate-300 hover:bg-slate-50"><Download className="h-3 w-3" />Probability GeoTIFF</a>
                  </div>}>
                <div className="mb-3 flex flex-wrap gap-1 text-xs font-medium" role="tablist">
                  {([["compare", "Compare"], ["overlay", "Overlay"], ["probability", "Probability"], ["mask", "Mask"],
                    ...(result.ground_truth ? [["truth", "vs. ground truth"]] : [])] as [View, string][]).map(([v, l]) => (
                    <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)}
                      className={`rounded-md px-2.5 py-1 ${view === v ? "bg-ink-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`}>{l}</button>
                  ))}
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  {view === "compare" && <div className="lg:col-span-2 mx-auto w-full max-w-2xl"><CompareSlider before={img("input.png")} after={img("overlay.png")} /></div>}
                  {view === "overlay" && <><Figure src={img("input.png")} caption="Input (display composite)" /><Figure src={img("overlay.png")} caption="Prediction overlay" /></>}
                  {view === "probability" && <><Figure src={img("input.png")} caption="Input" /><Figure src={img("probability.png")} caption="P(water) — model confidence" checker /></>}
                  {view === "mask" && <><Figure src={img("input.png")} caption="Input" /><Figure src={img("mask.png")} caption={`Binary mask (threshold ${result.settings.threshold})`} /></>}
                  {view === "truth" && result.ground_truth && <><Figure src={img("ground_truth.png")} caption="Ground truth: Sen1Floods11 hand label (NOT a model output)" /><Figure src={img("errors.png")} caption="Prediction errors vs. hand label" /></>}
                </div>
                <div className="mt-3"><Legend view={view} /></div>
              </Card>

              {result.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}

              <div className="grid gap-4 lg:grid-cols-3">
                <Card title="Prediction summary" bodyClass="p-4">
                  <KV items={[
                    ["Water pixels", fmtPct(result.stats.water_fraction, 2)],
                    ["Water area", result.area ? `${result.area.water_area_km2.toFixed(3)} km²` : "needs georeferencing"],
                    ["Scene area", result.area ? `${result.area.scene_area_km2.toFixed(2)} km²` : "–"],
                    ["Pixel size", result.area ? `${result.area.pixel_area_m2.toFixed(1)} m²` : "–"],
                    ["Confident pixels", fmtPct(result.stats.confident_pixels_fraction)],
                  ]} />
                  <p className="mt-2 text-[11px] text-slate-500">Confident = P &lt; 0.1 or P &gt; 0.9. Area = pixel count × geodesic pixel area.</p>
                </Card>
                <Card title={<span className="flex items-center gap-1.5"><Timer className="h-3.5 w-3.5" />Latency</span>} bodyClass="p-4">
                  <KV items={[
                    ["Preprocessing", fmtMs(result.timing.preprocess_ms)],
                    ["Model (tiled)", fmtMs(result.timing.model_ms)],
                    ["Total request", fmtMs(result.timing.total_ms)],
                    ["Device", result.timing.device],
                    ["TTA", result.timing.tta ? "on (4 passes)" : "off"],
                  ]} />
                  <p className="mt-2 text-[11px] text-slate-500">The first request after start-up includes loading the model.</p>
                </Card>
                {result.ground_truth ? (
                  <Card title="Agreement with hand label" bodyClass="p-4" subtitle="This chip only (not the overall test score)">
                    <KV items={[
                      ["IoU (water)", fmtNum(result.ground_truth.metrics.iou_water)],
                      ["F1 / Dice", fmtNum(result.ground_truth.metrics.f1_water)],
                      ["Precision", fmtNum(result.ground_truth.metrics.precision_water)],
                      ["Recall", fmtNum(result.ground_truth.metrics.recall_water)],
                      ["Valid pixels", fmtInt(result.ground_truth.metrics.valid_pixels)],
                    ]} />
                  </Card>
                ) : (
                  <Card title="Model" bodyClass="p-4">
                    <KV items={[["Architecture", modelLabel(result.model)], ["Checkpoint epoch", result.model.checkpoint_epoch],
                      ["Trained on", "Sen1Floods11 train split"], ["Input kind", result.input.kind.toUpperCase()]]} />
                  </Card>
                )}
              </div>

              {result.input.notes.length > 0 && (
                <Callout title="Input interpretation">{result.input.notes.join(" · ")}</Callout>
              )}
              {result.input.georeferenced ? (
                <Card bodyClass="flex flex-wrap items-center justify-between gap-3 p-4">
                  <div className="text-[13px] text-slate-700"><b>This prediction is georeferenced</b> ({result.input.crs}). It can be laid over OpenStreetMap roads for risk-aware routing.</div>
                  <Link to={`/routes?result=${result.id}${result.source.chip_id ? `&chip=${result.source.chip_id}` : ""}`}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-ink-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-ink-800">
                    <MapPinned className="h-3.5 w-3.5" />Open in route planner</Link>
                </Card>
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Figure({ src, caption, checker }: { src: string; caption: string; checker?: boolean }) {
  return (
    <figure>
      <div className={`overflow-hidden rounded-lg ${checker ? "bg-[repeating-conic-gradient(#e2e8f0_0_25%,#f8fafc_0_50%)] bg-[length:16px_16px]" : "bg-slate-900"}`}>
        <img src={src} alt={caption} className="aspect-square w-full object-contain [image-rendering:pixelated]" />
      </div>
      <figcaption className="mt-1.5 text-xs text-slate-500">{caption}</figcaption>
    </figure>
  );
}
