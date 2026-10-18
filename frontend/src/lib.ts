import { useCallback, useEffect, useRef, useState } from "react";

export const fmtPct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
export const fmtNum = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "–" : v.toFixed(digits);
export const fmtInt = (v: number | null | undefined) => (v == null ? "–" : v.toLocaleString());
export const fmtBytes = (b: number | null | undefined) => {
  if (b == null) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = b;
  while (v >= 1000 && i < u.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
};
export const fmtDist = (m: number | null | undefined) =>
  m == null ? "–" : m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`;
export const fmtDuration = (s: number | null | undefined) => {
  if (s == null) return "–";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${Math.round(s % 60)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
};
export const fmtMs = (ms: number | null | undefined) => (ms == null ? "–" : ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`);
export const fmtTime = (iso: string | null | undefined) => {
  if (!iso) return "–";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

export const MODALITY_LABEL: Record<string, string> = {
  s1s2: "Sentinel-1 + Sentinel-2 (15 bands)",
  s1s2i: "S1 + S2 + NDWI/MNDWI (17 bands)",
  s2: "Sentinel-2 optical (13 bands)",
  s2i: "S2 + NDWI/MNDWI (15 bands)",
  s1: "Sentinel-1 SAR (VV, VH)",
  rgb: "True-colour RGB",
};
export const ARCH_LABEL: Record<string, string> = { unet: "U-Net", deeplabv3plus: "DeepLabV3+" };
export const encoderLabel = (e: string) =>
  e.replace("tu-mobilenetv3_large_100", "MobileNetV3-L").replace("resnet34", "ResNet-34");
export const modelLabel = (m: { arch: string; encoder: string; modality: string }) =>
  `${ARCH_LABEL[m.arch] ?? m.arch} · ${encoderLabel(m.encoder)} · ${m.modality.toUpperCase()}`;

export const RISK_COLORS: Record<string, string> = {
  clear: "#16a34a",
  low: "#84cc16",
  partial: "#f59e0b",
  flooded: "#dc2626",
  unknown: "#94a3b8",
};
export const RISK_LABEL: Record<string, string> = {
  clear: "No flooding predicted",
  low: "Elevated P(water), below threshold",
  partial: "Partly flooded (< 50% of length)",
  flooded: "Mostly flooded (≥ 50% of length)",
  unknown: "No flood data",
};

export const METRIC_INFO: Record<string, string> = {
  iou: "Intersection over union of the water class: TP / (TP + FP + FN), pooled over all valid pixels of the split. Pixels labelled −1 are ignored.",
  f1: "F1 = Dice for binary masks: 2TP / (2TP + FP + FN).",
  precision: "Of the pixels predicted as water, the share that are water in the label.",
  recall: "Of the labelled water pixels, the share that were predicted as water.",
  miou: "Mean of water IoU and not-water IoU.",
  threshold: "Decision threshold on P(water). 'val' = chosen to maximise IoU on the validation split only, then applied unchanged to held-out data.",
  bolivia: "The 15-chip Bolivia event: never used for training, checkpoint selection or threshold selection; ≥1,200 km from any training chip.",
  test: "Official Sen1Floods11 test split. Same events and dates as training; 42% of chips lie within 10 km of a training chip, so it measures in-distribution performance.",
  worldfloods: "WorldFloods v2 test split: 18 Sentinel-2 scenes from Copernicus EMS activations, labelled by a different team. Evaluation only. Clear-sky valid pixels.",
  latency: "Model forward pass on 1×C×512×512, fp16 autocast, batch 1, CUDA-synchronised, RTX 3050 Laptop GPU.",
};

/** CSV from header + rows (RFC 4180 quoting). */
export function toCsv(header: string[], rows: unknown[][]): string {
  const q = (v: unknown) => {
    if (v == null) return "";
    const s = typeof v === "number" ? String(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header.map(q).join(","), ...rows.map((r) => r.map(q).join(","))].join("\n");
}

export function downloadText(name: string, text: string, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Fetch with loading / error state and optional polling (paused while the tab is hidden). */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = [], pollMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const d = await fnRef.current();
      setData(d);
      setError(null);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    if (!pollMs) return;
    const t = setInterval(() => {
      if (document.visibilityState === "visible") load(true);
    }, pollMs);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, reload: () => load() };
}
