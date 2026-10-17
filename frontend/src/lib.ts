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
export const fmtDist = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);
export const fmtDuration = (s: number | null | undefined) => {
  if (s == null) return "–";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${Math.round(s % 60)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
};
export const fmtMs = (ms: number | null | undefined) => (ms == null ? "–" : ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`);

export const MODALITY_LABEL: Record<string, string> = {
  s1s2: "Sentinel-1 + Sentinel-2 (15 bands)",
  s2: "Sentinel-2 optical (13 bands)",
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
  low: "Elevated probability (< threshold)",
  partial: "Partly flooded (< 50% of length)",
  flooded: "Mostly flooded (≥ 50% of length)",
  unknown: "No flood data",
};

/** Fetch with loading / error state and optional polling. */
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
