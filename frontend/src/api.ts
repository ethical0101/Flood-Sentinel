// Typed client for the FastAPI backend. All UI data comes from these calls - nothing is mocked.

export class ApiError extends Error {
  status: number;
  code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "network", "Cannot reach the backend. Is the API server running on port 8000?");
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    const b = (body ?? {}) as { message?: string; error?: string; detail?: unknown };
    let msg = b.message ?? (typeof b.detail === "string" ? b.detail : undefined);
    if (!msg && Array.isArray(b.detail)) msg = b.detail.map((d: { msg?: string }) => d.msg).join("; ");
    throw new ApiError(res.status, b.error ?? "http_error", msg ?? `Request failed (${res.status})`);
  }
  return body as T;
}

const post = <T,>(path: string, json: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(json) });

// ------------------------------------------------------------------ types
export interface Metrics {
  iou_water: number;
  iou_not_water: number;
  miou: number;
  dice_water: number;
  f1_water: number;
  precision_water: number;
  recall_water: number;
  pixel_accuracy: number;
  omission_error: number;
  commission_error: number;
  valid_pixels: number;
  water_fraction_gt: number;
  confusion_matrix: number[][];
  mean_chip_iou_water?: number;
  n_chips?: number;
}

export interface ModelInfo {
  name: string;
  modality: string;
  arch: string;
  encoder: string;
  training_state: string;
  best_val_iou: number | null;
  best_epoch: number | null;
  evaluated: boolean;
  test_iou: number | null;
  bolivia_iou: number | null;
  n_params: number | null;
  checkpoint_mb: number;
  loaded: boolean;
}

export interface ModelStatus {
  ready: boolean;
  default_model: string;
  device_setting: string;
  cuda_available: boolean;
  loaded: string[];
  models: ModelInfo[];
}

export interface GpuLive {
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  utilization_pct: number;
  temperature_c: number;
  driver: string;
  power_w?: number;
}

export interface SystemInfo {
  os: string;
  python: string;
  cpu: { logical_cores: number; physical_cores: number; utilization_pct: number };
  ram: { total_gib: number; available_gib: number; used_pct: number };
  disk: { drive: string; total_gb: number; free_gb: number };
  torch: { version: string; cuda_build: string | null; cuda_available: boolean };
  gpu: GpuLive | null;
}

export interface SplitInfo {
  chips: number;
  events: string[];
  mean_water_fraction: number;
}

export interface DatasetStatus {
  name: string;
  source: string;
  license: string;
  citation: string;
  downloaded: boolean;
  downloaded_utc: string | null;
  n_files: number | null;
  download_bytes: number | null;
  raw_bytes_on_disk: number;
  processed_bytes_on_disk: number;
  verification: {
    passed: boolean;
    integrity_mode: string;
    files_checked: number;
    files_ok: number;
    unique_chips: number;
    crs: Record<string, number>;
    label_pixel_counts: Record<string, number>;
    errors: string[];
  } | null;
  processed: boolean;
  splits: Record<string, SplitInfo>;
  train_water_fraction: number | null;
  layers: string[];
}

export interface HistoryRow {
  epoch: number;
  train_loss: number;
  lr: number;
  epoch_time_s: number;
  peak_gpu_mem_mib: number;
  val_loss: number;
  val_iou_water: number;
  val_f1_water: number;
  val_precision_water: number;
  val_recall_water: number;
  val_miou: number;
}

export interface TrainingRun {
  name: string;
  state: string;
  epoch: number;
  epochs: number;
  best: number;
  best_epoch?: number;
  monitor: string;
  started?: string;
  finished?: string;
  total_time_s?: number;
  eta_s?: number;
  note?: string;
  error?: string;
  history: HistoryRow[];
  checkpoints: { file: string; mb: number; modified: string }[];
  resume_command: string;
}

export interface EvaluationRun {
  run: string;
  checkpoint_epoch: number;
  threshold: number;
  modality: string;
  config: { model: { arch: string; encoder: string; encoder_weights: string | null }; data: Record<string, unknown>; train: Record<string, unknown>; seed: number };
  meta: { n_params: number; pretrained: string; trained_on: string; selected_on: string };
  splits: Record<string, Metrics>;
  per_event: Record<string, Record<string, Metrics>>;
  latency: {
    input: string;
    device: string;
    gpu: string | null;
    precision: string;
    model_only_ms_mean: number;
    model_only_ms_p95: number;
    end_to_end_ms_mean: number;
    end_to_end_includes: string;
    throughput_fps_model_only: number;
  };
  history: HistoryRow[];
  sample_images: string[];
}

export interface Baseline {
  description: string;
  splits: Record<string, Metrics>;
}

export interface RoutingTriple { shortest: number; risk_aware_model: number; risk_aware_oracle: number }
export interface RoutingEvaluation {
  protocol: { pairs_per_area: number; seed: number; min_straight_line_m: number; note: string };
  areas: Record<string, {
    edges: number; flooded_road_km_model: number; flooded_road_km_label: number;
    edge_agreement: { accuracy: number; precision: number; recall: number; tp: number; fp: number; fn: number; tn: number };
    n_pairs: number; n_pairs_shortest_route_crosses_labelled_flood: number;
    mean_gt_flooded_m: RoutingTriple; mean_length_m: RoutingTriple;
  }>;
  overall: {
    n_pairs: number; mean_gt_flooded_m: RoutingTriple; mean_length_m: RoutingTriple;
    reduction_in_gt_flooded_length_vs_shortest: number; mean_detour_pct: number;
    pairs_where_model_route_has_more_gt_flood_than_shortest: number;
  };
}

export interface MetricsResponse {
  summary: { protocol: string } | null;
  baselines: Record<string, Baseline> | null;
  runs: Record<string, EvaluationRun>;
  routing: RoutingEvaluation | null;
}

export interface SegResult {
  id: string;
  created: string;
  kind: string;
  source: { type: string; filename?: string; chip_id?: string; split?: string; event?: string };
  model: { name: string; modality: string; arch: string; encoder: string; checkpoint_epoch: number; trained_on: string };
  input: {
    kind: string;
    width: number;
    height: number;
    georeferenced: boolean;
    crs: string | null;
    bounds: number[] | null;
    notes: string[];
    available_modalities: string[];
  };
  settings: { threshold: number; tta: boolean };
  stats: { water_fraction: number; mean_probability: number; confident_pixels_fraction: number };
  area: { pixel_area_m2: number; water_area_km2: number; scene_area_km2: number; method: string } | null;
  timing: { preprocess_ms: number; model_ms: number; queue_ms: number; total_ms: number; device: string; tta: boolean };
  files: string[];
  ground_truth: { label: string; metrics: Metrics } | null;
  warnings: string[];
}

export interface Sample {
  id: string;
  event: string;
  split: string;
  water_fraction: number;
  valid_fraction: number;
  bounds: number[];
}

export interface Area {
  id: string;
  kind: "curated" | "upload";
  chip_id?: string;
  name: string;
  region?: string;
  event?: string;
  description: string;
  bounds: [number, number, number, number];
  split?: string;
  sources: string[];
  default_start?: [number, number];
  default_end?: [number, number];
  road_graph_cached: boolean;
  road_meta: { fetched_utc: string; nodes: number; edges: number; source: string } | null;
  model?: string;
}

export type RiskClass = "clear" | "low" | "partial" | "flooded" | "unknown";

export interface RoadFeature {
  type: "Feature";
  geometry: { type: "LineString"; coordinates: [number, number][] };
  properties: { risk_class: RiskClass; flooded_len_m: number; length_m: number; flood_prob_max: number | null; highway: string; name: string | null };
}

export interface AreaDetail {
  area: Area;
  layer: { source: string; description: string; bounds_wgs84: [number, number, number, number] };
  risk_stats: { edges: number; risk_class_counts: Record<RiskClass, number>; flooded_road_km: number; threshold: number; step_m: number; buffer_px: number };
  alignment: { graph_fraction_covered: number; raster_crs: string; graph_crs: string; ok: boolean };
  roads: { type: "FeatureCollection"; features: RoadFeature[] };
  attribution: string;
  disclaimer: string;
}

export interface Route {
  distance_m: number;
  travel_time_s: number | null;
  flooded_length_m: number;
  flooded_share: number;
  unknown_length_m: number;
  max_flood_prob: number;
  n_flood_affected_segments: number;
  n_segments: number;
  coordinates: [number, number][];
  segments: { coords: [number, number][]; risk_class: RiskClass; flooded_len_m: number; length_m: number; name: string | null; highway: string }[];
  cost?: number;
}

export interface PlanResponse {
  start_snap_m: number;
  destination_snap_m: number;
  baseline: Route;
  risk_aware: Route;
  alternatives: Route[];
  same_as_baseline: boolean;
  closure_fallback: boolean;
  closed_edges: number;
  warnings: string[];
  explanation: string[];
  flood_layer: { source: string; description: string; is_model_prediction: boolean; threshold: number; buffer_px: number };
  compute_ms: number;
  attribution: string;
  disclaimer: string;
  travel_time_note: string;
}

export interface PlanRequest {
  area_id: string;
  source: "model" | "ground_truth";
  model?: string | null;
  start: { lat: number; lon: number };
  end: { lat: number; lon: number };
  threshold: number;
  buffer_px: number;
  flood_penalty: number;
  unknown_penalty: number;
  close_flooded: boolean;
  closure_min_flooded_m: number;
  alternatives: number;
}

export interface RecentItem {
  id: string;
  created: string;
  source: SegResult["source"];
  stats: SegResult["stats"];
  model: string;
  total_ms: number;
}

// ------------------------------------------------------------------ calls
export const api = {
  health: () => request<{ status: string; uptime_s: number; version: string }>("/api/health"),
  system: () => request<SystemInfo>("/api/system"),
  modelStatus: () => request<ModelStatus>("/api/model/status"),
  datasets: () => request<DatasetStatus>("/api/datasets/status"),
  training: () => request<{ runs: TrainingRun[]; gpu: GpuLive | null }>("/api/training/status"),
  trainingLog: (run: string, lines = 200) =>
    request<{ run: string; lines: string[] }>(`/api/training/runs/${encodeURIComponent(run)}/log?lines=${lines}`),
  metrics: () => request<MetricsResponse>("/api/metrics"),
  samples: () => request<{ samples: Sample[] }>("/api/samples"),
  recent: () => request<{ recent: RecentItem[] }>("/api/segmentation/recent"),
  predictUpload: (file: File, opts: { model?: string; threshold: number; tta: boolean }) => {
    const fd = new FormData();
    fd.append("file", file);
    if (opts.model) fd.append("model", opts.model);
    fd.append("threshold", String(opts.threshold));
    fd.append("tta", String(opts.tta));
    return request<SegResult>("/api/segmentation/predict", { method: "POST", body: fd });
  },
  predictSample: (body: { chip_id: string; model?: string; threshold: number; tta: boolean; input: string }) =>
    post<SegResult>("/api/segmentation/predict-sample", body),
  result: (id: string) => request<SegResult>(`/api/segmentation/results/${id}`),
  areas: () => request<{ areas: Area[]; attribution: string; disclaimer: string }>("/api/routes/areas"),
  area: (id: string, q: { source: string; threshold: number; buffer_px: number; model?: string }) =>
    request<AreaDetail>(
      `/api/routes/areas/${encodeURIComponent(id)}?source=${q.source}&threshold=${q.threshold}&buffer_px=${q.buffer_px}` +
        (q.model ? `&model=${encodeURIComponent(q.model)}` : ""),
    ),
  plan: (body: PlanRequest) => post<PlanResponse>("/api/routes/plan", body),
};

export const resultFile = (id: string, name: string, download = false) =>
  `/api/segmentation/results/${id}/files/${name}${download ? "?download=true" : ""}`;
export const runImage = (run: string, name: string) =>
  `/api/metrics/runs/${encodeURIComponent(run)}/images/${encodeURIComponent(name)}`;
export const overlayUrl = (id: string, source: string, threshold: number, model?: string) =>
  `/api/routes/areas/${encodeURIComponent(id)}/overlay.png?source=${source}&threshold=${threshold}` +
  (model ? `&model=${encodeURIComponent(model)}` : "");
export const imageryUrl = (id: string) => `/api/routes/areas/${encodeURIComponent(id)}/imagery.png`;
