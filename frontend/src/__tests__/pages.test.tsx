// Page-level tests with a mocked backend: numbers must come from API responses, errors must surface.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Command from "../pages/Command";
import History from "../pages/History";
import Studio from "../pages/Studio";

const metrics = (iou: number) => ({ iou_water: iou, iou_not_water: 0.9, miou: 0.85, dice_water: 0.9, f1_water: 0.9, precision_water: 0.9, recall_water: 0.8,
  pixel_accuracy: 0.97, omission_error: 0.1, commission_error: 0.1, valid_pixels: 100, water_fraction_gt: 0.1, confusion_matrix: [[80, 2], [3, 15]] });
const model = { name: "unet_resnet34_s1s2", modality: "s1s2", arch: "unet", encoder: "resnet34", training_state: "completed", best_val_iou: 0.84,
  best_epoch: 31, evaluated: true, test_iou: 0.8233, bolivia_iou: 0.7836, n_params: 24e6, checkpoint_mb: 94, loaded: false };

const routes: Record<string, unknown> = {
  "/api/model/status": { ready: true, default_model: model.name, device_setting: "auto", cuda_available: true, loaded: [], models: [model] },
  "/api/metrics": { summary: { protocol: "p" }, routing: null, baselines: { s2_ndwi_gt0: { description: "", splits: { test: metrics(0.7578), bolivia: metrics(0.832) } } },
    runs: { [model.name]: { run: model.name, modality: "s1s2", val_selected_threshold: 0.4, splits: { test: metrics(0.8233), bolivia: metrics(0.7836) },
      splits_at_val_threshold: { test: metrics(0.8253), bolivia: metrics(0.7939) }, latency: { model_only_ms_mean: 17.1, precision: "fp16 autocast" } } } },
  "/api/evaluation/external": { worldfloods: null, audit: null, weakness_analysis: null },
  "/api/datasets/status": { splits: {}, verification: null, external: null, storage: {} },
  "/api/system": { gpu: null, ram: { available_gib: 2, total_gib: 16 }, cpu: {}, disk: {}, torch: {} },
  "/api/warnings": { warnings: [{ severity: "warning", source: "system", message: "Low free RAM: 0.20 GiB available.", observed: "t" }] },
  "/api/routes/areas": { areas: [], attribution: "OSM", disclaimer: "" },
  "/api/samples": { samples: [{ id: "Spain_7370579", event: "Spain", split: "test", water_fraction: 0.27, valid_fraction: 1, bounds: [0, 0, 1, 1] }] },
  "/api/segmentation/formats": { max_upload_mb: 40, max_pixels: 4194304, min_side_px: 32, formats: [], notes: [] },
};

function mockFetch(overrides: Record<string, { status: number; body: unknown }> = {}) {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const path = url.split("?")[0];
    if (overrides[path]) return new Response(JSON.stringify(overrides[path].body), { status: overrides[path].status });
    if (path.startsWith("/api/history")) return new Response(JSON.stringify({ total: 0, items: [] }));
    const body = routes[path];
    return body !== undefined ? new Response(JSON.stringify(body)) : new Response(JSON.stringify({ message: "not mocked" }), { status: 404 });
  }));
}

beforeEach(() => mockFetch());
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("Command page", () => {
  it("shows readiness and held-out metrics from the API (validation-selected threshold)", async () => {
    render(<MemoryRouter><Command /></MemoryRouter>);
    await waitFor(() => expect(screen.getByText("Ready")).toBeTruthy());
    expect((await screen.findAllByText("0.825")).length).toBeGreaterThan(0); // test IoU @ val threshold
    expect(screen.getAllByText("0.794").length).toBeGreaterThan(0); // Bolivia IoU @ val threshold
    expect(screen.getByText(/Low free RAM/)).toBeTruthy(); // real warning feed
    expect(screen.getByText("No segmentation has been run yet.")).toBeTruthy(); // honest empty state
  });
});

describe("Segmentation studio", () => {
  it("surfaces the server's error message when inference is rejected", async () => {
    mockFetch({ "/api/segmentation/predict-sample": { status: 422, body: { error: "invalid_input", message: "model x needs 's1' input" } } });
    render(<MemoryRouter><Studio /></MemoryRouter>);
    const run = await screen.findByRole("button", { name: /Run segmentation/ });
    await waitFor(() => expect((run as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(run);
    expect((await screen.findByRole("alert")).textContent).toContain("needs 's1' input");
  });
});

describe("History page", () => {
  it("renders an honest empty state and export links", async () => {
    render(<MemoryRouter><History /></MemoryRouter>);
    expect(await screen.findByText("No analyses yet")).toBeTruthy();
    expect(screen.getByRole("link", { name: /CSV/ }).getAttribute("href")).toContain("/api/history/export?format=csv");
  });
});
