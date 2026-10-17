/**
 * Frontend-to-backend integration (runs only when INTEGRATION_API is set, e.g.
 *   $env:INTEGRATION_API="http://127.0.0.1:8000"; npx vitest run integration
 * with the FastAPI server running). Exercises the same endpoints the dashboard calls.
 */
import { describe, expect, it } from "vitest";

const BASE = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.INTEGRATION_API;
const run = BASE ? describe : describe.skip;

run("dashboard ↔ API integration", () => {
  const get = async (p: string) => {
    const r = await fetch(BASE + p);
    expect(r.ok, `${p} -> ${r.status}`).toBe(true);
    return r.json();
  };
  it("serves status endpoints used by the overview", async () => {
    expect((await get("/api/health")).status).toBe("ok");
    const models = await get("/api/model/status");
    expect(models.ready).toBe(true);
    const ds = await get("/api/datasets/status");
    expect(ds.verification.passed).toBe(true);
  });
  it("segments a held-out sample and serves its overlay", async () => {
    const r = await fetch(BASE + "/api/segmentation/predict-sample", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chip_id: "Spain_7370579", threshold: 0.5, tta: false, input: "s1s2" }),
    });
    expect(r.status).toBe(200);
    const j = await r.json();
    expect(j.kind).toBe("model_prediction");
    const img = await fetch(`${BASE}/api/segmentation/results/${j.id}/files/overlay.png`);
    expect(img.headers.get("content-type")).toBe("image/png");
  }, 60000);
  it("plans a risk-aware route on a study area", async () => {
    const { areas } = await get("/api/routes/areas");
    const a = areas[0];
    const r = await fetch(BASE + "/api/routes/plan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ area_id: a.id, source: "model", start: { lat: a.default_start[0], lon: a.default_start[1] },
        end: { lat: a.default_end[0], lon: a.default_end[1] }, threshold: 0.5, buffer_px: 1, flood_penalty: 10,
        unknown_penalty: 0.5, close_flooded: false, closure_min_flooded_m: 20, alternatives: 2 }),
    });
    expect(r.status).toBe(200);
    const j = await r.json();
    expect(j.risk_aware.flooded_length_m).toBeLessThanOrEqual(j.baseline.flooded_length_m);
  }, 120000);
});
