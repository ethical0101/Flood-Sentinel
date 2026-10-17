import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, resultFile } from "../api";

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("surfaces backend error messages", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: "invalid_input", message: "File is empty." }), { status: 422 })));
    await expect(api.health()).rejects.toMatchObject({ status: 422, code: "invalid_input", message: "File is empty." });
  });
  it("reports an unreachable backend clearly", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("network");
  });
  it("formats FastAPI validation details", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: [{ msg: "bad threshold" }] }), { status: 422 })));
    await expect(api.health()).rejects.toThrow("bad threshold");
  });
  it("builds result file URLs", () => {
    expect(resultFile("abc", "mask.png", true)).toBe("/api/segmentation/results/abc/files/mask.png?download=true");
  });
});
