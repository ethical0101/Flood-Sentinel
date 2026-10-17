import { describe, expect, it } from "vitest";
import { fmtBytes, fmtDist, fmtDuration, fmtNum, fmtPct, modelLabel } from "../lib";

describe("formatters", () => {
  it("formats percentages and missing values", () => {
    expect(fmtPct(0.8445)).toBe("84.5%");
    expect(fmtPct(null)).toBe("–");
    expect(fmtPct(Number.NaN)).toBe("–");
    expect(fmtNum(0.123456)).toBe("0.123");
  });
  it("formats distances, durations and bytes", () => {
    expect(fmtDist(387)).toBe("387 m");
    expect(fmtDist(4964.1)).toBe("4.96 km");
    expect(fmtDuration(45)).toBe("45 s");
    expect(fmtDuration(3725)).toBe("1 h 2 min");
    expect(fmtBytes(1_752_900_000)).toBe("1.8 GB");
  });
  it("labels models", () => {
    expect(modelLabel({ arch: "unet", encoder: "resnet34", modality: "s1s2" })).toBe("U-Net · ResNet-34 · S1S2");
    expect(modelLabel({ arch: "deeplabv3plus", encoder: "tu-mobilenetv3_large_100", modality: "s1s2" })).toContain("MobileNetV3-L");
  });
});
