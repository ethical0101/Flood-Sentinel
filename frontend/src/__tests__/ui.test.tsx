import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { DataTable, InfoTip, Metric, Segmented } from "../components/ui";
import { toCsv } from "../lib";

afterEach(cleanup);

describe("design-system components", () => {
  it("DataTable sorts numerically and shows unavailable values as a dash", () => {
    const rows = [{ id: "a", v: 0.5 }, { id: "b", v: 0.9 }, { id: "c", v: null }];
    render(<DataTable rows={rows} rowKey={(r) => r.id} columns={[{ key: "id", label: "Id", value: (r) => r.id }, { key: "v", label: "Value", numeric: true, value: (r) => r.v }]} />);
    const cells = () => screen.getAllByRole("row").slice(1).map((r) => r.textContent);
    expect(cells()[2]).toContain("–");
    fireEvent.click(screen.getByRole("button", { name: /Value/ }));
    expect(cells()[0]).toContain("a"); // ascending: 0.5 first, null last
    fireEvent.click(screen.getByRole("button", { name: /Value/ }));
    expect(cells()[0]).toContain("b"); // descending
  });

  it("toCsv quotes commas, quotes and newlines", () => {
    expect(toCsv(["a", "b"], [["x,y", 'say "hi"'], [1, null]])).toBe('a,b\n"x,y","say ""hi"""\n1,');
  });

  it("Segmented exposes radio semantics", () => {
    const onChange = vi.fn();
    render(<Segmented label="Mode" value="a" onChange={onChange} options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} />);
    expect(screen.getByRole("radio", { name: "A" }).getAttribute("aria-checked")).toBe("true");
    fireEvent.click(screen.getByRole("radio", { name: "B" }));
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("InfoTip is keyboard accessible", () => {
    render(<InfoTip text="IoU definition" />);
    fireEvent.focus(screen.getByRole("button", { name: "More information" }));
    expect(screen.getByRole("tooltip").textContent).toBe("IoU definition");
  });

  it("Metric renders label, value and hint", () => {
    render(<Metric label="Test IoU" value="0.823" hint="threshold 0.4" />);
    expect(screen.getByText("Test IoU")).toBeTruthy();
    expect(screen.getByText("0.823")).toBeTruthy();
    expect(screen.getByText("threshold 0.4")).toBeTruthy();
  });
});
