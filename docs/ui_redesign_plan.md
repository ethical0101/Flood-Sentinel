# UI redesign plan (v2)

## Why redesign
The v1 dashboard worked, but it read as a set of cards rather than an analysis tool:
* every surface had the same visual weight, with large rounded cards and generous empty space;
* maps and images competed with the control panels for room;
* there was no persistent history, no export of structured results, and no way to inspect the data behind a chart;
* there was no global status (API, GPU, active model) and no diagnostics page;
* the routing view did not break risk down per road segment, and its fallback cases were not prominent.

The stack stays the same (React 19, TypeScript, Vite 7, Tailwind 4, Recharts, Leaflet). None of the findings required changing it.

## Design principles
1. **Workspace first.** Maps and imagery get the largest area. Controls sit in compact, collapsible side panels.
2. **Information density with hierarchy.** Small uppercase labels, tabular numerals, and one accent colour per meaning.
3. **Provenance on every number.** Each metric shows its source (model / hand label / baseline / assumption) on hover, and "unavailable" is shown as such.
4. **Honest states.** Every view handles loading skeletons, empty, error and idle states explicitly. Nothing is simulated.
5. **Restraint.** No gradients except the map legend ramps; shadows only on floating layers; 6–8 px radii; motion under 200 ms and disabled under `prefers-reduced-motion`.

## Design tokens (`src/index.css`, `@theme`)
| Token | Value | Use |
|---|---|---|
| `--color-nav` | `#0b1220` | navigation rail and top bar |
| `--color-nav-2` | `#111a2e` | nav hover, panels on dark |
| `--color-canvas` | `#f4f6f9` | app background |
| `--color-panel` | `#ffffff` | panels |
| `--color-line` | `#e3e8ef` | 1 px borders |
| `--color-ink` / `-2` / `-3` | `#0f172a` / `#475569` / `#94a3b8` | text levels |
| `--color-accent` | `#0e7490` (cyan-700) | primary actions, focus, active nav |
| `--color-water` | `#2563eb` | predicted water |
| `--color-risk-low` / `-mid` / `-high` | `#65a30d` / `#d97706` / `#dc2626` | road and route risk |
| `--color-unknown` | `#94a3b8` | missing data |
| `--color-truth` | `#7c3aed` | hand-label (reference) layers, never used for predictions |

Type: system UI stack (Segoe UI Variable on Windows), sizes 11 / 12 / 13 / 14 / 16 / 20 px, with tabular numerals for all metrics and a monospace face for IDs and paths. Spacing: a 4 px grid, 12 px panel padding (16 px on wide screens), 8 px gaps inside panels.

## Component library (`src/components/`)
| Component | Purpose |
|---|---|
| `AppShell` | Collapsible dark navigation rail (icons + labels) and a top bar with the page title and global status chips (API, GPU memory, active model, training state) |
| `Panel` | Header with title, info tooltip and a toolbar slot; dense body; optional footer |
| `Metric` | Label, value, unit, delta and a provenance tooltip |
| `InfoTip` | Keyboard-accessible tooltip (focusable button, `aria-describedby`) for metric definitions |
| `Segmented`, `Tabs`, `Toggle`, `Slider`, `Select`, `Field` | Form controls with labels, hints and validation messages |
| `DataTable` | Sortable columns, numeric alignment, CSV export, empty state |
| `ChartFrame` | Wraps every chart with a "View data" drawer (table + CSV) and a source line |
| `StatusDot`, `Badge`, `Kbd` | Compact status indicators |
| `Skeleton`, `EmptyState`, `ErrorState`, `Toast` | System states |
| `Drawer` | Side sheet used for chart data, run details and route segment lists |

## Page hierarchy
```
/                Command        system readiness · active model · key metrics · recent analyses and routes · warnings
/segmentation    Studio         input (upload | held-out chip) → settings → result workspace (compare | overlay+opacity |
                                mask | confidence | uncertainty | vs NDWI | vs hand label) → stats, exports, history compare
/routes          Route planner  map workspace (imagery, flood layer, risk roads, routes) + left panel (area, layer, assumptions,
                                points with search) + right panel (route comparison, risk breakdown, segment table,
                                explanation, export)
/evaluation      Evaluation lab model table (v1 + v2 + baselines) · test vs Bolivia vs WorldFloods · curves · confusion ·
                                per-event · threshold sweep · latency/VRAM · configs · samples · data drawers
/training        Training & system monitor   live run state from status.json · curves · logs · checkpoints · GPU (NVML) ·
                                guarded start/stop controls · resume commands
/datasets        Dataset explorer   Sen1Floods11 + WorldFloods: provenance, licence, verification, splits, channels, labels,
                                coverage (event table), storage (raw / processed / cache), quality warnings
/history         History & reports  persisted analyses (SQLite) · filter · select two to compare · CSV/JSON export ·
                                downloadable HTML report per analysis or route
/about           Docs & settings    architecture, methodology, formats, offline capability, versions, diagnostics run,
                                safety disclaimer
```

## Key workflows and their states
| Workflow | States |
|---|---|
| Segment upload | idle → file validated client-side (type/size) → uploading → inferring (spinner + elapsed time) → result, or error with the server message (format, channels, size, georeferencing) |
| Plan route | area loading (skeleton map) → scoring roads → ready → planning → result, or an explicit **"No route avoids the impassable segments"** danger banner (never a silent fallback) |
| Start training | disabled while any run is active or VRAM is insufficient → confirmation dialog listing GPU/disk checks → started (PID) → live progress → stop request (graceful, at the end of the current epoch) |

## Accessibility and responsiveness
* All interactive elements are reachable by keyboard and have visible focus rings (accent colour, 2 px). Icon-only buttons carry `aria-label`; tooltips work on focus.
* Colour is never the only signal: risk classes also carry text labels, and route lines use dash patterns.
* Breakpoints: ≥ 1280 px for the three-column route/studio workspace; 768–1279 px for two columns; < 768 px stacks into a single column, with the navigation collapsing to a top bar. No horizontal page scroll at 375 px.

## Data contract rules
* Every number comes from an API response. Static explanatory text is visually distinct (secondary text, no metric styling).
* Unavailable values render as "–" with an "unavailable" tooltip, never as zero.
* Hand-label layers use the violet "reference" colour and badge throughout.
