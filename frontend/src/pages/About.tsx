import type { ReactNode } from "react";
import { ArrowDown, ArrowRight } from "lucide-react";
import { Callout, Card, PageHeader } from "../components/ui";

function Box({ title, children, tone = "slate" }: { title: string; children: ReactNode; tone?: "slate" | "blue" | "green" | "amber" }) {
  const cls = { slate: "border-slate-300 bg-white", blue: "border-water-500/40 bg-water-50", green: "border-emerald-300 bg-emerald-50", amber: "border-amber-300 bg-amber-50" }[tone];
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${cls}`}>
      <div className="text-[12.5px] font-semibold text-slate-900">{title}</div>
      <div className="mt-0.5 text-[11.5px] leading-snug text-slate-600">{children}</div>
    </div>
  );
}

function Arrow({ down }: { down?: boolean }) {
  return <div className="flex items-center justify-center text-slate-400">{down ? <ArrowDown className="h-4 w-4" /> : <ArrowRight className="h-4 w-4" />}</div>;
}

export default function About() {
  return (
    <div className="max-w-6xl">
      <PageHeader eyebrow="Documentation" title="How the system works">
        Methodology, data sources and limits of this research prototype. Full details are in <code>README.md</code> and the
        <code> docs/</code> folder of the repository.
      </PageHeader>

      <Card title="Architecture">
        <div className="grid items-stretch gap-2 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          <Box title="Sentinel-1 / Sentinel-2 imagery">Sen1Floods11 GeoTIFF chips (EPSG:4326, 10 m), or a user GeoTIFF / RGB image</Box>
          <Arrow />
          <Box title="Preprocessing" tone="blue">Band standardisation with train-split statistics; 512² tiling with blended overlap</Box>
          <Arrow />
          <Box title="Segmentation CNN" tone="blue">U-Net (ResNet-34) or DeepLabV3+ (MobileNetV3), trained locally with AMP on an RTX 3050</Box>
          <Arrow />
          <Box title="P(water) raster" tone="blue">Georeferenced probability map + mask, overlay and area estimate</Box>
        </div>
        <div className="my-2 grid lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]"><div className="lg:col-start-7"><Arrow down /></div></div>
        <div className="grid items-stretch gap-2 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          <Box title="Dashboard (React + Leaflet)" tone="amber">Map, comparison and explanation; labels every layer as a prediction or a reference</Box>
          <div className="flex items-center justify-center text-slate-400"><ArrowRight className="h-4 w-4 rotate-180" /></div>
          <Box title="Dijkstra routing" tone="green">Shortest-distance baseline vs flood-penalised cost; penalty-method alternatives</Box>
          <div className="flex items-center justify-center text-slate-400"><ArrowRight className="h-4 w-4 rotate-180" /></div>
          <Box title="Edge risk scoring" tone="green">Sample P(water) every 5 m along each road; CRS-aware alignment; unknown ≠ dry</Box>
          <div className="flex items-center justify-center text-slate-400"><ArrowRight className="h-4 w-4 rotate-180" /></div>
          <Box title="OpenStreetMap roads" tone="green">OSMnx drive network per study area, fetched once and cached as GraphML</Box>
        </div>
        <p className="mt-3 text-[11.5px] text-slate-500">Backend: FastAPI (models loaded once and reused; GPU inference serialised). Training runs as a separate, resumable process. API reference: <a className="text-water-600 hover:underline" href="/docs" target="_blank" rel="noreferrer">/docs</a> (OpenAPI).</p>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Research methodology">
          <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-slate-700">
            <li><b>Task:</b> binary segmentation, positive class = water (flood water and permanent water, as hand-labelled). Pixels labelled −1 are ignored in the loss and in all metrics.</li>
            <li><b>Splits:</b> the official Sen1Floods11 train / valid / test lists are used unchanged. The Bolivia event is held out entirely as an unseen-region test.</li>
            <li><b>Training:</b> random 256² crops, flips and rotations, BCE + soft Dice, AdamW with warmup and cosine LR, fp16 autocast, early stopping on validation IoU.</li>
            <li><b>Model selection</b> uses the validation split only. The test and Bolivia splits are evaluated once, at a fixed 0.5 threshold.</li>
            <li><b>Baselines:</b> a constant predictor, NDWI &gt; 0, MNDWI &gt; 0, the dataset's Sentinel-1 Otsu threshold labels, and JRC permanent water.</li>
            <li><b>Ablations:</b> input modality (S1, S2, S1+S2, RGB) with the same U-Net, and U-Net/ResNet-34 against DeepLabV3+/MobileNetV3 on S1+S2.</li>
          </ul>
        </Card>
        <Card title="Routing method">
          <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-slate-700">
            <li>Each OSM edge is densified to one sample per 5 m. Points are projected into the raster CRS and the raster is max-filtered by a configurable buffer, which allows for road width and misalignment.</li>
            <li>Edge cost = length + <i>penalty</i> × predicted-flooded length + <i>unknown penalty</i> × length without data + length × mean P(water).</li>
            <li>An optional closure rule removes edges with at least <i>N</i> m of predicted flooding. This is a user-chosen conservative assumption, not a verified closure.</li>
            <li>Disconnected graphs, start/end points far from any road, and the case where no route avoids every closure all return explicit, explained results.</li>
            <li>Travel time is a free-flow estimate from OSM maxspeed or highway-type averages. Slowdowns caused by flooding are not modelled.</li>
          </ul>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="How to run">
          <pre className="overflow-x-auto rounded-lg bg-ink-950 p-3 font-mono text-[11.5px] leading-relaxed text-slate-200">{`# backend (from the repository root)
.venv\\Scripts\\python -m uvicorn backend.app.main:app --port 8000

# frontend (development)
cd frontend && npm run dev        # http://127.0.0.1:5173

# or serve the production build from FastAPI
cd frontend && npm run build      # then open http://127.0.0.1:8000

# reproduce data, training and evaluation
.venv\\Scripts\\python scripts\\download_datasets.py
.venv\\Scripts\\python scripts\\verify_datasets.py
.venv\\Scripts\\python -m training.prepare_data
.venv\\Scripts\\python scripts\\run_experiments.py`}</pre>
        </Card>
        <Card title="Data sources & citations">
          <ul className="space-y-2.5 text-[12.5px] leading-relaxed text-slate-700">
            <li><b>Sen1Floods11</b>: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). <i>Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.</i> CVPR Workshops, pp. 210–211. Data: gs://sen1floods11 (Cloud to Street). No explicit licence file is published; the data is used here for non-commercial research and is not redistributed.</li>
            <li><b>Copernicus Sentinel-1/-2</b> imagery (contained in Sen1Floods11), © ESA / Copernicus.</li>
            <li><b>OpenStreetMap</b> road network and basemap tiles, © OpenStreetMap contributors, ODbL 1.0. Accessed through OSMnx (Boeing, 2017).</li>
            <li><b>NDWI</b>: McFeeters (1996), Int. J. Remote Sensing 17(7). <b>MNDWI</b>: Xu (2006), Int. J. Remote Sensing 27(14).</li>
            <li><b>Pretrained encoders</b>: ImageNet weights distributed with segmentation-models-pytorch / timm. Every flood-specific weight was trained locally.</li>
          </ul>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Known limitations">
          <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-slate-700">
            <li>The labels mark all surface water, so permanent rivers and lakes count as "flooded" for routing unless a bridge carries the road.</li>
            <li>A 10 m pixel is comparable to the width of a road. Narrow flooded streets in towns can be missed, and bridges over water can be flagged.</li>
            <li>The imagery comes from 2016–2019 flood events, but the OSM roads are current. The two are not contemporaneous.</li>
            <li>Optical bands fail under cloud. SAR misses water under vegetation and confuses smooth surfaces such as tarmac and sand with water.</li>
            <li>RGB photos from drones or phones are outside the training distribution and carry no geolocation.</li>
            <li>Routing considers only roads inside the 5 × 5 km raster footprint.</li>
          </ul>
        </Card>
        <div className="min-w-0 space-y-4">
          <Callout tone="danger" title="Safety disclaimer">
            <span>
              This is a student research prototype. Flood extents are statistical model estimates and the routes have
              not been verified. Never use it to make real evacuation or navigation decisions. Follow official emergency
              services, local authorities and verified road-closure information.</span>
          </Callout>
          <Callout title="Integrity statement">
            Every metric, curve and prediction in this dashboard is read from files produced by the training,
            evaluation and inference code in this repository. Hand labels are always marked as ground truth and are
            never presented as predictions.
          </Callout>
        </div>
      </div>
    </div>
  );
}
