import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import AppShell from "./components/AppShell";
import { Spinner } from "./components/ui";

// Each page is its own chunk: maps (Leaflet) and charts (Recharts) load only where they are used.
const Command = lazy(() => import("./pages/Command"));
const Studio = lazy(() => import("./pages/Studio"));
const RoutesPage = lazy(() => import("./pages/Routes"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Monitor = lazy(() => import("./pages/Monitor"));
const Datasets = lazy(() => import("./pages/Datasets"));
const History = lazy(() => import("./pages/History"));
const About = lazy(() => import("./pages/About"));

const page = (el: React.ReactNode) => <Suspense fallback={<div className="p-6"><Spinner label="Loading page…" /></div>}>{el}</Suspense>;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={page(<Command />)} />
          <Route path="segmentation" element={page(<Studio />)} />
          <Route path="routes" element={page(<RoutesPage />)} />
          <Route path="evaluation" element={page(<Evaluation />)} />
          <Route path="training" element={page(<Monitor />)} />
          <Route path="datasets" element={page(<Datasets />)} />
          <Route path="history" element={page(<History />)} />
          <Route path="about" element={page(<About />)} />
          <Route path="*" element={page(<Command />)} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
