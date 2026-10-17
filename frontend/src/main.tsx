import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Segmentation from "./pages/Segmentation";
import RoutePlanner from "./pages/RoutePlanner";
import Evaluation from "./pages/Evaluation";
import Training from "./pages/Training";
import About from "./pages/About";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="segmentation" element={<Segmentation />} />
          <Route path="routes" element={<RoutePlanner />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="training" element={<Training />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<Overview />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
