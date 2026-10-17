import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev server proxies /api to the FastAPI backend (uvicorn on 127.0.0.1:8000).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } },
  },
  build: { chunkSizeWarningLimit: 1200 },
  test: { environment: "jsdom", globals: false },
});
