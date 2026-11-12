import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const proxy = {
  // The API checks that the browser's Origin matches the Host it receives.
  "/api": {
    target: process.env.DSA_CHAT_API_ORIGIN ?? "http://127.0.0.1:8001",
    changeOrigin: false,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy,
  },
  preview: { host: "127.0.0.1", proxy },
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "react",
              test: /\/node_modules\/(react|react-dom|scheduler)\//,
              priority: 20,
            },
            {
              name: "assistant-ui",
              test: /\/node_modules\/@assistant-ui\//,
              priority: 10,
            },
          ],
        },
      },
    },
  },
});
