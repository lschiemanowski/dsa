import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";

const repo = fileURLToPath(new URL("..", import.meta.url));
const ui = fileURLToPath(new URL(".", import.meta.url));
const backend = "http://127.0.0.1:8011";
const frontend = "http://127.0.0.1:5176";

export default defineConfig({
  testDir: "./test",
  use: {
    baseURL: frontend,
    launchOptions: {
      executablePath: process.env.DSA_TEST_CHROMIUM || undefined,
    },
  },
  webServer: [
    {
      cwd: repo,
      command:
        "uv run --no-sync uvicorn tests.private_data_chat.browser_fixture:make_app --factory --host 127.0.0.1 --port 8011",
      url: backend + "/openapi.json",
    },
    {
      cwd: ui,
      command: process.env.DSA_TEST_BUILD
        ? "npm run build && npm exec vite preview -- --port 5176 --strictPort"
        : "npm run dev -- --port 5176 --strictPort",
      env: { DSA_CHAT_API_ORIGIN: backend },
      url: frontend,
    },
  ],
});
