import { expect, test, type Route } from "@playwright/test";

test("clarifies, revises, approves and opens the verified notebook after refresh", async ({
  page,
  request,
}) => {
  const pageErrors: Error[] = [];
  const traffic: string[] = [];
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    const url = message.location().url;
    const text = message.text();
    const simulatedFailure =
      url.includes("/api/conversations/") &&
      /net::ERR_FAILED|status of (409|503)/.test(text);
    if (message.type() === "error" && !simulatedFailure) {
      consoleErrors.push(url + " " + text);
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error));
  page.on("request", (request) => traffic.push(request.url()));
  await page.goto("/");
  const question = page.getByRole("textbox", { name: "Ask a question" });
  const send = async (text: string) => {
    await question.fill(text);
    await page.getByRole("button", { name: "Send question" }).click();
  };
  await expect(
    page.getByRole("heading", { name: "Instructions" }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator(".brand img")
        .evaluate((image: HTMLImageElement) => image.naturalWidth),
    )
    .toBeGreaterThan(0);
  await send("Show me net sales by month.");
  await expect(
    page.getByText("Which calendar year should I use?"),
  ).toBeVisible();
  await send("Use 2011.");

  const proposal = page
    .getByRole("region", { name: "Analysis proposal" })
    .first();
  await expect(
    proposal.getByText("What were net sales by month in 2011?", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    proposal.getByText("Total net sales (pence)", { exact: true }),
  ).toBeVisible();
  await expect(
    proposal.getByText("Calendar year 2011", { exact: true }),
  ).toBeVisible();
  await expect(
    proposal.getByText('"total_pence"', { exact: true }),
  ).toBeVisible();
  await expect(
    proposal.getByText('"sales_pence"', { exact: true }),
  ).toBeVisible();
  const requestJson = proposal.locator("details pre");
  await expect(requestJson).toBeHidden();
  await proposal.getByText("View request JSON").click();
  await expect(requestJson).toContainText('"allow_plots": true');
  const decisionRoute = "**/proposals/*/decision";
  async function dropAcceptedResponse(route: Route) {
    expect((await route.fetch()).ok()).toBeTruthy();
    await route.abort("failed");
  }
  await page.route(decisionRoute, dropAcceptedResponse, { times: 1 });
  await proposal.getByRole("button", { name: "Keep refining" }).click();
  await expect(question).toBeEnabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await send("Use 2011 and exclude returns.");

  const runAnalysis = page.getByRole("button", { name: "Run analysis" });
  await page.route(
    decisionRoute,
    (route) =>
      route.fulfill({
        status: 409,
        json: { detail: "proposal_digest_mismatch" },
      }),
    { times: 1 },
  );
  await runAnalysis.click();
  await expect(page.getByRole("alert")).toContainText("no longer matches");
  await page.route(decisionRoute, dropAcceptedResponse, { times: 1 });
  await runAnalysis.click();
  await expect(page.getByText("Running the approved analysis…")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);

  const conversationRoute = "**/api/conversations/*";
  await page.route(conversationRoute, (route) =>
    route.fulfill({
      status: 503,
      json: { detail: "service_unavailable" },
    }),
  );
  await page.reload();
  await expect(page.getByRole("alert")).toHaveText(
    "The chat server could not complete the request. Try again.",
  );
  await expect(
    page.getByRole("button", { name: "New conversation" }),
  ).toBeDisabled();
  await page.unroute(conversationRoute);
  await page.getByRole("button", { name: "Retry connection" }).click();
  await expect(page.getByText("Running the approved analysis…")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(
    page.getByRole("region", { name: "Analysis proposal" }),
  ).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Run analysis" })).toHaveCount(
    0,
  );

  expect(
    (await request.post("http://127.0.0.1:8011/__test/release")).ok(),
  ).toBeTruthy();
  const result = page.getByRole("region", { name: "Analysis result" });
  await expect(
    result.getByText("Total net sales (pence)", { exact: true }),
  ).toBeVisible();
  await expect(
    result.getByRole("cell", { name: "9007199254740993" }),
  ).toBeVisible();
  await expect(
    result.getByText('"total_pence"', { exact: true }),
  ).toBeVisible();
  await expect(
    result.getByRole("columnheader", { name: 'Sales (pence) "sales_pence"' }),
  ).toBeVisible();
  await expect(result.locator("details pre")).toBeHidden();
  await result.getByText("View response JSON").click();
  await expect(result.locator("details pre")).toContainText(
    '"total_pence": 9007199254740993',
  );
  expect(traffic.some((url) => url.endsWith("/notebook"))).toBe(false);

  await page.setViewportSize({ width: 1280, height: 860 });
  await result.getByRole("button", { name: "Open notebook" }).click();
  const notebook = page.getByRole("complementary", {
    name: "Derivation notebook",
  });
  await expect(
    notebook.getByRole("heading", { name: "Verified derivation" }),
  ).toBeVisible();
  await expect(
    notebook.getByText("print(total)", { exact: false }),
  ).toBeVisible();
  await expect(
    notebook.getByRole("img", { name: "Monthly sales" }),
  ).toBeVisible();
  await expect(notebook.locator("img")).toHaveCount(1);
  await expect(notebook.getByText("Active output")).toHaveCount(0);
  await expect(
    notebook.getByText("ValueError: example", { exact: true }),
  ).toBeVisible();
  await expect(notebook.getByRole("link", { name: "remote link" })).toHaveCount(
    0,
  );
  expect(traffic.some((url) => url.includes("example.invalid"))).toBe(false);
  const pane = await notebook.boundingBox();
  const chat = await page.locator(".chat").boundingBox();
  expect(pane!.x).toBeGreaterThanOrEqual(chat!.x + chat!.width - 1);
  const downloading = page.waitForEvent("download");
  await notebook.getByRole("link", { name: "Download", exact: true }).click();
  expect((await downloading).suggestedFilename()).toBe("dsa-derivation.ipynb");

  await page.setViewportSize({ width: 390, height: 780 });
  expect((await notebook.boundingBox())!.width).toBe(390);
  await expect(page.locator(".chat")).toBeHidden();
  await notebook.getByRole("button", { name: "Close notebook" }).click();
  await expect(
    result.getByRole("button", { name: "Open notebook" }),
  ).toBeFocused();

  await page.getByRole("button", { name: "New conversation" }).click();
  await send("Report net sales for 2011.");
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(
    page.getByText("No verified notebook was produced."),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Open notebook" })).toHaveCount(
    0,
  );
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
