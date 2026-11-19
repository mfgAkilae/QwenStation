// Run against an isolated station started with --mock --no-autostart --keep.
const { before, after, test } = require("node:test");
const assert = require("node:assert/strict");
const { chromium } = require("playwright-core");

const url = process.env.QWEN_STATION_TEST_URL;
let browser;
before(async () => {
  const state = await (await fetch(url + "/api/state")).json();
  assert.equal(state.station.mock, true, "Dashboard tests require an isolated mock station");
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
});
after(async () => { await browser?.close(); });

async function dashboard(t) {
  const page = await browser.newPage();
  t.after(() => page.context().close());
  page.setDefaultTimeout(4500);
  await page.goto(url);
  await page.locator("#preset option").first().waitFor({state:"attached"});
  return page;
}

test("a preset survives leaving the selector and a status refresh", async t => {
  const page = await dashboard(t);
  const select = page.locator("#preset");
  const original = await select.inputValue();
  const next = await select.locator("option").evaluateAll((options, current) =>
    options.find(o => o.value !== current).value, original);
  await select.selectOption(next);
  await page.locator("#primary").focus();
  await page.evaluate(() => poll());
  assert.equal(await select.inputValue(), next);
  let submitted;
  await page.route("**/api/start", route => {
    submitted = route.request().postDataJSON().preset;
    return route.fulfill({json:{ok:true}});
  });
  await page.locator("#primary").click();
  assert.equal(submitted, next);
});

test("a rejected action explains the failure after the next successful refresh", async t => {
  const page = await dashboard(t);
  await page.route("**/api/start", route => route.fulfill({status:409, json:{error:"Finish the setup first"}}));
  await page.locator("#primary").click();
  await page.evaluate(() => poll());
  await page.locator("#banner").filter({hasText:"Finish the setup first"}).waitFor();
});

test("rejected quit keeps the dashboard usable", async t => {
  const page = await dashboard(t);
  await page.route("**/api/quit", route => route.fulfill({status:403,body:"Forbidden"}));
  await page.locator("#quit").click();
  await page.locator("#banner").filter({hasText:/quit|403|reload/i}).waitFor();
  await page.locator("#openSettings").click();
  await page.locator("dialog[open]").waitFor();
});

test("a failed fresh Linux setup can change its location", async t => {
  const page = await dashboard(t);
  let setupState = "failed";
  await page.route("**/api/state?**", async route => {
    const data = await (await route.fetch()).json();
    data.setup = {...data.setup, state:setupState, platform:"linux", reason:"install", location:"/old/engine",
                  error:"No space left on device", started_at:data.now - 60, finished_at:data.now - 40, steps:[], checks:[]};
    await route.fulfill({json:data});
  });
  await page.route("**/api/setup/check", route => {
    setupState = "needed";
    return route.fulfill({json:{ok:true}});
  });
  await page.evaluate(() => poll());
  assert.equal(await page.locator("#elapsed").textContent(), "00:20");
  await page.getByRole("button",{name:"Change folder"}).click();
  const location = page.locator("#setupLocation");
  await location.fill("/new/engine");
  let submitted;
  await page.route("**/api/setup/start", route => {
    submitted = route.request().postDataJSON().location;
    return route.fulfill({json:{ok:true}});
  });
  await page.getByRole("button",{name:"Set up",exact:true}).click();
  assert.equal(submitted, "/new/engine");
});

test("overlapping refreshes share a single state request", async t => {
  const page = await dashboard(t);
  let pending = 0, peak = 0;
  await page.route("**/api/state?**", async route => {
    peak = Math.max(peak, ++pending);
    const response = await route.fetch();
    await new Promise(resolve => setTimeout(resolve, 120));
    await route.fulfill({response});
    pending--;
  });
  await page.evaluate(() => Promise.all([poll(), poll()]));
  assert.equal(peak, 1);
});

test("when clipboard access is denied the new key is selected for manual copy", async t => {
  const page = await dashboard(t);
  await page.evaluate(() => Object.defineProperty(navigator, "clipboard", {
    value: {writeText: async () => {throw new Error("Permission denied");}}
  }));
  await page.locator("#openSettings").click();
  await page.locator("#keyNew").click();
  await page.locator("#keyCopy").click();
  assert.ok(await page.evaluate(() => {
    const value = document.getElementById("keyValue").textContent;
    return value.length > 0 && window.getSelection().toString() === value;
  }), "select the displayed value when automatic copying fails");
  await page.locator("#settingsMessage").filter({hasText:/Ctrl\+C|manually/}).waitFor();
});

test("save settings and start and stop the mock model from the dashboard", async t => {
  const page = await dashboard(t);
  page.on("dialog", dialog => dialog.accept());
  await page.locator("#openSettings").click();
  await page.locator("#tab-model").click();
  const tokens = page.locator("#draftTokens");
  const original = await tokens.inputValue();
  const next = original === "8" ? "9" : "8";
  await tokens.fill("999");
  await page.locator("#settingsSave").click();
  await page.locator('[data-key="draft_tokens"][data-invalid] .err').waitFor();
  await tokens.fill(next);
  await page.locator("#settingsSave").click();
  await page.locator("dialog").waitFor({state:"hidden"});
  await page.locator("#openSettings").click();
  await page.locator("#tab-model").click();
  assert.equal(await tokens.inputValue(), next);
  await tokens.fill(original);
  await page.locator("#settingsSave").click();
  await page.locator("dialog").waitFor({state:"hidden"});
  await page.locator("#primary").click();
  await page.locator("#stateLabel").filter({hasText:/^Ready$/}).waitFor({timeout:60000});
  await page.locator("#primary").click();
  await page.locator("#stateLabel").filter({hasText:/^Stopped$/}).waitFor({timeout:30000});
});
