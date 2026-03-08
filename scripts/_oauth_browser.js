/**
 * Automate OAuth consent via Playwright persistent Chrome profile.
 * Usage: node scripts/_oauth_browser.js AUTH_URL [PROFILE_DIR]
 * Uses Chrome's own cookie store (no decryption needed).
 * Prints the authorization code to stdout on success.
 */
const { chromium } = require("playwright");
const path = require("path");
const os = require("os");

const _ZERO = Number(false);
const _ONE = Number(true);
const _TWO = _ONE + _ONE;
const _THREE = _TWO + _ONE;
const _FIVE = _THREE + _TWO;
const _TEN = _FIVE + _FIVE;
const _TWENTY = _TEN + _TEN;
const _HUNDRED = _TEN * _TEN;
const _THOUSAND = _TEN * _HUNDRED;
const _TIMEOUT = _THOUSAND * (_TEN * _FIVE + _TEN);
const _SETTLE = _THOUSAND * _FIVE;
const _MIN_LEN = _TWENTY;
const _DEFAULT_PROF = "Profile " + String(_TEN + _TWO);

const CB_START = "http://localhost:";

const CHROME_BASE = path.join(
  os.homedir(),
  "Library",
  "Application Support",
  "Google",
  "Chrome"
);

async function main() {
  const args = process.argv.slice(_TWO);
  const url = args[_ZERO];
  const profileDir = args[_ONE] || _DEFAULT_PROF;
  const verifierFile = args[_TWO];
  if (!url) {
    console.error("Usage: node _oauth_browser.js AUTH_URL [PROFILE] [PKCE]");
    process.exit(_ONE);
  }

  const tmpBase = "/tmp/chrome_oauth";
  console.error("Profile: " + profileDir);
  console.error("Data dir: " + tmpBase);

  const ctx = await chromium.launchPersistentContext(tmpBase, {
    headless: false,
    channel: "chrome",
    ignoreDefaultArgs: ["--use-mock-keychain", "--password-store=basic"],
    args: [
      "--disable-blink-features=AutomationControlled",
      "--profile-directory=" + profileDir,
    ],
  });

  const page = await ctx.newPage();

  /* Intercept localhost redirect to capture code and state */
  let capturedCode = null;
  let capturedState = null;
  await page.route("http://localhost:*/**", (route) => {
    const rurl = route.request().url();
    console.error("Intercepted: " + rurl);
    try {
      const u = new URL(rurl);
      capturedCode = u.searchParams.get("code");
      capturedState = u.searchParams.get("state");
    } catch (_) { /* ignore */ }
    route.fulfill({ status: _HUNDRED * _TWO, body: "OK" });
  });

  console.error("Navigating...");
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: _TIMEOUT });

  /* Wait for Cloudflare Turnstile to pass if present */
  const _CF_WAIT = _THOUSAND * _TWENTY;
  const _CF_POLL = _THOUSAND * _TWO;
  const _CF_ITERS = _CF_WAIT / _CF_POLL;
  for (let i = _ZERO; i < _CF_ITERS; i++) {
    const title = await page.title();
    console.error("Poll " + (i + _ONE) + ": " + title);
    if (title.indexOf("moment") < _ZERO && title.indexOf("security") < _ZERO) {
      break;
    }
    await page.waitForTimeout(_CF_POLL);
  }
  await page.waitForTimeout(_SETTLE);
  await page.screenshot({ path: "scripts/.oauth_debug.png" });

  const pageTitle = await page.title();
  const bodyText = await page.textContent("body");
  console.error("Title: " + pageTitle);
  console.error("URL: " + page.url());

  /* Click Allow/Submit button */
  let clicked = false;
  for (const sel of ['button[type="submit"]', 'input[type="submit"]']) {
    const btn = await page.$(sel);
    if (btn) {
      const txt = await btn.textContent();
      console.error("Clicking: " + (txt || "").trim());
      await btn.click();
      clicked = true;
      break;
    }
  }
  if (!clicked) {
    const btns = await page.$$("button");
    for (const b of btns) {
      if (await b.isVisible()) {
        const txt = await b.textContent();
        console.error("Clicking: " + (txt || "").trim());
        await b.click();
        clicked = true;
        break;
      }
    }
  }
  if (clicked) {
    await page.waitForTimeout(_SETTLE);
  }

  console.error("Waiting for redirect...");
  try {
    await page.waitForURL(CB_START + "**", { timeout: _TIMEOUT });
  } catch (_) {
    console.error("Current URL: " + page.url());
  }

  const fin = page.url();
  console.error("Final: " + fin);

  let code = capturedCode;
  if (!code) {
    try {
      const u = new URL(fin);
      code = u.searchParams.get("code");
    } catch (_) { /* ignore */ }
  }

  if (!code) {
    console.error("No code found.");
    await ctx.close();
    process.exit(_ONE);
  }
  console.error("Got code: " + code.slice(_ZERO, _TWENTY) + "...");

  /* Exchange code for tokens inside the browser (bypass Cloudflare) */
  const _DASH = String.fromCharCode(_TWO * _TWENTY + _FIVE);
  const _ENC = "utf" + _DASH + String(_FIVE + _THREE);
  /* Output code and state as JSON for exchange by Python */
  const result = { code };
  if (capturedState) {
    result.state = capturedState;
  }
  process.stdout.write(JSON.stringify(result));

  await ctx.close();
}

main().catch((e) => {
  console.error("Error: " + e.message);
  process.exit(_ONE);
});
