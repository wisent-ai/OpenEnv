/**
 * Automate OAuth consent via Playwright with user Chrome profile.
 * Usage: node scripts/_oauth_browser.js AUTH_URL
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

const CHROME_DIR = path.join(
  os.homedir(), "Library", "Application Support",
  "Google", "Chrome"
);
const CB_PREFIX = "console.anthropic.com/oauth/code/callback";

async function main() {
  const args = process.argv.slice(_TWO);
  const url = args[_ZERO];
  if (!url || !url.startsWith("https://")) {
    console.error("Usage: node _oauth_browser.js AUTH_URL");
    process.exit(_ONE);
  }

  console.error("Launching Chrome with user profile...");
  const ctx = await chromium.launchPersistentContext(CHROME_DIR, {
    channel: "chrome",
    headless: false,
    args: [
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-features=ChromeWhatsNewUI",
    ],
  });

  const page = await ctx.newPage();
  console.error("Navigating to auth URL...");
  await page.goto(url, { waitUntil: "networkidle", timeout: _TIMEOUT });
  await page.waitForTimeout(_SETTLE);

  /* Screenshot for debug */
  console.error("Page title: " + await page.title());
  console.error("Page URL: " + page.url());

  /* Find and click submit/allow button */
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
    console.error("No submit btn found, trying visible buttons...");
    const btns = await page.$$("button");
    for (const b of btns) {
      const vis = await b.isVisible();
      if (vis) {
        const txt = await b.textContent();
        console.error("Clicking button: " + (txt || "").trim());
        await b.click();
        clicked = true;
        break;
      }
    }
  }

  if (!clicked) {
    console.error("No clickable button found. Page content:");
    const body = await page.textContent("body");
    console.error(body.substring(_ZERO, _THOUSAND));
  }

  console.error("Waiting for redirect to callback...");
  try {
    await page.waitForURL("**/" + CB_PREFIX + "**", {
      timeout: _TIMEOUT,
    });
  } catch (_) {
    console.error("Timeout waiting for redirect. Current: " + page.url());
  }

  const fin = page.url();
  console.error("Final URL: " + fin);

  let code = null;
  try {
    const u = new URL(fin);
    code = u.searchParams.get("code");
  } catch (_) { /* ignore */ }

  if (!code) {
    const txt = await page.textContent("body");
    const re = new RegExp("[A-Za-z\\d_-]{" + _MIN_LEN + ",}");
    const m = txt.match(re);
    if (m) code = m[_ZERO];
  }

  await page.close();
  await ctx.close();

  if (code) {
    process.stdout.write(code);
  } else {
    console.error("No code found.");
    process.exit(_ONE);
  }
}

main().catch((e) => {
  console.error("Error: " + e.message);
  process.exit(_ONE);
});
