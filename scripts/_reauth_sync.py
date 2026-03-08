"""Re-authorize OAuth tokens via browser PKCE flow.

Usage:
    python scripts/_reauth_sync.py anthropic        (open browser)
    python scripts/_reauth_sync.py anthropic CODE   (exchange code)
    python scripts/_reauth_sync.py openai            (auto localhost)
"""
import base64, hashlib, http.server, json, os, secrets, sys
import urllib.parse, webbrowser

sys.path.insert(int(), os.path.join(os.path.dirname(__file__), ".."))

import httpx
from train.self_play.oauth import (
    _read_env_file, _supabase_headers, save_refresh_token,
    fetch_refresh_token,
)
from constant_definitions.var.meta.self_play_constants import (
    ANTHROPIC_OAUTH_TOKEN_URL, ANTHROPIC_OAUTH_CLIENT_ID,
    ANTHROPIC_OAUTH_BETA_HEADER,
    OPENAI_OAUTH_TOKEN_URL, OPENAI_OAUTH_CLIENT_ID,
    SUPABASE_PROVIDER_ANTHROPIC, SUPABASE_PROVIDER_OPENAI,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _TWO + _TWO
_FIVE = _FOUR + _ONE
_TEN = _FIVE + _FIVE
_THIRTY_TWO = _TWO ** _FIVE
_HUNDRED = _FIVE ** _TWO * _FOUR
_HTTP_OK = _TWO * _HUNDRED
_FORM = "application/x-www-form-urlencoded"
_PKCE_FILE = os.path.join(os.path.dirname(__file__), ".pkce_state.json")
_ANT_AUTH = "https://claude.ai/oauth/authorize"
_ANT_REDIR = "https://console.anthropic.com/oauth/code/callback"
_ANT_SCOPES = "org:create_api_key user:profile user:inference"
_OAI_AUTH = "https://auth.openai.com/oauth/authorize"
_OAI_SCOPES = "openid profile email offline_access"
_LH = "localhost"
_OAI_PORT = _ONE + _FOUR * (
    _THREE * _HUNDRED + _FIVE * _TEN + _FOUR)


def _pkce():
    raw = secrets.token_bytes(_THIRTY_TWO)
    v = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    c = base64.urlsafe_b64encode(
        hashlib.sha256(v.encode()).digest()
    ).rstrip(b"=").decode()
    return v, c


def _save_pkce(verifier):
    with open(_PKCE_FILE, "w") as f:
        json.dump({"v": verifier}, f)


def _load_pkce():
    with open(_PKCE_FILE) as f:
        return json.load(f)["v"]


def _upsert(provider, rt, at=""):
    env = _read_env_file()
    su, sk = env["NEXT_PUBLIC_SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"]
    try:
        cid, _ = fetch_refresh_token(provider, su, sk)
    except RuntimeError:
        cid = provider
    save_refresh_token(cid, rt, at, su, sk)
    print("  Saved to Supabase.")


def anthropic_open():
    v, c = _pkce()
    state = secrets.token_urlsafe(_THIRTY_TWO)
    url = _ANT_AUTH + "?" + urllib.parse.urlencode({
        "code": "true", "response_type": "code",
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        "redirect_uri": _ANT_REDIR, "scope": _ANT_SCOPES,
        "code_challenge": c, "code_challenge_method": "S256",
        "state": state,
    })
    _save_pkce(v)
    # Only print URL, do NOT open browser (Puppeteer handles it)
    print("AUTH_URL=" + url)


def anthropic_exchange(raw):
    code = raw.split("#")[_ZERO] if "#" in raw else raw
    verifier = _load_pkce()
    payload = {
        "grant_type": "authorization_code", "code": code,
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        "redirect_uri": _ANT_REDIR, "code_verifier": verifier,
    }
    beta = {"anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER}
    eps = [
        ANTHROPIC_OAUTH_TOKEN_URL.replace(
            "platform.claude.com", "console.anthropic.com"),
        ANTHROPIC_OAUTH_TOKEN_URL,
    ]
    resp = None
    for ep in eps:
        for js in (False, True):
            for bh in (True, False):
                h = dict(beta) if bh else {}
                if js:
                    h["Content-Type"] = "application/json"
                    resp = httpx.post(ep, json=payload, headers=h)
                else:
                    h["Content-Type"] = _FORM
                    resp = httpx.post(ep, data=payload, headers=h)
                tag = ("json" if js else "form") + ("+beta" if bh else "")
                print(f"  {ep} ({tag}): {resp.status_code}")
                if resp.status_code == _HTTP_OK:
                    break
            if resp and resp.status_code == _HTTP_OK:
                break
        if resp and resp.status_code == _HTTP_OK:
            break
    if not resp or resp.status_code != _HTTP_OK:
        if "#" in raw:
            print("  Retrying with full code...")
            payload["code"] = raw
            for ep in eps:
                h = dict(beta)
                h["Content-Type"] = _FORM
                resp = httpx.post(ep, data=payload, headers=h)
                print(f"  {ep} (full): {resp.status_code}")
                if resp.status_code == _HTTP_OK:
                    break
    if not resp or resp.status_code != _HTTP_OK:
        print(f"  FAILED: {resp.text if resp else 'none'}")
        return False
    d = resp.json()
    at, rt = d.get("access_token", ""), d.get("refresh_token", "")
    print(f"  Access token: {len(at)} chars")
    if rt:
        _upsert(SUPABASE_PROVIDER_ANTHROPIC, rt, at)
    print("  Done!")
    return True


def reauth_openai():
    v, c = _pkce()
    redir = f"http://{_LH}:{_OAI_PORT}/auth/callback"
    url = _OAI_AUTH + "?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": OPENAI_OAUTH_CLIENT_ID,
        "redirect_uri": redir, "scope": _OAI_SCOPES,
        "code_challenge": c, "code_challenge_method": "S256",
        "state": secrets.token_urlsafe(_THIRTY_TWO),
    })
    result = {"code": None}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            p = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in p:
                result["code"] = p["code"][_ZERO]
            self.send_response(_HTTP_OK)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"Done. Close this tab.")
        def log_message(self, *a):
            pass

    print(f"Server on {_OAI_PORT}, opening browser...")
    webbrowser.open(url)
    srv = http.server.HTTPServer((_LH, _OAI_PORT), H)
    srv.timeout = _THIRTY_TWO * _FIVE
    srv.handle_request()
    srv.server_close()
    if not result["code"]:
        print("No code. Aborted.")
        return False
    print("Exchanging...")
    resp = httpx.post(OPENAI_OAUTH_TOKEN_URL, data={
        "grant_type": "authorization_code", "code": result["code"],
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "redirect_uri": redir, "code_verifier": v,
    }, headers={"Content-Type": _FORM})
    if resp.status_code != _HTTP_OK:
        print(f"  FAILED ({resp.status_code}): {resp.text}")
        return False
    d = resp.json()
    at, rt = d.get("access_token", ""), d.get("refresh_token", "")
    print(f"  Access token: {len(at)} chars")
    if rt:
        _upsert(SUPABASE_PROVIDER_OPENAI, rt, at)
    print("  Done!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < _TWO:
        print("Usage: _reauth_sync.py anthropic [CODE] | openai")
        sys.exit(_ONE)
    t = sys.argv[_ONE].lower()
    if t == "anthropic":
        if len(sys.argv) >= _THREE:
            anthropic_exchange(sys.argv[_TWO])
        else:
            anthropic_open()
    elif t == "openai":
        reauth_openai()
