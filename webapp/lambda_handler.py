"""Stateless AWS Lambda adapter for the hosted web app.

The local server (`webapp/server.py`) keeps a process-wide `DraftState` store:
a visitor publishes a board, the agent clones it into sandboxes, and a later
request can still read those sandboxes. A Lambda container has no such shared
lifetime, so this adapter rebuilds the store from the posted board on every
invocation and drops it afterwards. That is a deliberate trade: the tool loop,
state tools and transcript are all real, but sandboxes do not survive a
request. The browser already re-sends the board on each call, so nothing the
demo shows is lost.

No DynamoDB, no VPC, no second state format. The only new code is this shim
over the same handlers the local server uses, plus a deployment-only tool list
that removes the two data tools the slim image does not ship.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import server  # noqa: E402

# suggest.py already ablates both for its measured reranking arms. The free-
# form chat needs the same list on the hosted deployment: no SQLite corpus and
# no DuckDB store are in the image, so neither tool can be advertised.
WEB_ABLATE = frozenset({"sql", "soloq_matchup"})


def _web_system():
    """The agent prompt with only the two data-tool paragraphs removed.

    `loop.SYSTEM` is pinned by tests and stays intact for local runs. Cutting
    the numbered instructions for `sql`/`soloq_matchup` matters: an ablated
    tool still refuses a call, but a prompt that advertises it invites a
    wasted step on every turn.
    """
    from pro.model.agent import loop

    text = loop.SYSTEM
    start = text.index("5. `sql` only")
    end = text.index("\nComparing named champions", start)
    text = text[:start] + text[end + 1:]
    start = text.index("- `soloq_matchup` returns")
    end = text.index("- The model cannot see", start)
    return text[:start] + text[end:]


def _bootstrap():
    """Load the shipped heads once per container, on the first invocation.

    `load_predictor` swaps both the slot and the engine under any state the
    store already holds, so doing this before the first board is published is
    all the wiring needed. The CPU wheel matches the dev torch version, so the
    checkpoints load identically.
    """
    app = server.APP
    if app.get("_web_booted"):
        return
    mapping, names_by_idx, champions = server.load_champ_data()
    app.update(mapping=mapping, names_by_idx=names_by_idx,
               champions=champions, device="cpu",
               agent_calls=int(os.getenv("AGENT_CALLS", "12")),
               agent_ablate=WEB_ABLATE, agent_system=_web_system())

    masked = server.MASKED_DEFAULT
    if not masked.is_file():
        masked, _score = server.find_best_ckpt()
    server.load_predictor(masked, device="cpu", role="masked")
    if server.POLICY_DEFAULT.is_file():
        server.load_predictor(server.POLICY_DEFAULT, device="cpu",
                              role="policy")
    app["_web_booted"] = True


class _Shim(server.Handler):
    """The same handler methods, fed by a Lambda event instead of a socket."""

    def __init__(self, payload, method="POST", path="/"):
        self.payload = payload
        self.command = method
        self.path = path
        self.status = 200
        self.body = None
        self.ctype = None

    def _read_json(self, max_bytes=1_000_000):
        if self.payload is None:
            raise ValueError("request body missing")
        return self.payload

    def _send(self, code, body, ctype="application/json"):
        self.status = code
        self.body = body
        self.ctype = ctype


def _dispatch(method, path, payload):
    shim = _Shim(payload, method=method, path=path)
    if method == "GET":
        shim.do_GET()
        return shim.status, shim.body, shim.ctype
    if method != "POST":
        shim._send(405, {"error": "method not allowed"})
        return shim.status, shim.body, shim.ctype

    route = path.split("?")[0]
    routes = {
        "/api/suggest": shim._handle_suggest,
        "/api/next-move": shim._handle_next_move,
        "/api/agent": shim._handle_agent,
        "/api/agent-suggest": shim._handle_agent_suggest,
        "/api/agent-rank": shim._handle_agent_rank,
        "/api/state": shim._handle_state,
    }
    fn = routes.get(route)
    if fn is None:
        shim._send(404, {"error": "unknown endpoint"})
    else:
        fn()
    return shim.status, shim.body, shim.ctype


def _request_path(event):
    """Rebuild the full request target from a Function URL event.

    Function URLs deliver `rawPath` and `rawQueryString` as separate fields.
    `server.Handler.do_GET` parses `self.path` with urlparse, so handing it a
    bare path silently drops every `?i=` / `?q=` / `?limit=` parameter: the
    draft picker asks for draft -1 and the filter shows all games.
    """
    path = event.get("rawPath") or event.get("path") or "/"
    if "?" in path:
        return path
    query = event.get("rawQueryString")
    if not query:
        params = event.get("queryStringParameters") or {}
        if params:
            from urllib.parse import urlencode
            query = urlencode(params)
    return "%s?%s" % (path, query) if query else path


def lambda_handler(event, context=None):
    """AWS Function URL entry point. One request, one fresh draft state."""
    _bootstrap()
    # A warm container must not leak one visitor's board or sandboxes to the
    # next. `_publish_state` writes `real` from this request's payload.
    server._STATE_STORE = None
    server.APP["state_error"] = None

    ctx = event.get("requestContext") or {}
    http = ctx.get("http") or {}
    method = str(http.get("method") or event.get("httpMethod") or "GET").upper()
    path = _request_path(event)
    headers = {str(k).lower(): v for k, v in (event.get("headers") or {}).items()}

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": _cors_headers(headers, event),
                "body": ""}

    payload = None
    if method == "POST":
        raw = event.get("body") or ""
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8", "replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return _response(400, {"error": "invalid JSON: %s" % exc},
                             headers, event)

    try:
        status, body, ctype = _dispatch(method, path, payload)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        status, body, ctype = 400, {"error": str(exc)}, None
    except Exception as exc:  # noqa: BLE001 - one panel, not the function
        status, body, ctype = 500, {"error": "%s: %s" % (type(exc).__name__, exc)}, None
    return _response(status, body, headers, event, ctype=ctype)


def _cors_headers(request_headers, event):
    """Mirror the configured GitHub Pages origin, never a wildcard with auth."""
    origin = os.getenv("WEB_ORIGIN", "").strip()
    sent = request_headers.get("origin", "")
    out = {"Cache-Control": "no-store"}
    if origin and (not sent or sent == origin):
        out["Access-Control-Allow-Origin"] = origin
        out["Vary"] = "Origin"
    return out


def _response(status, body, headers, event, ctype=None):
    out = _cors_headers(headers, event)
    if isinstance(body, bytes):
        # server.do_GET passes text/html, text/javascript or text/css here.
        # Without it browsers can refuse to execute the iframe's scripts.
        out["Content-Type"] = ctype or "application/octet-stream"
        data = body
    else:
        out["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    out["Content-Length"] = str(len(data))
    return {"statusCode": status, "headers": out,
            "body": data.decode("utf-8"), "isBase64Encoded": False}


if __name__ == "__main__":
    # Local smoke test without an HTTP server:
    #   python webapp/lambda_handler.py '{"httpMethod":"GET","path":"/api/champions"}'
    print(json.dumps(lambda_handler(json.loads(sys.argv[1])), indent=2))
