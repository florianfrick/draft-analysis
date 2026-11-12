"""An MCP server over stdio, so any MCP client can drive the tool layer.

`mcp` 2.3.0 ships a different handler API from the decorator style the docs
show (`Server` has `add_request_handler` / `streamable_http_app`, not
`@server.list_tools()`), so rather than pin a version I cannot exercise, this
speaks the wire protocol directly. The surface a tools-only server needs is
four methods:

    initialize                 -> protocol version, capabilities, serverInfo
    tools/list                 -> the specs `Session.available()` advertises
    tools/call                 -> `session.call`, result as JSON text
    ping                       -> {}

That is not a reimplementation of MCP; it is the four messages, in the order
the spec defines, and a test that pipes them through.

Two things it inherits from `Session` on purpose:

**The ablation applies here too.** `--ablate sql` starts a server that does not
advertise and does not accept `sql`, so an MCP client can be measured on the
same axis as the in-process loop.

**A tool failure is a result, not a protocol error.** A refused query comes back
as `isError: true` with the text of the refusal, never as a JSON-RPC error,
because the agent has to be able to read it and try again. Protocol errors are
reserved for malformed requests - the model cannot see those.

The one thing that is not obvious: **stdout is the wire.** Anything else that
writes there - torch's import banner, a stray print, a warning - corrupts the
stream and the client drops the connection with no useful message. So `main`
takes the real stdout, reassigns `sys.stdout` to stderr, and only then loads
the checkpoint.

    python pro/model/agent/mcp_server.py                  # serve on stdio
    python pro/model/agent/mcp_server.py --ablate sql     # two tools, no escape hatch
    python pro/model/agent/mcp_server.py --self-test      # the protocol, no client
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state, model_tools, schema  # noqa: E402
from pro.model.agent.session import Session           # noqa: E402

PROTOCOL_VERSION = "2025-06-18"
# Echoed back when the client asks for a version we know; a client that asked
# for 2024-11-05 and is told 2025-06-18 may disconnect on principle.
KNOWN_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "lol-draft-tools", "version": "0.1.0"}


def tool_list(session):
    return [{"name": s["name"], "description": s["description"],
             "inputSchema": s["parameters"]}
            for s in session.available()]


def _ok(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": code, "message": message}}


def handle(session, request):
    """One JSON-RPC message in, one response dict out - or None for a notification."""
    if not isinstance(request, dict):
        return _err(None, -32600, "invalid request: expected a JSON object")
    method = request.get("method")
    rid = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        asked = params.get("protocolVersion") if isinstance(params, dict) else None
        return _ok(rid, {"protocolVersion": asked if asked in KNOWN_VERSIONS
                         else PROTOCOL_VERSION,
                         "capabilities": {"tools": {"listChanged": False}},
                         "serverInfo": SERVER_INFO})
    if method == "tools/list":
        return _ok(rid, {"tools": tool_list(session)})
    if method == "tools/call":
        return _call(session, rid, params)
    if method == "ping":
        return _ok(rid, {})
    if rid is None:
        # notifications/initialized, notifications/cancelled, and anything else
        # the client tells us about: no response, by definition.
        return None
    return _err(rid, -32601, "method not found: %r" % (method,))


def _call(session, rid, params):
    if not isinstance(params, dict):
        return _err(rid, -32602, "params must be an object")
    name = params.get("name")
    args = params.get("arguments") or {}
    if not isinstance(name, str):
        return _err(rid, -32602, "params.name must be a string")
    if not isinstance(args, dict):
        return _err(rid, -32602, "params.arguments must be an object")
    out = session.call(name, **args)
    return _ok(rid, {"content": [{"type": "text",
                                  "text": json.dumps(out, ensure_ascii=False,
                                                     default=str)}],
                     "isError": "error" in out})


def serve(session, stdin=None, stdout=None, log=None):
    """Newline-delimited JSON-RPC until stdin closes. -> responses written.

    The count includes a parse error, because a parse error is a response the
    client received; a test that counted only successful requests would report
    a corrupted stream as a quiet one.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    log = sys.stderr if log is None else log
    n = 0
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, _err(None, -32700, "parse error: %s" % exc))
            n += 1
            continue
        try:
            response = handle(session, request)
        except Exception as exc:  # noqa: BLE001 - a server must not die on a call
            response = _err(request.get("id") if isinstance(request, dict) else None,
                            -32603, "internal error: %s: %s"
                                    % (type(exc).__name__, exc))
        if response is not None:
            _write(stdout, response)
            n += 1
    return n


def _write(stdout, payload):
    stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    stdout.flush()


# ----------------------------------------------------------------------- cli
def _load(ckpt, db, ablate, max_calls, moves=(), red_first=False, patch=None,
          fearless=()):
    names = model_tools.ChampIndex(schema.names())
    predictor = champs = con = None
    if Path(ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        predictor = PolicyPredictor(ckpt, device="cpu")
        champs = names
    else:
        print("no checkpoint at %s - model tools unavailable" % ckpt,
              file=sys.stderr)
    if Path(db).exists():
        con = schema.connect(db)
    else:
        print("no database at %s - run build_db.py first" % db, file=sys.stderr)
    # `--moves` is the channel a frontend would use: a chronological move log,
    # which the store turns into the board the tools read. Nothing in the tool
    # layer can write to it.
    live = draft_state.DraftState(
        patch=patch, red_first=red_first,
        unavailable=tuple(draft_state.key(f) for f in fearless))
    for move in moves:
        live = live.apply(move)
    return Session(con=con, predictor=predictor, champs=champs,
                   ablate=ablate, max_calls=max_calls,
                   store=draft_state.Store(names, real=live))


SELF_TEST = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "self-test", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "sql", "arguments": {"query": "SELECT COUNT(*) FROM move"}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
     "params": {"name": "sql", "arguments": {"query": "DELETE FROM move"}}},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
     "params": {"name": "nope", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 6, "method": "no/such/method"},
    {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
     "params": {"name": "get_state", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
     "params": {"name": "clone_state", "arguments": {"name": "line"}}},
    {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
     "params": {"name": "apply_move",
                "arguments": {"state": "line", "champion": "Alistar"}}},
    {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
     "params": {"name": "apply_move",
                "arguments": {"state": "real", "champion": "Kled"}}},
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "pro" / "model" / "checkpoints"
                                          / "nextpick-sharedsplitthrough-las-dk-c-t1-r"
                                          / "best.pt"))
    ap.add_argument("--db", default=str(schema.DB_PATH))
    ap.add_argument("--ablate", default="")
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--moves", default="",
                    help="seed the live board with a chronological move log, "
                         "e.g. 'Kled,Olaf,Aatrox'")
    ap.add_argument("--red-first", action="store_true",
                    help="red has the first pick of the game")
    ap.add_argument("--fearless", default="",
                    help="champions picked in earlier games of this series; "
                         "they are illegal here and the head masks them")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="pipe canned messages through the server and exit")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    # The wire is stdout. Take it before anything else can write to it.
    wire = sys.stdout
    sys.stdout = sys.stderr
    for stream in (sys.stdin, wire):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    session = _load(args.ckpt, args.db,
                    {x.strip() for x in args.ablate.split(",") if x.strip()},
                    args.max_calls,
                    moves=[x.strip() for x in args.moves.split(",") if x.strip()],
                    red_first=args.red_first, patch=args.patch,
                    fearless=[x.strip() for x in args.fearless.split(",")
                              if x.strip()])
    print("mcp: serving %s (ablate=%s)"
          % (", ".join(session.names()) or "no tools",
             ",".join(sorted(session.ablate)) or "-"), file=sys.stderr)

    if args.self_test:
        import io
        buf = io.StringIO()
        n = serve(session, stdin=io.StringIO("\n".join(
            json.dumps(m) for m in SELF_TEST) + "\n"), stdout=buf)
        for line in buf.getvalue().splitlines():
            print(line[:300])
        print("\n%d responses; summary: %s" % (n, session.summary()),
              file=sys.stderr)
        return 0

    try:
        serve(session, stdout=wire)
    except KeyboardInterrupt:
        pass
    print("mcp: %s" % session.summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
