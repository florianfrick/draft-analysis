"""An OpenAI-compatible chat client, with tool calling, over urllib.

No dependency, and the same env contract the frontend's agent panel reads:
`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT`. That matters because
the two now have to agree about which model answered - a transcript that says
"the LLM said" without a model id is not reproducible.

The panel's client returns prose; this one returns the assistant message verbatim
(`content` plus normalised `tool_calls`), because the loop needs to know whether
the model asked for a tool before it decides whether to answer.

`tool_calls` is normalised to `[{id, name, args}]` with `args` already parsed:
every provider nests it slightly differently, and a JSON parse error inside the
loop is a harness bug, not a model error.
"""
from __future__ import annotations

import json
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com"
# DeepSeek's current id is `deepseek-flash`, served by DeepSeek-V4.1-Flash-1017.
# The legacy `deepseek-v4-flash` still resolves to the same model, but
# `deepseek-v4.1-flash` - the obvious guess, and what this file used to say -
# does not exist and 404s. DeepSeek defaults here because it is the model this
# experiment was specified against; any OpenAI-compatible endpoint works.
DEFAULT_MODEL = "deepseek-flash"

# Cloudflare's WAF returns 1010 for urllib's default `Python-urllib/x.y` UA
# (confirmed from Lambda and from a local probe). An explicit, honest UA gets
# through to the Worker's own auth and validation without spoofing a browser.
USER_AGENT = "DraftAssistantWeb/1.0"


class LLMError(RuntimeError):
    pass


def describe(env=None):
    """-> {model, base_url, has_key}. What a run would use, without using it."""
    env = os.environ if env is None else env
    return {"model": env.get("LLM_MODEL") or DEFAULT_MODEL,
            "base_url": (env.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            "has_key": bool((env.get("LLM_API_KEY") or "").strip())}


def load_dotenv(path=None, env=None):
    """Read KEY=VALUE lines from `.env` into `env`; never override what is set.

    The repo keeps its secrets in `.env` and docker compose injects them, so a
    host-side script that ignores the file is a dead end - the user sets a key
    and is told there is none. Returns the names it added, so a CLI can say
    which ones came from the file rather than the shell.

    Deliberately minimal: no interpolation, no `export`, no multiline values.
    A parser with opinions about `.env` is a dependency wearing a trench coat.
    """
    env = os.environ if env is None else env
    path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / ".env"
    if not path.exists():
        return []
    added = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not (env.get(key) or "").strip():
            env[key] = value
            added.append(key)
    return added


def _normalise(message):
    calls = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        except json.JSONDecodeError:
            args = {"__unparsed_arguments__": raw}
        calls.append({"id": c.get("id") or "call_%d" % len(calls),
                      "name": fn.get("name"), "args": args})
    return {"role": "assistant", "content": message.get("content") or "",
            "tool_calls": calls,
            # Audit only, and never sent back: `loop._to_wire` rebuilds the
            # wire message from `content`/`tool_calls`, so a reasoning trace
            # here cannot end up in the next request. A reasoning model that
            # spends its budget thinking and returns empty `content` looks
            # like a refusal otherwise, and the trace is what tells them apart.
            "reasoning": message.get("reasoning_content") or ""}


def client_from_env(env=None, temperature=0.2, max_tokens=None,
                    extra_headers=None):
    """-> callable(messages, tools=None) -> normalised assistant message.

    Returns None when no key is configured, which is the same contract the
    frontend uses: the caller decides whether that is an error or a degraded
    path. It is not an error here - a run without a key still exercises the
    graph and the tools, it just gets no answer.

    `max_tokens` is opt-in and absent from the body when None, because the
    loop's answers are prose of unknown length while a reranker eval asks for
    one name and pays for every token it does not need.

    `extra_headers` is for a proxy that needs request metadata the model does
    not. The Cloudflare Worker uses `X-Agent-Turn` to charge one agent turn
    rather than each of its LLM calls, so a turn id is generated once per
    client when the caller does not supply one. Providers ignore the header.
    """
    env = os.environ if env is None else env
    key = (env.get("LLM_API_KEY") or "").strip()
    if not key:
        return None
    base = (env.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = env.get("LLM_MODEL") or DEFAULT_MODEL
    timeout = float(env.get("LLM_TIMEOUT") or 60)

    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key,
               "User-Agent": USER_AGENT}
    has_turn = False
    for name, value in (extra_headers or {}).items():
        headers[name] = value
        if str(name).lower() == "x-agent-turn":
            has_turn = True
    if not has_turn:
        headers["X-Agent-Turn"] = secrets.token_urlsafe(12)

    def call(messages, tools=None):
        body = {"model": model, "messages": messages,
                "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise LLMError("HTTP %s from %s: %s" % (exc.code, base, detail))
        except urllib.error.URLError as exc:
            raise LLMError("cannot reach %s: %s" % (base, exc.reason))
        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError("unexpected response shape: %s" % exc)
        out = _normalise(msg)
        out["model"] = payload.get("model") or model
        # Token accounting, audit only and never sent back. The agent loop
        # makes several calls per question, so a cost estimate has to be able
        # to sum them - and an estimate from character counts is off by a
        # different factor for every tokenizer.
        out["usage"] = payload.get("usage") or {}
        # `length` is how a caller learns that a small `max_tokens` was spent
        # on reasoning tokens rather than on the answer.
        try:
            out["finish_reason"] = payload["choices"][0].get("finish_reason")
        except (KeyError, IndexError):
            out["finish_reason"] = None
        return out

    call.model = model
    return call
