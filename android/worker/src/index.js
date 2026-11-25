// Two routes, one deployment. The DeepSeek key lives here as a Worker secret;
// it is never shipped in the APK or to the webapp.
//
//   POST /                       Android chat panel. Anonymous install id +
//                                pre-computed evidence; answer-only response.
//   POST /v1/chat/completions    OpenAI-compatible passthrough for the hosted
//                                webapp agent. Bearer token, verbatim messages
//                                and tools, verbatim assistant message back.
//
// The second route is what a tool-calling loop needs: it must see `tools`,
// `tool_choice` and `tool_calls`, which the first route intentionally drops.

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "content-type, authorization, x-install-id, x-agent-turn",
};

const SYSTEM = `You are a draft assistant for professional League of Legends.
You are explaining an on-device model to a player, not replacing it.

The app has already run its tools. The numbered observations below are their
outputs. Treat them as measurements: use them, do not invent numbers, and do
not claim a number no observation contains.

Keep the whole answer to at most EIGHT sentences. Give the direct answer
first, then use the remaining sentences for the explanation. Short paragraphs
or a short list are fine; every sentence counts toward the cap. Be concrete,
but do not pad. If the observations are thin, say what you cannot tell. You may
use **bold**, *italic* and inline code spans; the app renders them.

Facts you must keep straight:
- The next-move model is a trained set transformer over professional drafts.
  It is marginal: it predicts what pros picked in this situation, not what wins.
  It has no win probability and cannot tell blue from red.
- p_head (called "share" in the observations) is a share of the legal champion
  pool, not a probability of being correct. The head's top-1 is right about one
  time in five; its top-5 about half the time.
- The role assignment comes from champion pick history, one role per side and
  no duplicates. It tells you what a team still needs; it is not a prediction
  about this game.
- The fearless pool blocks picks but not bans. Never recommend a champion that
  is on the board or in the pool for a pick.
- Any composition archetype or strategic read is YOUR interpretation from the
  picks and roles. Label it as such. Do not present it as a model output.

When the question asks to justify the optimal next move: name the champion
first, then justify it from the observed shortlist and the open roles. If you
disagree with the head's #1, say why using only the observations; otherwise
say plainly that #1 is also your choice given the evidence.

When the question asks about team composition archetypes: describe each side
from its roles and picks, and say how certain the read is. Use terms like
front-to-back, pick, poke, dive, or scaling only when the observed picks
actually support them.`;

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

function clamp(text, max) {
  const s = String(text ?? "").trim();
  return s.length > max ? s.slice(0, max) + "…" : s;
}

async function sha256(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// Compare hashes, not the raw strings: both sides are 64 hex chars, so there
// is no length oracle and no early exit on the first differing byte.
async function bearerMatches(request, env) {
  const expected = String(env.WEBAPP_LLM_TOKEN || "").trim();
  if (!expected) return false;
  const header = String(request.headers.get("authorization") || "").trim();
  const match = /^Bearer\s+(.+)$/i.exec(header);
  if (!match) return false;
  const got = await sha256(match[1].trim());
  const want = await sha256(expected);
  let diff = 0;
  for (let i = 0; i < want.length; i += 1) diff |= got.charCodeAt(i) ^ want.charCodeAt(i);
  return diff === 0;
}

// An optional allowlist keeps a hosted deployment from being used as an open
// relay from unrelated pages. Unset means the old wildcard behaviour, so local
// curl and the Android app are unaffected.
function originAllowed(request, env) {
  const raw = String(env.WEBAPP_ORIGINS || "").trim();
  if (!raw) return true;
  const origin = request.headers.get("origin");
  if (!origin) return true;               // curl / server-side scripts
  return raw.split(",").map((s) => s.trim()).filter(Boolean).includes(origin);
}

const FORWARD_FIELDS = [
  "messages", "model", "temperature", "top_p", "max_tokens",
  "max_completion_tokens", "tools", "tool_choice", "response_format",
  "frequency_penalty", "presence_penalty", "stop", "seed", "n",
  "thinking", "reasoning_effort",
];

// A turn is one webapp request, which can make several LLM calls as the loop
// picks tools. KV remembers the turn id so the cap counts turns, not calls.
// Calls inside a turn are sequential, so the marker is set before the next one.
async function chargeTurn(request, env) {
  if (!env.CHAT_QUOTA) {
    return { error: "CHAT_QUOTA KV namespace is not bound", status: 503 };
  }
  const turn = String(request.headers.get("x-agent-turn") || "").trim();
  if (!/^[A-Za-z0-9_-]{8,64}$/.test(turn)) {
    return { error: "Missing or invalid X-Agent-Turn", status: 400 };
  }
  const day = new Date().toISOString().slice(0, 10);
  const turnKey = `at:${await sha256(`${day}|${turn}`)}`;
  if (await env.CHAT_QUOTA.get(turnKey)) return {};

  if (env.CHAT_BURST) {
    const burst = await env.CHAT_BURST.limit({ key: `agent:${turn}` });
    if (!burst.success) {
      return { error: "Too many agent turns in a short time. Try again soon.", status: 429 };
    }
  }

  const dailyKey = `ad:${day}`;
  const raw = await env.CHAT_QUOTA.get(dailyKey);
  let count = 0;
  try {
    count = raw ? JSON.parse(raw).count || 0 : 0;
  } catch {
    count = 0;
  }
  const limit = Number(env.AGENT_DAILY_LIMIT || 200);
  if (count >= limit) {
    return { error: `Daily agent turn limit reached (${limit}).`, status: 429 };
  }
  await env.CHAT_QUOTA.put(turnKey, "1", { expirationTtl: 90_000 });
  await env.CHAT_QUOTA.put(dailyKey, JSON.stringify({ count: count + 1, day }),
                          { expirationTtl: 90_000 });
  return {};
}

function buildMessages(body) {
  const question = clamp(body.question, 2_000);
  const evidence = (Array.isArray(body.evidence) ? body.evidence : [])
    .slice(0, 24)
    .map((line) => clamp(line, 800));

  const history = (Array.isArray(body.history) ? body.history : [])
    .slice(-6)
    .filter((m) => m && (m.role === "user" || m.role === "assistant"))
    .map((m) => ({ role: m.role, content: clamp(m.content, 2_000) }));

  const observations = evidence.length
    ? evidence.map((line, i) => `${i + 1}. ${line}`).join("\n")
    : "No observations were supplied. Answer from general knowledge and say so.";

  return [
    { role: "system", content: `${SYSTEM}\n\n## Observations from the app's tools\n${observations}` },
    ...history,
    { role: "user", content: question },
  ];
}

async function callDeepSeek(messages, env) {
  const base = String(env.DEEPSEEK_BASE_URL || "https://api.deepseek.com")
    .replace(/\/+$/, "");
  const model = String(env.DEEPSEEK_MODEL || "deepseek-flash");
  const maxTokens = Number(env.MAX_TOKENS || 1600);
  // Thinking stays on by preference. "low" keeps the hidden reasoning from
  // eating the whole token budget; set REASONING_EFFORT=high in the dashboard
  // if you want deeper reasoning (and raise MAX_TOKENS with it).
  const reasoningEffort = String(env.REASONING_EFFORT || "low");

  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      messages,
      // Explicit rather than relying on DeepSeek's default (enabled/high).
      thinking: { type: "enabled" },
      reasoning_effort: reasoningEffort,
      temperature: 0.35,
      max_tokens: maxTokens,
      stream: false,
    }),
  });

  if (!response.ok) {
    // Never echo the upstream body: it can contain request metadata.
    return { error: `DeepSeek returned HTTP ${response.status}` };
  }

  const payload = await response.json();
  const answer = String(payload?.choices?.[0]?.message?.content || "").trim();
  if (!answer) return { error: "DeepSeek returned an empty answer" };
  return { answer, usage: payload.usage ?? null };
}

// The webapp agent loop needs the OpenAI message shape, not a prose answer.
// Forward only fields it can legitimately send, add the Worker's reasoning and
// token defaults, and return the upstream JSON untouched.
async function handleOpenAI(request, env) {
  if (!originAllowed(request, env)) {
    return json({ error: "Origin is not allowed" }, 403);
  }
  if (!(await bearerMatches(request, env))) {
    return json({ error: "Missing or invalid bearer token" }, 401);
  }
  if (!env.DEEPSEEK_API_KEY) {
    return json({ error: "DEEPSEEK_API_KEY is not configured" }, 503);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json({ error: "JSON object required" }, 400);
  }
  if (body.stream === true) {
    return json({ error: "Streaming is not supported by this proxy" }, 400);
  }
  if (!Array.isArray(body.messages) || body.messages.length < 1 || body.messages.length > 64) {
    return json({ error: "messages must be an array of 1-64 messages" }, 400);
  }
  if (body.tools != null && (!Array.isArray(body.tools) || body.tools.length > 32)) {
    return json({ error: "tools must be an array of at most 32 specs" }, 400);
  }
  if (body.tool_choice != null && typeof body.tool_choice !== "string"
      && typeof body.tool_choice !== "object") {
    return json({ error: "tool_choice must be a string or object" }, 400);
  }

  const charge = await chargeTurn(request, env);
  if (charge.error) return json({ error: charge.error }, charge.status || 429);

  const upstream = {};
  for (const field of FORWARD_FIELDS) {
    if (body[field] !== undefined) upstream[field] = body[field];
  }
  const base = String(env.DEEPSEEK_BASE_URL || "https://api.deepseek.com")
    .replace(/\/+$/, "");
  const maxTokens = Number(env.MAX_TOKENS || 3000);
  const cap = Number(env.AGENT_MAX_TOKENS || maxTokens || 3000);
  const asked = Number(upstream.max_tokens ?? upstream.max_completion_tokens);
  if (!Number.isFinite(asked) || asked <= 0) {
    upstream.max_tokens = maxTokens;
  } else {
    const bounded = Math.min(asked, cap);
    delete upstream.max_completion_tokens;
    upstream.max_tokens = bounded;
  }
  // The Android route always thinks; keep that for the agent too unless a
  // caller explicitly opted out. `reasoning_effort=low` is the deployed value.
  if (upstream.thinking === undefined) upstream.thinking = { type: "enabled" };
  if (upstream.reasoning_effort === undefined) {
    upstream.reasoning_effort = String(env.REASONING_EFFORT || "low");
  }
  upstream.stream = false;

  let response;
  try {
    response = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${env.DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: String(upstream.model || env.DEEPSEEK_MODEL || "deepseek-flash"),
        ...upstream,
      }),
    });
  } catch {
    return json({ error: "Cannot reach the LLM upstream" }, 502);
  }
  if (!response.ok) {
    // Never echo the upstream body: it can contain request metadata.
    return json({ error: `LLM returned HTTP ${response.status}` }, 502);
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    return json({ error: "LLM returned invalid JSON" }, 502);
  }
  // Verbatim on purpose: content, tool_calls, finish_reason, reasoning_content,
  // model and usage all have to survive for the loop to work and be auditable.
  return json(payload, response.status);
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }
    if (request.method !== "POST") {
      return json({ error: "POST only" }, 405);
    }

    const path = new URL(request.url).pathname.replace(/\/+$/, "") || "/";
    if (path === "/v1/chat/completions" || path === "/chat/completions") {
      return handleOpenAI(request, env);
    }

    // Everything else stays the Android answer-only route.
    const installId = String(request.headers.get("x-install-id") || "").trim();
    if (!/^[A-Za-z0-9_-]{8,64}$/.test(installId)) {
      return json({ error: "Missing or invalid X-Install-Id" }, 400);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON body" }, 400);
    }
    if (!String(body?.question || "").trim()) {
      return json({ error: "Question is required" }, 400);
    }
    if (!env.DEEPSEEK_API_KEY) {
      return json({ error: "DEEPSEEK_API_KEY is not configured" }, 503);
    }

    // Burst control. The binding is optional so a dashboard-only deployment
    // still works; add it through wrangler for real abuse protection.
    if (env.CHAT_BURST) {
      const burst = await env.CHAT_BURST.limit({ key: installId });
      if (!burst.success) {
        return json({ error: "Too many questions in a short time. Try again soon." }, 429);
      }
    }

    // Daily cap. This is an abuse throttle, not authentication: the install id
    // is anonymous and can be changed. Pair it with the IP so switching the id
    // alone does not buy a fresh allowance from the same device.
    if (!env.CHAT_QUOTA) {
      return json({ error: "CHAT_QUOTA KV namespace is not bound" }, 503);
    }
    const day = new Date().toISOString().slice(0, 10);
    const ip = request.headers.get("cf-connecting-ip") || "";
    const quotaKey = `q:${await sha256(`${installId}|${ip}|${day}`)}`;
    const raw = await env.CHAT_QUOTA.get(quotaKey);
    let count = 0;
    try {
      count = raw ? JSON.parse(raw).count || 0 : 0;
    } catch {
      count = 0;
    }
    const limit = Number(env.DAILY_LIMIT || 30);
    if (count >= limit) {
      return json({ error: `Daily question limit reached (${limit}).` }, 429);
    }

    const result = await callDeepSeek(buildMessages(body), env);
    if (result.error) {
      return json({ error: result.error }, 502);
    }

    await env.CHAT_QUOTA.put(
      quotaKey,
      JSON.stringify({ count: count + 1, day }),
      { expirationTtl: 90_000 },
    );

    return json({
      answer: result.answer,
      usage: result.usage,
      remaining: Math.max(0, limit - count - 1),
    });
  },
};
