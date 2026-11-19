// Thin proxy for the Android chat panel. The DeepSeek key lives here as a
// Worker secret; it is never shipped in the APK. The app sends an anonymous
// install id and pre-computed evidence from its on-device tools.

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "content-type, x-install-id",
};

const SYSTEM = `You are a draft assistant for professional League of Legends.
You are explaining an on-device model to a player, not replacing it.

The app has already run its tools. The numbered observations below are their
outputs. Treat them as measurements: use them, do not invent numbers, and do
not claim a number no observation contains.

There is NO three-sentence limit on your answer. Give the direct answer first,
then explain it in two to four short paragraphs or a short list. Be concrete,
but do not pad. If the observations are thin, say what you cannot tell.

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
  const maxTokens = Number(env.MAX_TOKENS || 900);

  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      messages,
      // Thinking is ON by default with effort "high". On a long board prompt
      // it can consume the entire max_tokens budget on reasoning_content and
      // leave content empty - the "DeepSeek returned an empty answer" error on
      // device. We want the explanation, so turn thinking off explicitly.
      thinking: { type: "disabled" },
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

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }
    if (request.method !== "POST") {
      return json({ error: "POST only" }, 405);
    }

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
