"use strict";

const ROLE_NAMES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];

let CHAMPIONS = [];
let BY_IDX = {};
let GAMES = [];
let MODELS = [];
let CURRENT = -1;
const SELECTED = new Set();
let searchTimer = null;

const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function pct(x) {
  return (Number(x || 0) * 100).toFixed(1) + "%";
}

function imgTag(url, cls) {
  const c = cls ? " " + cls : "";
  if (!url) return `<span class="ic${c} broken"></span>`;
  return `<img class="ic${c}" src="${esc(url)}" loading="lazy" ` +
    `onerror="this.classList.add('broken')" alt="">`;
}

function shortLabel(label) {
  return String(label || "")
    .replace(/^\[[^\]]+\]\s*/, "")     // drop the "[kind] " tag
    .replace(/^run_\d+_\d+_/, "");     // drop the run timestamp prefix
}

function roleName(r) {
  return ROLE_NAMES[r] || "?";
}

// --------------------------------------------------------------- init
async function init() {
  const [champRes, ckptRes, draftRes] = await Promise.all([
    fetch("/api/champions").then((r) => r.json()),
    fetch("/api/checkpoints").then((r) => r.json()),
    fetch("/api/drafts?limit=200").then((r) => r.json()),
  ]);
  CHAMPIONS = champRes.champions || [];
  BY_IDX = {};
  CHAMPIONS.forEach((c) => { BY_IDX[c.idx] = c; });
  MODELS = ckptRes.checkpoints || [];
  // Default selection: the shipped pro-only pair - the patch-embedded
  // policy (the frontend default) and its masked init - not the no-patch
  // ablation control. Fall back to any policy if the pair is absent.
  const shipped = MODELS.filter(
    (m) => /proonly/.test(m.path) && /-patch-s0/.test(m.path));
  if (shipped.length) {
    shipped.forEach((m) => SELECTED.add(m.path));
  } else {
    const pol = MODELS.find((m) => m.kind === "policy");
    if (pol) SELECTED.add(pol.path);
  }
  renderModels();
  GAMES = draftRes.drafts || [];
  renderGames();
  if (GAMES.length) selectGame(GAMES[0].i);
  updateGenerate();
  // Show the default pair's breakdown immediately (scoring is <1s).
  if (CURRENT >= 0 && SELECTED.size) generate();
}

// --------------------------------------------------------------- games
function renderGames() {
  const sel = $("game-select");
  sel.innerHTML = GAMES.map((d) =>
    `<option value="${d.i}">${esc(d.blue)} vs ${esc(d.red)} ` +
    `&middot; ${esc(d.league || "")} &middot; ${esc(d.date)} ` +
    `&middot; patch ${esc(d.patch || "?")}</option>`).join("");
  if (GAMES.length) sel.value = String(GAMES[0].i);
}

function selectGame(i) {
  CURRENT = i;
  const d = GAMES.find((g) => g.i === i);
  if (d) {
    $("game-meta").textContent =
      `${d.blue} vs ${d.red} · ${d.league || ""} · ${d.date} · ` +
      `patch ${d.patch || "?"} · ${d.winner} win`;
  }
  updateGenerate();
}

async function refreshGames(q) {
  const res = await fetch("/api/drafts?limit=200&q=" + encodeURIComponent(q || ""));
  const data = await res.json();
  GAMES = data.drafts || [];
  renderGames();
  if (GAMES.length) selectGame(GAMES[0].i);
  else { CURRENT = -1; updateGenerate(); }
}

// --------------------------------------------------------------- models
function renderModels() {
  const box = $("model-list");
  box.innerHTML = MODELS.map((m) => {
    const checked = SELECTED.has(m.path) ? "checked" : "";
    return `<label class="model-chk" title="${esc(m.path)}">
      <input type="checkbox" data-path="${esc(m.path)}" ${checked}>
      <span class="kind k-${esc(m.kind)}">${esc(m.kind)}</span>
      <span class="mpath">${esc(shortLabel(m.label))}</span>
    </label>`;
  }).join("");
  box.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) SELECTED.add(cb.dataset.path);
      else SELECTED.delete(cb.dataset.path);
      updateGenerate();
    });
  });
}

async function loadCheckpoints() {
  const res = await fetch("/api/checkpoints");
  const data = await res.json();
  MODELS = data.checkpoints || [];
  renderModels();
  updateGenerate();
}

function updateGenerate() {
  $("btn-generate").disabled = !(CURRENT >= 0 && SELECTED.size);
}

// --------------------------------------------------------------- generate
async function generate() {
  if (CURRENT < 0 || !SELECTED.size) return;
  const btn = $("btn-generate");
  btn.disabled = true;
  $("status").textContent =
    `Scoring ${SELECTED.size} model(s) over the draft…`;
  try {
    const res = await fetch("/api/breakdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_i: CURRENT, models: [...SELECTED] }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    renderResult(data);
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// --------------------------------------------------------------- render
function renderScoreboard(models) {
  const ok = models.filter((m) => !m.error);
  const errs = models.filter((m) => m.error);
  let html = `<h2>Scoreboard</h2><table class="sb"><thead><tr>
    <th>Model</th><th>Kind</th><th>Top-1</th><th>Top-5</th>
    <th>Mean rank</th><th>Moves</th></tr></thead><tbody>`;
  ok.forEach((m) => {
    const s = m.score;
    const val = (m.val_top5 != null)
      ? `<span class="muted"> · val top5 ${pct(m.val_top5)}</span>` : "";
    html += `<tr>
      <td class="lbl">${esc(m.label)}${val}</td>
      <td><span class="kind k-${esc(m.kind)}">${esc(m.kind)}</span></td>
      <td class="num ${s.top1 >= 0.5 ? "good" : ""}">${pct(s.top1)}</td>
      <td class="num ${s.top5 >= 0.7 ? "good" : ""}">${pct(s.top5)}</td>
      <td class="num">${Number(s.mean_rank).toFixed(2)}</td>
      <td class="num muted">${s.n}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  if (errs.length) {
    html += `<div class="err">` + errs.map((e) =>
      `<div>${esc(e.path)}: ${esc(e.error)}</div>`).join("") + `</div>`;
  }
  return html;
}

function modelCell(m, moveI) {
  const r = m.rows[String(moveI)];
  if (!r) {
    return `<td class="mcol na" title="masked models do not predict bans">—</td>`;
  }
  const cls = r.hit_top1 ? "hit1" : (r.hit_top5 ? "hit5" : "miss");
  const tip = `predicted ${r.top1_name}; real rank #${r.target_rank} ` +
    `(${pct(r.target_prob)})`;
  return `<td class="mcol pred ${cls}" title="${esc(tip)}">
    ${imgTag(r.top1_img, "small")}<span class="rk">#${r.target_rank}</span>
  </td>`;
}

function modelDetail(m, moveI) {
  const r = m.rows[String(moveI)];
  const head = `<div class="dhead">${esc(shortLabel(m.label))}`;
  if (!r) {
    return `<div class="dcol">${head} <span class="muted">· n/a</span></div>
      <div class="muted">masked models do not predict bans</div></div>`;
  }
  const chips = r.top5.map((c) =>
    `<span class="chip ${c.idx === r.real_idx ? "isreal" : ""}">
      ${imgTag(c.img)}<span class="cn">${esc(c.name)}</span>
      <span class="cp">${pct(c.prob)}</span></span>`).join("");
  return `<div class="dcol">
    ${head} <span class="muted">· real rank #${r.target_rank} ·
      p=${pct(r.target_prob)}</span></div>
    <div class="chips">${chips}</div>
  </div>`;
}

function renderTable(data) {
  const models = data.models.filter((m) => !m.error);
  const moves = data.game.moves || [];
  let html = `<h2>Per-move breakdown</h2>
    <div class="legend">
      <span class="lg hit1">#1</span> real champ was the model's top pick
      <span class="lg hit5">top-5</span> real champ in the top 5
      <span class="lg miss">miss</span> outside the top 5
      <span class="muted">· click a row for each model's top-5</span>
    </div>
    <table class="bd"><thead><tr>
      <th>#</th><th>Side</th><th>Move</th><th>Real</th>
      ${models.map((m) =>
        `<th class="mcol" title="${esc(m.label)}">${esc(shortLabel(m.label))}</th>`).join("")}
    </tr></thead><tbody>`;
  moves.forEach((mv) => {
    const role = (mv.role != null)
      ? `<span class="role">${roleName(mv.role)}</span>` : "";
    html += `<tr class="mv" data-move="${mv.i}">
      <td class="num">${mv.i + 1}</td>
      <td class="side s-${mv.side}">${mv.side === 0 ? "BLUE" : "RED"}</td>
      <td><span class="kind ${mv.kind}">${mv.kind === "ban" ? "BAN" : "PICK"}</span>${role}</td>
      <td class="real"><span class="real-in">${imgTag(mv.img, "small")}<span class="nm">${esc(mv.name)}</span></span></td>
      ${models.map((m) => modelCell(m, mv.i)).join("")}
    </tr>`;
    html += `<tr class="detail hidden" data-detail="${mv.i}">
      <td colspan="${4 + models.length}">
        <div class="detail-grid">
          ${models.map((m) => modelDetail(m, mv.i)).join("")}
        </div>
      </td></tr>`;
  });
  html += `</tbody></table>`;
  return html;
}

function wireRows() {
  document.querySelectorAll("tr.mv").forEach((tr) => {
    tr.addEventListener("click", () => {
      const d = document.querySelector(
        `tr.detail[data-detail="${tr.dataset.move}"]`);
      if (d) d.classList.toggle("hidden");
      tr.classList.toggle("open");
    });
  });
}

function renderResult(data) {
  $("scoreboard").innerHTML = renderScoreboard(data.models);
  $("scoreboard").classList.remove("hidden");
  $("table-wrap").innerHTML = renderTable(data);
  $("table-wrap").classList.remove("hidden");
  wireRows();
  const g = data.game;
  $("game-meta").textContent =
    `${g.blue} vs ${g.red} · ${g.league || ""} · ${g.date} · ` +
    `patch ${g.patch || "?"} · ${g.winner} win`;
}

// --------------------------------------------------------------- wire up
$("btn-generate").addEventListener("click", generate);
$("btn-refresh").addEventListener("click", () => refreshGames($("game-search").value));
$("game-select").addEventListener("change", (e) => {
  selectGame(parseInt(e.target.value, 10));
  generate();
});
$("game-search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => refreshGames(e.target.value), 250);
});
$("btn-upload").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("status").textContent = "Uploading " + f.name + "…";
  try {
    const res = await fetch("/api/upload?name=" + encodeURIComponent(f.name), {
      method: "POST",
      body: await f.arrayBuffer(),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    SELECTED.add(data.model.path);
    await loadCheckpoints();
    $("status").textContent = "Uploaded " + f.name;
  } catch (err) {
    $("status").textContent = "Upload failed: " + err.message;
  } finally {
    e.target.value = "";
  }
});

init();
