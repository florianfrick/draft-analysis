const S = {
  meta: null, champs: [], byRiot: {}, metrics: [], byKey: {},
  patch: "", tab: "draft", metric: "win",
  showEstimate: false, showGames: false,
  blue: [null, null, null, null, null],
  red: [null, null, null, null, null],
  m: { x: null, y: null, mode: "with", role: "" },
  pickTarget: null,
  draft: null, openChamp: null, stat: null,
  req: 0, sugReq: 0, detail: {}, sugCache: null, openSug: null,
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const img = (c, cls) => c && c.img
  ? `<img class="${cls || ""}" src="${c.img}" alt="" loading="lazy">` : "";
const pct = (v, digits = 1) => v === null || v === undefined
  ? "—" : (v * 100).toFixed(digits) + "%";
const num = (v, digits = 0) => v === null || v === undefined
  ? "—" : Number(v).toFixed(digits);
const kfmt = (v) => v === null || v === undefined ? "—"
  : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Number(v).toFixed(1);
const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? "" : "s"}`;

const OBJ_LABEL = {
  dragon: "dragons", baron: "barons", tower: "towers",
  inhib: "inhibitors", herald: "heralds", horde: "grubs",
};

/* the server tags every metric with a group; this is the only place that
   decides how those groups are labelled and ordered */
const GROUPS = {
  core: "box score", rate: "per minute", lane: "lane phase",
  tempo: "tempo & firsts", team: "team & objectives", combat: "combat profile",
};
const secs = (v) => {
  const s = Math.round(v);
  return Math.floor(s / 60) + ":" + String(Math.abs(s) % 60).padStart(2, "0");
};

/* metric formatting, driven entirely by the server's registry */
function fmtMetric(key, v) {
  const m = S.byKey[key];
  if (v === null || v === undefined) return "—";
  if (!m) return num(v, 1);
  if (m.fmt === "pct") return (v * 100).toFixed(1) + "%";
  if (m.fmt === "int") return Math.round(v).toLocaleString();
  if (m.fmt === "num2") return Number(v).toFixed(2);
  if (m.fmt === "sec") return secs(v);
  return Number(v).toFixed(1);
}
function fmtDelta(key, d) {
  const m = S.byKey[key];
  if (d === null || d === undefined) return "";
  if (!m) return (d > 0 ? "+" : "") + d.toFixed(1);
  if (m.fmt === "pct") return (d > 0 ? "+" : "") + (d * 100).toFixed(1) + " pts";
  if (m.fmt === "int") return (d > 0 ? "+" : "") + Math.round(d).toLocaleString();
  if (m.fmt === "sec") return (d > 0 ? "+" : "") + Math.round(d) + "s";
  return (d > 0 ? "+" : "") + d.toFixed(1);
}
const signOf = (key) => (S.byKey[key] && S.byKey[key].better < 0 ? -1 : 1);
/* a delta is "good" when it moves in the metric's preferred direction */
const dclsOf = (key, d) => d === null || d === undefined || Math.abs(d) < 1e-9
  ? "flat" : (d * signOf(key) > 0 ? "up" : "down");
const dsgn = (v, digits = 1, scale = 1) => v === null || v === undefined
  ? "" : (v > 0 ? "+" : "") + (v * scale).toFixed(digits);

function conf(n) {
  const t = S.meta ? S.meta.thresholds : { strong: 200, ok: 50, weak: 20 };
  if (n === 0) return ["none", "never played"];
  if (n >= t.strong) return ["solid", "solid sample"];
  if (n >= t.ok) return ["ok", "usable sample"];
  if (n >= t.weak) return ["weak", "weak sample"];
  return ["noise", "noise"];
}
const badge = (n) => {
  const [cls, label] = conf(n);
  return `<span class="badge ${cls}">${label}</span>`;
};

/* ------------------------------------------------------------------ init */
async function init() {
  const r = await fetch("/api/meta");
  S.meta = await r.json();
  S.champs = S.meta.champions;
  S.champs.forEach((c) => { S.byRiot[c.riot] = c; });
  S.metrics = S.meta.metrics;
  S.metrics.forEach((m) => { S.byKey[m.key] = m; });
  S.patch = S.meta.patches.length ? S.meta.patches[0].patch : "";

  $("corpus").innerHTML =
    `<b>${S.meta.games.toLocaleString()}</b> games · ` +
    `<b>${S.meta.players.toLocaleString()}</b> player-rows · ` +
    `${S.champs.length} champions`;

  $("patch").innerHTML = S.meta.patches
    .map((p) => `<option value="${p.patch}">${p.patch} (${p.games.toLocaleString()})</option>`)
    .join("") + `<option value="">all patches</option>`;
  $("patch").value = S.patch;

  const groups = GROUPS;
  $("metric").innerHTML = Object.keys(groups).map((g) => {
    const opts = S.metrics.filter((m) => m.group === g && !m.hidden)
      .map((m) => `<option value="${m.key}">${m.label}</option>`).join("");
    return `<optgroup label="${groups[g]}">${opts}</optgroup>`;
  }).join("");
  $("metric").value = S.metric;

  $("m-role").innerHTML = `<option value="">any</option>` +
    S.meta.roles.map((r) => `<option value="${r}">${r.toLowerCase()}</option>`).join("");

  renderSlots();
  renderMatchupPicks();
  wire();
  readHash();
}

/* A draft is shareable: #103,64,84/113,126
   A matchup is shareable: #m=103,64,vs
   Revealed cards: ?est=1&games=1 */
function readHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ""));
  const q = new URLSearchParams(location.search);
  if (q.get("est") === "1") { S.showEstimate = true; $("btn-estimate").classList.add("on"); }
  if (q.get("games") === "1") { S.showGames = true; $("btn-games").classList.add("on"); }
  if (!h) return;
  if (h.startsWith("m=")) {
    const [a, b, mode] = h.slice(2).split(",");
    S.m.x = S.byRiot[parseInt(a, 10)] || null;
    S.m.y = S.byRiot[parseInt(b, 10)] || null;
    if (mode === "vs" || mode === "with") {
      S.m.mode = mode;
      document.querySelectorAll("#m-mode button").forEach((x) => {
        x.classList.toggle("active", x.dataset.mode === mode);
      });
    }
    showTab("matchup");
    renderMatchupPicks();
    loadMatchup();
    return;
  }
  const [b, r] = h.split("/");
  const parse = (s) => (s || "").split(",").map((x) => parseInt(x, 10))
    .filter((x) => !isNaN(x)).map((riot) => S.byRiot[riot]).filter(Boolean);
  const blue = parse(b), red = parse(r);
  if (!blue.length && !red.length) return;
  S.blue = [0, 1, 2, 3, 4].map((i) => blue[i] || null);
  S.red = [0, 1, 2, 3, 4].map((i) => red[i] || null);
  renderSlots();
  loadDraft();
}

function showTab(tab) {
  S.tab = tab;
  document.querySelectorAll("#tabs button").forEach((x) => {
    x.classList.toggle("active", x.dataset.tab === tab);
  });
  $("view-draft").classList.toggle("hidden", tab !== "draft");
  $("view-matchup").classList.toggle("hidden", tab !== "matchup");
}

function writeHash() {
  const b = S.blue.filter(Boolean).map((c) => c.riot).join(",");
  const r = S.red.filter(Boolean).map((c) => c.riot).join(",");
  history.replaceState(null, "", b || r ? `#${b}/${r}` : location.pathname);
}

/* ------------------------------------------------------------------ wiring */
function wire() {
  $("patch").onchange = () => {
    S.patch = $("patch").value;
    if (S.tab === "draft") loadDraft(); else loadMatchup();
  };

  document.querySelectorAll("#tabs button").forEach((b) => {
    b.onclick = () => {
      showTab(b.dataset.tab);
      if (S.tab === "matchup") loadMatchup();
    };
  });

  $("metric").onchange = () => {
    S.metric = $("metric").value;
    if (S.draft) renderDraft(S.draft);
  };
  $("btn-estimate").onclick = () => {
    S.showEstimate = !S.showEstimate;
    $("btn-estimate").classList.toggle("on", S.showEstimate);
    if (S.draft) renderDraft(S.draft);
  };
  $("btn-games").onclick = () => {
    S.showGames = !S.showGames;
    $("btn-games").classList.toggle("on", S.showGames);
    if (S.draft) renderDraft(S.draft);
  };

  $("btn-clear").onclick = () => {
    S.blue = [null, null, null, null, null];
    S.red = [null, null, null, null, null];
    S.draft = null;
    renderSlots();
    renderDraftEmpty();
  };

  $("btn-random").onclick = async () => {
    const d = await (await fetch(`/api/draft?patch=${S.patch}&sides=any`)).json();
    const g = d.games && d.games[0];
    if (!g) return;
    const find = (name) => S.champs.find((c) => c.name === name);
    S.blue = g.blue.map((n) => find(n) || null);
    S.red = g.red.map((n) => find(n) || null);
    renderSlots();
    loadDraft();
  };

  document.querySelectorAll("#m-mode button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#m-mode button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      S.m.mode = b.dataset.mode;
      loadMatchup();
    };
  });
  $("m-role").onchange = () => { S.m.role = $("m-role").value; loadMatchup(); };
  $("m-x").onclick = () => openPicker("x");
  $("m-y").onclick = () => openPicker("y");

  $("picker-close").onclick = closePicker;
  $("picker").onclick = (e) => { if (e.target.id === "picker") closePicker(); };
  $("picker-search").oninput = (e) => fillPicker(e.target.value);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePicker();
  });
}

/* ------------------------------------------------------------------ slots */
function renderSlots() {
  [["blue", "slots-blue", S.blue], ["red", "slots-red", S.red]].forEach(([side, id, arr]) => {
    const host = $(id);
    host.innerHTML = "";
    arr.forEach((c, i) => {
      const s = el("div", "slot" + (c ? "" : " empty"));
      if (c) {
        s.innerHTML = img(c) + `<span class="nm">${c.name}</span><span class="x">✕</span>`;
        s.querySelector(".x").onclick = (e) => {
          e.stopPropagation();
          arr[i] = null;
          renderSlots();
          loadDraft();
        };
      } else {
        s.innerHTML = `+ ${side} ${i + 1}`;
      }
      s.onclick = () => openPicker({ side, i });
      host.appendChild(s);
    });
  });
}

/* ------------------------------------------------------------------ picker */
function openPicker(target) {
  S.pickTarget = target;
  $("picker").classList.remove("hidden");
  $("picker-search").value = "";
  $("picker-title").textContent = typeof target === "string"
    ? "Pick the " + (target === "x" ? "first" : "second") + " champion"
    : `Pick ${target.side} ${target.i + 1}`;
  fillPicker("");
  setTimeout(() => $("picker-search").focus(), 30);
}
function closePicker() { $("picker").classList.add("hidden"); }

function fillPicker(q) {
  const taken = new Set([...S.blue, ...S.red].filter(Boolean).map((c) => c.riot));
  const grid = $("picker-grid");
  grid.innerHTML = "";
  const needle = q.trim().toLowerCase();
  S.champs
    .filter((c) => !needle || c.name.toLowerCase().includes(needle))
    .forEach((c) => {
      const b = el("button", taken.has(c.riot) ? "taken" : "",
        img(c) + `<span>${c.name}</span>`);
      if (taken.has(c.riot)) b.style.opacity = .35;
      b.onclick = () => choose(c);
      grid.appendChild(b);
    });
}

function choose(c) {
  const t = S.pickTarget;
  if (typeof t === "string") {
    S.m[t] = c;
    renderMatchupPicks();
    closePicker();
    loadMatchup();
    return;
  }
  S[t.side][t.i] = c;
  renderSlots();
  closePicker();
  loadDraft();
}

/* ------------------------------------------------------------------ draft */
let draftTimer = null;
function loadDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(async () => {
    const blue = S.blue.filter(Boolean).map((c) => c.riot);
    const red = S.red.filter(Boolean).map((c) => c.riot);
    if (!blue.length && !red.length) { S.draft = null; renderDraftEmpty(); writeHash(); return; }
    writeHash();
    const t0 = performance.now();
    const r = await fetch(`/api/draft?blue=${blue.join(",")}&red=${red.join(",")}` +
      `&patch=${encodeURIComponent(S.patch)}&sides=any`);
    const d = await r.json();
    $("latency").textContent = `${Math.round(performance.now() - t0)} ms · live scan`;
    if (d.error) { $("draft-results").innerHTML = `<div class="card">${d.error}</div>`; return; }
    S.draft = d;
    S.openChamp = null;
    S.openSug = null;
    renderDraft(d);
  }, 40);
}

function renderDraftEmpty() {
  $("draft-results").innerHTML =
    `<div class="empty">Pick a champion to start. The count below is how many
     real games contain everything you have selected.</div>`;
  $("toolbar").classList.add("hidden");
  $("latency").textContent = "";
}

function renderDraft(d) {
  $("toolbar").classList.remove("hidden");
  renderOverall(d);
}

/* ------------------------------------------------------------------ overall */
function renderOverall(d) {
  const host = $("draft-results");
  host.innerHTML = "";
  const t = S.meta.thresholds;

  host.appendChild(heroCard(d));
  if (d.team) host.appendChild(teamCard(d));
  if (d.champ_stats && d.champ_stats.length && d.n > 0) host.appendChild(champCard(d));
  if (d.pairs && d.pairs.length && d.selected > 1) host.appendChild(matrixCard(d));
  if (d.selected) host.appendChild(suggestHost(d));
  if (d.drop_one && d.drop_one.length > 1 && d.n < t.ok) host.appendChild(dropCard(d));
  if (S.showEstimate && d.estimate && d.estimate.length) host.appendChild(estimateCard(d));
  if (S.showGames && d.games && d.games.length) host.appendChild(gamesCard(d));
}

function heroCard(d) {
  const t = S.meta.thresholds;
  const verdict = d.n === 0
    ? "No game in the corpus contains this exact set."
    : d.n >= t.strong
      ? "Enough games to read as a real effect."
      : d.n >= t.ok
        ? "Directionally useful, but check the deltas are not noise."
        : d.n >= t.weak
          ? "Too few games to separate from luck. Use the pair data below."
          : "A handful of games. Treat any average here as anecdote.";
  const card = el("div", "card");
  card.innerHTML = `
    <div class="hero">
      <div class="num">${d.n.toLocaleString()}<small>${d.n === 1 ? "game" : "games"}</small></div>
      ${badge(d.n)}
      <div class="sub">${verdict}</div>
    </div>`;
  return card;
}

function teamCard(d) {
  const tm = d.team;
  const card = el("div", "card");
  card.appendChild(el("h3", null, "How the draft plays out"));
  if (!tm || !tm.n || tm.sel_win_rate === null) {
    card.appendChild(el("p", "muted", "No matching game to score."));
    return card;
  }
  const rate = tm.sel_win_rate;
  const wins = tm.sel_wins;
  const losses = tm.n - wins;
  const sideName = (s) => (s === "blue" ? "blue" : "red") + " selection";
  const other = tm.subject === "blue" ? "red" : "blue";
  const gd = tm.gold_diff === null ? "" : `${tm.gold_diff > 0 ? "+" : ""}${Math.round(tm.gold_diff).toLocaleString()} gold`;
  const kd = tm.kill_diff === null ? "" : `${tm.kill_diff > 0 ? "+" : ""}${tm.kill_diff.toFixed(1)} kills`;
  const obj = tm.obj_diff || {};
  const objBits = Object.keys(OBJ_LABEL)
    .filter((k) => obj[k] !== null && obj[k] !== undefined)
    .map((k) => `${obj[k] > 0 ? "+" : ""}${obj[k].toFixed(2)} ${OBJ_LABEL[k]}`)
    .join(", ");
  const both = d.blue.length && d.red.length;

  let html = `<div class="versus ${both ? "" : "solo"}">
    <div class="vside">
      <div class="vname"><span class="dot ${tm.subject}"></span>${sideName(tm.subject)}</div>
      <div class="vbig">${pct(rate)}</div>
      <div class="tiny muted">${wins.toLocaleString()} of ${tm.n.toLocaleString()} games</div>
    </div>`;
  if (both) {
    html += `<div class="vmid">
        <div class="vbar"><i class="b" style="width:${(rate * 100).toFixed(1)}%"></i><i class="r" style="width:${((1 - rate) * 100).toFixed(1)}%"></i></div>
        <div class="tiny muted" style="margin-top:8px">
          ${sideName(tm.subject)} won ${wins.toLocaleString()} · ${sideName(other)} won ${losses.toLocaleString()}
          ${gd || kd ? `· ${[gd, kd].filter(Boolean).join(", ")} per game for ${sideName(tm.subject)}` : ""}
        </div>
      </div>
      <div class="vside right">
        <div class="vname"><span class="dot ${other}"></span>${sideName(other)}</div>
        <div class="vbig">${pct(1 - rate)}</div>
        <div class="tiny muted">${losses.toLocaleString()} of ${tm.n.toLocaleString()} games</div>
      </div>`;
  } else {
    html += `<div class="vmid"><div class="tiny muted">
      ${[gd, kd].filter(Boolean).join(", ") || "—"} per game</div></div>`;
  }
  html += `</div>`;
  if (objBits) {
    html += `<p class="tiny muted objline" style="margin:10px 0 0">
      Objectives per game for the ${sideName(tm.subject)}: ${objBits}
      <span class="muted">— from the team block of every matching game</span></p>`;
  }
  html += `<p class="tiny muted" style="margin:12px 0 0">
      Games are matched either way round: this selection was blue in roughly half
      of them and red in the other half, and the win rate is always read for
      <em>your</em> team. ${both ? "" : "Add champions to the other side to get a head-to-head."}</p>`;
  card.innerHTML += html;
  return card;
}

/* ---- champion table: click a champion for its full line, a stat for a strip */
const CORE_COLS = ["win", "kills", "deaths", "assists", "gold", "dmg", "cs", "vision"];

function champCard(d) {
  /* the box score always shows; a non-core metric from the selector is appended
     so that "gold / min" or "kill participation" can be inspected per champion */
  const cols = S.metric && !CORE_COLS.includes(S.metric)
    ? [...CORE_COLS, S.metric] : CORE_COLS;
  const card = el("div", "card");
  card.appendChild(el("h3", null, d.n === 1
    ? "Champions in that game" : `Champions in those ${plural(d.n, "game")}`));

  const sortKey = S.stat || null;
  const rows = d.champ_stats.slice().sort((a, b) => {
    if (sortKey) {
      const va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return signOf(sortKey) * (vb - va);
    }
    return b.n - a.n;
  });

  let html = `<div class="hint tiny muted">click a champion for its full stat line ·
    click a column to see it across the draft</div>
    <table class="stats champ-table"><thead><tr><th>champion</th><th>n</th>`;
  cols.forEach((k) => {
    html += `<th class="clickable ${S.stat === k ? "sel" : ""}" data-k="${k}">
      ${S.byKey[k].label}</th>`;
  });
  html += `</tr></thead><tbody>`;
  rows.forEach((r) => {
    const c = S.byRiot[r.riot];
    const open = S.openChamp === r.riot;
    html += `<tr class="crow ${open ? "open" : ""}" data-riot="${r.riot}">
      <td class="champ">${img(c)}${r.name}</td>
      <td>${r.n.toLocaleString()}</td>`;
    cols.forEach((k) => {
      const base = d.baseline && d.baseline[r.riot] ? d.baseline[r.riot][k] : null;
      const v = r[k];
      const dl = v === null || v === undefined || base === null || base === undefined
        ? null : v - base;
      html += `<td class="${S.stat === k ? "selcol" : ""}">${fmtMetric(k, r[k])}
        ${S.stat === k && dl !== null ? `<span class="sub ${dclsOf(k, dl)}">${fmtDelta(k, dl)}</span>` : ""}</td>`;
    });
    html += `</tr>`;
    if (open) {
      html += `<tr class="detail"><td colspan="${cols.length + 2}">
        <div class="dbody">${detailBody(r.riot)}</div></td></tr>`;
    }
  });
  html += `</tbody></table>`;

  if (S.stat) html += statStrip(d, S.stat, rows);
  card.innerHTML += html;

  card.querySelectorAll("th.clickable").forEach((th) => {
    th.onclick = () => {
      S.stat = S.stat === th.dataset.k ? null : th.dataset.k;
      renderDraft(S.draft);
    };
  });
  card.querySelectorAll("tr.crow").forEach((tr) => {
    tr.onclick = () => toggleChamp(Number(tr.dataset.riot));
  });
  return card;
}

function statStrip(d, key, rows) {
  const m = S.byKey[key];
  const vals = rows.map((r) => r[key]).filter((v) => v !== null && v !== undefined);
  if (!vals.length) return "";
  const max = Math.max(...vals);
  const min = Math.min(...vals, 0);
  const span = max - min || 1;
  let html = `<div class="strip"><div class="striptitle">${m.label} per champion
    <span class="tiny muted">— bar length is the value, the tick is that champion's own
    average over the patch</span></div>`;
  rows.forEach((r) => {
    const c = S.byRiot[r.riot];
    const v = r[key];
    const w = v === null || v === undefined ? 0 : ((v - min) / span) * 100;
    const base = d.baseline && d.baseline[r.riot] ? d.baseline[r.riot][key] : null;
    const bw = base === null || base === undefined ? null : ((base - min) / span) * 100;
    html += `<div class="striprow">
      ${img(c)}
      <span class="sn">${r.name}</span>
      <div class="sbar"><i style="width:${Math.max(0, w).toFixed(1)}%"></i>
        ${bw === null ? "" : `<b style="left:${Math.max(0, Math.min(100, bw)).toFixed(1)}%"></b>`}</div>
      <span class="sv">${fmtMetric(key, v)}</span>
    </div>`;
  });
  html += `</div>`;
  return html;
}

function detailBody(riot) {
  const cd = S.detail[riot];
  if (!cd) return `<div class="tiny muted">loading…</div>`;
  const groups = GROUPS;
  let html = `<div class="dgrid">`;
  Object.keys(groups).forEach((g) => {
    /* a partial stat with no rows in these games has nothing to say */
    const ms = S.metrics.filter((m) => m.group === g)
      .filter((m) => !(m.partial && !cd.in_games["n_" + m.key]));
    html += `<div class="dgroup"><div class="dgt">${groups[g]}</div>`;
    if (!ms.length) {
      html += `<div class="tiny muted">not recorded in these games</div>`;
    }
    ms.forEach((m) => {
      const v = cd.in_games[m.key];
      const dl = cd.delta[m.key];
      const pn = m.partial ? cd.in_games["n_" + m.key] : null;
      html += `<div class="dcell">
        <span class="dk">${m.label}${pn === null ? "" : ` <span class="tiny muted">n=${(pn || 0).toLocaleString()}</span>`}</span>
        <span class="dv">${fmtMetric(m.key, v)}</span>
        <span class="dd ${dclsOf(m.key, dl)}">${dl === null || dl === undefined ? "" : fmtDelta(m.key, dl)}</span>
      </div>`;
    });
    html += `</div>`;
  });
  html += `</div>`;
  html += `<p class="tiny muted" style="margin:8px 0 0">
    Right-hand numbers are the change against ${cd.name}'s own average over
    ${plural(cd.baseline.n, "game")} in this patch, so a column of positives means
    the draft itself is doing something.</p>`;
  return html;
}

async function toggleChamp(riot) {
  if (S.openChamp === riot) { S.openChamp = null; renderDraft(S.draft); return; }
  S.openChamp = riot;
  renderDraft(S.draft);
  if (S.detail[riot]) { renderDraft(S.draft); return; }
  const qs = `champ=${riot}&blue=${S.draft.blue.join(",")}&red=${S.draft.red.join(",")}` +
    `&patch=${encodeURIComponent(S.patch)}`;
  const cd = await (await fetch("/api/champion?" + qs)).json();
  S.detail[riot] = cd;
  if (S.openChamp === riot) renderDraft(S.draft);
}

/* ---- full pairwise delta */
function matrixCard(d) {
  const key = S.metric;
  const card = el("div", "card");
  const m = S.byKey[key];
  card.appendChild(el("h3", null,
    `Pairwise ${m.label} — row champion against column champion`));
  card.appendChild(el("div", "hint tiny muted",
    `Each <b>row</b> is the champion being measured; each <b>column</b> is who it
     played <b>with</b> (same team) or <b>against</b> (other team). The two diagonal
     blocks — blue×blue and red×red — are teammate pairs; the off-diagonal blocks are
     opponents. Green means the row champion does better than its own average in this
     patch, red worse; the number is that difference, not a win rate.`));

  const order = [...d.blue, ...d.red];
  /* where the red block starts; -1 when only one side has champions, so no
     divider is drawn for a "block" that is the whole table */
  const nb = d.blue.length && d.red.length ? d.blue.length : -1;
  const sideOf = {};
  d.blue.forEach((r) => { sideOf[r] = "blue"; });
  d.red.forEach((r) => { sideOf[r] = "red"; });
  const lookup = {};
  d.pairs.forEach((p) => { lookup[`${p.champ}:${p.partner}:${p.kind}`] = p; });

  let maxAbs = 0;
  order.forEach((row) => order.forEach((col) => {
    if (row === col) return;
    const kind = sideOf[row] === sideOf[col] ? "with" : "vs";
    const p = lookup[`${row}:${col}:${kind}`];
    if (p && p.deltas[key] !== undefined) {
      maxAbs = Math.max(maxAbs, Math.abs(p.deltas[key]) * (m.fmt === "pct" ? 100 : 1));
    }
  }));

  let html = `<div class="matrix-wrap"><table class="matrix"><thead><tr>
    <th class="corner"><span class="ax">row ↓</span><span class="ax">col →</span></th>`;
  order.forEach((r, j) => {
    html += `<th class="${j === nb ? "split" : ""}" title="${S.byRiot[r].name}">${img(S.byRiot[r])}</th>`;
  });
  html += "</tr></thead><tbody>";
  order.forEach((row, i) => {
    html += `<tr><th class="rowhead ${i === nb ? "split" : ""}"
      title="${S.byRiot[row].name}">${img(S.byRiot[row])}</th>`;
    order.forEach((col, j) => {
      if (row === col) { html += `<td class="self ${j === nb ? "split" : ""}">·</td>`; return; }
      const kind = sideOf[row] === sideOf[col] ? "with" : "vs";
      const p = lookup[`${row}:${col}:${kind}`];
      if (!p || p.deltas[key] === undefined) {
        html += `<td class="${j === nb ? "split" : ""}">—</td>`;
        return;
      }
      const dv = p.deltas[key];
      const shown = m.fmt === "pct" ? dv * 100 : dv;
      const a = Math.min(0.62, 0.10 + 0.52 * (maxAbs ? Math.abs(shown) / maxAbs : 0));
      const good = dv * signOf(key) > 0;
      const bg = good ? `rgba(79,184,106,${a})` : `rgba(217,92,72,${a})`;
      let txt = m.fmt === "pct" ? shown.toFixed(1)
        : Math.abs(shown) >= 100 ? String(Math.round(shown)) : shown.toFixed(1);
      if (/^-0(\.0)?$/.test(txt)) txt = txt.slice(1);          /* no "-0.0" */
      html += `<td class="${j === nb ? "split" : ""}" style="background:${bg}"
        title="${S.byRiot[row].name} ${kind === "with" ? "with" : "against"} ${S.byRiot[col].name}: ${fmtMetric(key, p.vals[key])} · ${fmtDelta(key, dv)} · n=${p.n}">
        ${shown > 0 ? "+" : ""}${txt}</td>`;
    });
    html += "</tr>";
  });
  html += `</tbody></table></div>
    <div class="legend">
      <span><i style="background:rgba(79,184,106,.6)"></i>better than the row champion's own average</span>
      <span><i style="background:rgba(217,92,72,.6)"></i>worse</span>
      <span>${m.fmt === "pct" ? "percentage points" : "units"} · hover a cell for the raw value and n
        ${m.partial ? `· recorded for ${(m.cov * 100).toFixed(0)}% of player-rows, so the cells cover fewer games` : ""}</span>
    </div>`;
  card.innerHTML += html;
  return card;
}

function dropCard(d) {
  const card = el("div", "card");
  card.appendChild(el("h3", null, "Games you would recover by dropping one champion"));
  const chips = el("div", "chips");
  d.drop_one.slice().sort((a, b) => b.n_without - a.n_without).forEach((x) => {
    const c = S.byRiot[x.riot];
    const chip = el("div", "chip",
      `${img(c)}<span>${x.name}</span> <b class="${x.n_without > d.n ? "up" : "flat"}">` +
      `${x.n_without.toLocaleString()}</b>`);
    chip.title = `without ${x.name}: ${x.n_without.toLocaleString()} games`;
    chips.appendChild(chip);
  });
  card.appendChild(chips);
  card.appendChild(el("p", "tiny muted",
    "Games matching the rest of the draft if that champion is removed."));
  return card;
}

function estimateCard(d) {
  const card = el("div", "card");
  card.appendChild(el("h3", null, "First-order estimate from the pair data"));
  let html = `<table class="stats"><thead><tr>
    <th>champion</th><th>own WR</th><th>pairs</th>
    <th>avg pair effect</th><th>implied WR</th></tr></thead><tbody>`;
  d.estimate.forEach((e) => {
    const c = S.byRiot[e.riot];
    html += `<tr>
      <td class="champ">${img(c)}${e.name}</td>
      <td>${pct(e.base)}</td>
      <td class="flat" title="best ${dsgn(e.best, 1, 100)} / worst ${dsgn(e.worst, 1, 100)} pts">
        ${e.terms}</td>
      <td class="${dclsOf("win", e.mean_delta)}">${dsgn(e.mean_delta, 1, 100)} pts</td>
      <td>${pct(e.estimate)}</td>
    </tr>`;
  });
  html += `</tbody></table>
    <p class="tiny muted" style="margin:10px 0 0">
    Mean of the measured pair deviations against the other champions, added to the
    champion's own win rate. Averaged rather than summed: the pairs share games, so
    summing would count the same wins several times. A direction, not a forecast.</p>`;
  card.innerHTML += html;
  return card;
}

function gamesCard(d) {
  const card = el("div", "card");
  card.appendChild(el("h3", null, `Matching games${d.n > d.games.length
    ? ` (first ${d.games.length} of ${d.n.toLocaleString()})` : ""}`));
  const wrap = el("div", "games");
  d.games.forEach((g) => {
    const row = el("div", "game");
    const known = g.blue_win !== null && g.blue_win !== undefined;
    const blueWon = g.blue_win === 1;
    const mineWon = !known || g.mine === null ? null
      : (g.mine === "blue" ? blueWon : !blueWon);
    row.innerHTML = `
      <div class="side b ${g.mine === "blue" ? "mine" : ""}">${g.blue.map((n) => img(S.champs.find((c) => c.name === n))).join("")}</div>
      <div class="mid">${known ? `<span class="${blueWon ? "win-b" : "win-r"}">${blueWon ? "blue" : "red"} win</span>` : `<span class="flat">no result recorded</span>`}
        ${mineWon === null ? "" : `<span class="${mineWon ? "up" : "down"}">· yours ${mineWon ? "won" : "lost"}</span>`}
        · ${Math.floor(g.duration / 60)}:${String(g.duration % 60).padStart(2, "0")}</div>
      <div class="side r ${g.mine === "red" ? "mine" : ""}">${g.red.map((n) => img(S.champs.find((c) => c.name === n))).join("")}</div>`;
    wrap.appendChild(row);
  });
  card.appendChild(wrap);
  return card;
}

/* ---------------------------------------------------- what to add next */
/* One row, two columns: the best additions for each side. A side with nothing
   picked is measured against the other side instead of alongside it, so a
   column is never empty while a draft exists. The + on the left adds the
   champion; the rest of the row opens the pair cells the headline is averaged
   from, because "54% over 300 games" is only as good as the cells behind it. */
function suggestHost(d) {
  const card = el("div", "card");
  const key = `${d.blue.join(",")}/${d.red.join(",")}/${S.metric}/${S.patch}`;
  if (S.sugCache && S.sugCache.key === key) {
    paintSuggest(card, d, S.sugCache.val);      /* toggles must not re-fetch */
    return card;
  }
  card.innerHTML = `<h3>What to add next</h3>
    <div class="tiny muted">loading suggestions…</div>`;
  fillSuggest(card, d, key);
  return card;
}

async function fillSuggest(card, d, key) {
  const req = ++S.sugReq;
  const qs = `blue=${d.blue.join(",")}&red=${d.red.join(",")}` +
    `&metric=${S.metric}&patch=${encodeURIComponent(S.patch)}`;
  let sug;
  try {
    sug = await (await fetch("/api/suggest?" + qs)).json();
  } catch (e) {
    card.innerHTML = `<h3>What to add next</h3>
      <div class="muted">Suggestions could not be loaded.</div>`;
    return;
  }
  if (req !== S.sugReq || S.draft !== d) return;   /* a newer draft already won */
  S.sugCache = { key, val: sug };
  paintSuggest(card, d, sug);
}

function paintSuggest(card, d, sug) {
  card.innerHTML = suggestHtml(d, sug);
  /* the + adds; it must not also open the breakdown underneath */
  card.querySelectorAll("button.addbtn").forEach((b) => {
    b.onclick = (e) => {
      e.stopPropagation();
      addChamp(b.dataset.side, S.byRiot[Number(b.dataset.riot)]);
    };
  });
  card.querySelectorAll("tr.sugrow").forEach((tr) => {
    tr.onclick = () => {
      const k = tr.dataset.side + ":" + tr.dataset.riot;
      S.openSug = S.openSug === k ? null : k;
      paintSuggest(card, d, sug);
    };
  });
}

/* clicking + fills the next empty slot on that side */
function addChamp(side, c) {
  if (!c) return;
  if ([...S.blue, ...S.red].some((x) => x && x.riot === c.riot)) return;
  const slot = S[side].findIndex((x) => !x);
  if (slot < 0) return;                  /* team already has five */
  S[side][slot] = c;
  renderSlots();
  loadDraft();
}

function suggestHtml(d, sug) {
  const m = S.byKey[S.metric];
  const col = (side) => {
    const s = sug.sides[side];
    const other = side === "blue" ? "red" : "blue";
    const st = d.side_stats && d.side_stats[side];
    const rate = s.basis === "with" && st && st.sel_win_rate !== null && st.n
      ? ` · ${pct(st.sel_win_rate)} in ${plural(st.n, "game")}` : "";
    /* say which set of champions the number was measured against, because
       "alongside blue" and "against red" are different claims */
    const basis = s.basis === "vs"
      ? `measured against ${other}'s ${plural(s.subject.length, "pick")}`
      : `measured alongside ${side}'s ${plural(s.subject.length, "pick")}`;
    return `<div class="sugcol ${side}">
      <div class="sughead">
        <span class="dot ${side}"></span><b>Best for ${side}</b>
        <span class="tiny muted">${basis}${rate}</span>
      </div>
      ${sugTable(s, side)}
    </div>`;
  };
  return `<h3>What to add next</h3>
    <div class="hint tiny muted">One list per side, ranked by <b>${m.label}</b>.
      <b>+</b> puts the champion on that team; clicking the row breaks the number
      into the pair cells it is averaged from.</div>
    <div class="sugcols">${col("blue")}${col("red")}</div>
    <details class="methods">
      <summary class="tiny">how these five are chosen</summary>
      <div class="tiny muted">
        <p>Each column answers one question: which champion would most help that
        side. A side that already has picks is measured <b>alongside</b> them — its
        games with at least one of those champions on its own team. A side with
        nothing picked is measured <b>against</b> the other side instead, since
        there is nothing for it to stand next to yet.</p>
        <p>Only 2-way cells are used. Every champion not already in the draft is
        scored from the precomputed pair cube, the best 14 are re-measured on the
        games table, and the top five are shown — so <b>games</b> is a count of
        distinct games, not a sum of cells. <b>vs own avg</b> is the gap to that
        champion's average over the whole patch: a large gap on a small count is
        the interesting case and also the one most likely to be noise.</p>
        <p>Clicking a row splits the headline into the individual pair cells
        behind it, one per champion on the other side of the comparison. Those
        cells overlap — a game where two of the picks were on the team counts in
        both — so together they cover more games than the headline. The headline
        is the deduplicated union, and its mean is taken over games, not over
        cells.</p>
        <p>${m.partial
          ? `This stat is recorded for ${(m.cov * 100).toFixed(0)}% of player-rows.
             Averages are taken over the rows that carry it, while the game counts
             are all matching games, so a count can be larger than the sample
             behind the average.`
          : "Every matching game carries this stat."}</p>
      </div>
    </details>`;
}

function sugTable(s, addSide) {
  const m = S.byKey[S.metric];
  const items = s.best || [];
  if (!items.length) {
    return `<p class="muted tiny">No champion has a measured game here yet.</p>`;
  }
  const full = S[addSide].every(Boolean);
  const maxAbs = Math.max(...items.map((x) => Math.abs(x.delta)), 1e-9);
  let html = `<table class="stats sug"><thead><tr>
    <th class="addcol"></th><th>champion</th><th>${m.label}</th>
    <th>games</th><th>vs own avg</th>
  </tr></thead><tbody>`;
  items.forEach((x) => {
    const c = S.byRiot[x.riot];
    const dcls = dclsOf(S.metric, x.delta);
    const w = Math.min(100, Math.abs(x.delta) / maxAbs * 100);
    const open = S.openSug === addSide + ":" + x.riot;
    const addTitle = full ? `${addSide} is full`
      : `add ${x.name} to ${addSide}`;
    html += `<tr class="sugrow ${open ? "open" : ""}" data-riot="${x.riot}"
        data-side="${addSide}" title="click for the pair cells behind this number">
      <td class="addcol"><button class="addbtn" data-riot="${x.riot}"
        data-side="${addSide}" ${full ? "disabled" : ""}
        title="${addTitle}">+</button></td>
      <td class="champ">${img(c)}<span>${x.name}</span></td>
      <td>${fmtMetric(S.metric, x.specific)}</td>
      <td class="flat">${x.n.toLocaleString()}</td>
      <td class="${dcls}">${fmtDelta(S.metric, x.delta)}
        <div class="minibar"><i class="${dcls}" style="width:${w}%"></i></div></td>
    </tr>`;
    if (open) html += `<tr class="brkrow"><td colspan="5">${breakdown(x, s)}</td></tr>`;
  });
  return html + `</tbody></table>`;
}

/* The headline, then every pair cell it is made of. Read-only: the parts are
   context for the number, not another set of things to add. */
function breakdown(x, s) {
  const m = S.byKey[S.metric];
  const parts = x.parts || [];
  const cover = parts.reduce((a, p) => a + p.n, 0);
  const maxAbs = Math.max(...parts.map((p) => Math.abs(p.delta)), 1e-9);
  const cond = s.basis === "vs"
    ? `facing at least one of ${plural(s.subject.length, "pick")}`
    : `sharing a team with at least one of ${plural(s.subject.length, "pick")}`;
  let rows = parts.map((p) => {
    const dcls = dclsOf(S.metric, p.delta);
    const w = Math.min(50, Math.abs(p.delta) / maxAbs * 50);
    /* a bar that grows out from the champion's own average, so the cells that
       pull the headline up and the ones that pull it down are both visible */
    const bar = `<div class="dbar"><b></b><i class="${dcls}" style="${
      p.delta >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`}"></i></div>`;
    return `<tr>
      <td class="champ">${img(S.byRiot[p.riot])}<span>${p.name}</span></td>
      <td>${fmtMetric(S.metric, p.value)}</td>
      <td class="flat">${p.n.toLocaleString()}</td>
      <td class="${dcls}">${fmtDelta(S.metric, p.delta)}${bar}</td>
    </tr>`;
  }).join("");
  return `<div class="brkbox">
    <div class="brkhead">
      <span><b>${x.name}</b> ${fmtMetric(S.metric, x.specific)}
        over ${plural(x.n, "game")} ${cond}</span>
      <span class="tiny muted">own average ${fmtMetric(S.metric, x.general)}
        over ${plural(x.n_general, "game")}</span>
    </div>
    ${parts.length ? `<table class="stats brk">
      <thead><tr><th>${s.basis === "vs" ? "opponent" : "teammate"}</th>
        <th>${m.label}</th><th>games</th><th>vs own avg</th></tr></thead>
      <tbody>${rows}</tbody></table>` : `<p class="muted tiny">No per-champion cell.</p>`}
    <p class="tiny muted">Each row is one pair cell: ${x.name} in the games where
      that champion was ${s.basis === "vs" ? "on the other side" : "on the same team"}.
      The cells overlap — a game with two of the picks sits in two of them — so
      together they cover ${plural(cover, "game")} against the
      ${plural(x.n, "game")} above. The headline is the deduplicated union.</p>
  </div>`;
}


/* ------------------------------------------------------------------ matchup */
function renderMatchupPicks() {
  [["m-x", S.m.x], ["m-y", S.m.y]].forEach(([id, c]) => {
    const b = $(id);
    b.className = "pick" + (c ? "" : " empty");
    b.innerHTML = c ? `${img(c)}<span>${c.name}</span>` : "choose champion";
  });
}

async function loadMatchup() {
  if (!S.m.x || !S.m.y) return;
  const t0 = performance.now();
  const r = await fetch(`/api/matchup?champ=${S.m.x.riot}&partner=${S.m.y.riot}` +
    `&mode=${S.m.mode}&patch=${encodeURIComponent(S.patch)}` +
    (S.m.role ? `&role=${S.m.role}` : ""));
  const d = await r.json();
  $("latency").textContent = `${Math.round(performance.now() - t0)} ms · precomputed`;
  const host = $("matchup-results");
  if (d.error) { host.innerHTML = `<div class="card">${d.error}</div>`; return; }

  const verb = S.m.mode === "vs" ? "against" : "with";
  const title = `${S.m.x.name} ${verb} ${S.m.y.name}`;
  let html = `<div class="card">
    <div class="hero">
      <div class="num">${d.n.toLocaleString()}<small>games</small></div>
      ${badge(d.n)}
      <div class="sub">${title}</div>
    </div>`;

  if (d.stats) {
    const rows = [
      ["win rate", d.stats.win, d.delta && d.delta.win, (v) => pct(v), true, 100, 1],
      ["kills", d.stats.kills, d.delta && d.delta.kills, (v) => num(v, 2), true, 1, 2],
      ["deaths", d.stats.deaths, d.delta && d.delta.deaths, (v) => num(v, 2), false, 1, 2],
      ["assists", d.stats.assists, d.delta && d.delta.assists, (v) => num(v, 2), true, 1, 2],
      ["gold", d.stats.gold, d.delta && d.delta.gold, (v) => kfmt(v), true, 1, 0],
      ["damage", d.stats.dmg, d.delta && d.delta.dmg, (v) => kfmt(v), true, 1, 0],
      ["CS", d.stats.cs, d.delta && d.delta.cs, (v) => num(v, 1), true, 1, 1],
      ["vision", d.stats.vision, d.delta && d.delta.vision, (v) => num(v, 1), true, 1, 1],
    ];
    html += `<div class="statgrid" style="margin-top:14px">`;
    rows.forEach(([k, v, dl, fmt, better, scale, digits]) => {
      const shown = dl === null || dl === undefined ? dl : (better ? dl : -dl);
      html += `<div class="stat">
        <div class="k">${k}</div>
        <div class="v">${fmt(v)}</div>
        <div class="d ${shown === null || shown === undefined ? "flat" : (Math.abs(shown) < 1e-9 ? "flat" : (shown > 0 ? "up" : "down"))}">
          ${dl === null || dl === undefined ? "&nbsp;" : dsgn(dl, digits, scale) + " vs base"}</div>
      </div>`;
    });
    html += `</div>`;
    if (d.baseline) {
      html += `<p class="tiny muted" style="margin:12px 0 0">
        ${S.m.x.name}'s own average across ${plural(d.baseline.n, "game")}:
        ${pct(d.baseline.win)} win rate, ${num(d.baseline.kills, 2)} /
        ${num(d.baseline.deaths, 2)} / ${num(d.baseline.assists, 2)} K / D / A.</p>`;
    }
  } else {
    html += `<p class="muted" style="margin:12px 0 0">
      This pair has never been observed in the selected patch.</p>`;
  }
  if (d.role) {
    html += `<p class="tiny muted" style="margin:8px 0 0">
      ${S.m.x.name} as ${d.role.pos.toLowerCase()}: ${pct(d.role.win)} win rate
      over ${d.role.n.toLocaleString()} games.</p>`;
  }
  html += `</div>`;
  host.innerHTML = html;
}

init();
