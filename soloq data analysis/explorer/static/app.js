const S = {
  meta: null, champs: [], byRiot: {},
  patch: "", sides: "any", tab: "draft",
  blue: [null, null, null, null, null],
  red: [null, null, null, null, null],
  m: { x: null, y: null, mode: "with", role: "" },
  pickTarget: null,
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
const dcls = (d) => d === null || d === undefined || Math.abs(d) < 1e-9
  ? "flat" : (d > 0 ? "up" : "down");
const dsgn = (v, digits = 1, scale = 1) => v === null || v === undefined
  ? "" : (v > 0 ? "+" : "") + (v * scale).toFixed(digits);

/* ------------------------------------------------------------------ init */
async function init() {
  const r = await fetch("/api/meta");
  S.meta = await r.json();
  S.champs = S.meta.champions;
  S.champs.forEach((c) => { S.byRiot[c.riot] = c; });
  S.patch = S.meta.patches.length ? S.meta.patches[0].patch : "";

  $("corpus").innerHTML =
    `<b>${S.meta.games.toLocaleString()}</b> games · ` +
    `<b>${S.meta.players.toLocaleString()}</b> player-rows · ` +
    `${S.champs.length} champions`;

  $("patch").innerHTML = S.meta.patches
    .map((p) => `<option value="${p.patch}">${p.patch} (${p.games.toLocaleString()})</option>`)
    .join("") + `<option value="">all patches</option>`;
  $("patch").value = S.patch;

  $("m-role").innerHTML = `<option value="">any</option>` +
    S.meta.roles.map((r) => `<option value="${r}">${r.toLowerCase()}</option>`).join("");

  renderSlots();
  renderMatchupPicks();
  wire();
  readHash();
}

/* A draft is shareable: #103,64,84/113,126
   A matchup is shareable: #m=103,64,vs */
function readHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ""));
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
  $("sides").onchange = () => { S.sides = $("sides").value; loadDraft(); };

  document.querySelectorAll("#tabs button").forEach((b) => {
    b.onclick = () => {
      showTab(b.dataset.tab);
      if (S.tab === "matchup") loadMatchup();
    };
  });

  $("btn-clear").onclick = () => {
    S.blue = [null, null, null, null, null];
    S.red = [null, null, null, null, null];
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
        s.innerHTML = `+ ${side === "blue" ? "blue" : "red"} ${i + 1}`;
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
    if (!blue.length && !red.length) { renderDraftEmpty(); writeHash(); return; }
    writeHash();
    const t0 = performance.now();
    const r = await fetch(`/api/draft?blue=${blue.join(",")}&red=${red.join(",")}` +
      `&patch=${encodeURIComponent(S.patch)}&sides=${S.sides}`);
    const d = await r.json();
    $("latency").textContent = `${Math.round(performance.now() - t0)} ms · live scan`;
    if (d.error) { $("draft-results").innerHTML = `<div class="card">${d.error}</div>`; return; }
    renderDraft(d);
  }, 40);
}

function renderDraftEmpty() {
  $("draft-results").innerHTML =
    `<div class="empty">Pick a champion to start. The count below is how many
     real games contain everything you have selected.</div>`;
  $("latency").textContent = "";
}

function renderDraft(d) {
  const host = $("draft-results");
  host.innerHTML = "";
  const t = S.meta.thresholds;
  const [cls] = conf(d.n);

  /* hero */
  const hero = el("div", "card");
  const verdict = d.n === 0
    ? "No game in the corpus contains this exact set."
    : d.n >= t.strong
      ? "Enough games to read as a real effect."
      : d.n >= t.ok
        ? "Directionally useful, but check the deltas are not noise."
        : d.n >= t.weak
          ? "Too few games to separate from luck. Use the pair data below."
          : "A handful of games. Treat any average here as anecdote.";
  hero.innerHTML = `
    <div class="hero">
      <div class="num">${d.n.toLocaleString()}<small>${d.n === 1 ? "game" : "games"}</small></div>
      ${badge(d.n)}
      <div class="sub">${verdict}</div>
    </div>`;
  host.appendChild(hero);

  /* ladder */
  if (d.ladder.length > 1) {
    const card = el("div", "card");
    card.appendChild(el("h3", null, "How the sample collapses as you add champions"));
    const max = Math.max(...d.ladder.map((x) => x.n), 1);
    const wrap = el("div", "ladder");
    d.ladder.forEach((step) => {
      const c = S.byRiot[step.riot];
      const w = Math.max(2, (Math.log10(step.n + 1) / Math.log10(max + 1)) * 100);
      const [k] = conf(step.n);
      const colour = { solid: "#4fb86a", ok: "#5b9bd5", weak: "#d9a441", noise: "#d95c48", none: "#4a5468" }[k];
      const row = el("div", "rung");
      row.innerHTML = `
        <div class="who">${img(c)}<span>${step.name}</span></div>
        <div class="bar"><i style="width:${w}%;background:${colour}"></i></div>
        <div class="n">${step.n.toLocaleString()}</div>`;
      wrap.appendChild(row);
    });
    card.appendChild(wrap);
    host.appendChild(card);
  }

  /* estimate, when the exact count is not usable */
  if (d.estimate && d.estimate.length && d.n < t.ok) {
    const card = el("div", "card");
    card.appendChild(el("h3", null, "What the pair data says (no exact sample)"));
    let html = `<table class="stats"><thead><tr>
      <th>champion</th><th>base WR</th><th>pairs</th>
      <th>avg pair effect</th><th>implied WR</th></tr></thead><tbody>`;
    d.estimate.forEach((e) => {
      const c = S.byRiot[e.riot];
      html += `<tr>
        <td class="champ">${img(c)}${e.name}</td>
        <td>${pct(e.base)}</td>
        <td class="flat" title="best ${dsgn(e.best, 1, 100)} / worst ${dsgn(e.worst, 1, 100)} pts">
          ${e.terms}</td>
        <td class="${dcls(e.mean_delta)}">${dsgn(e.mean_delta, 1, 100)} pts</td>
        <td>${pct(e.estimate)}</td>
      </tr>`;
    });
    html += `</tbody></table>
      <p class="tiny muted" style="margin:10px 0 0">
      Mean of the measured pair deviations against the other nine champions, added
      to the champion's own win rate. Averaged rather than summed: the pairs share
      games, so summing them would count the same wins several times. Read it as a
      direction, not a forecast — hover the pair count for the best and worst term.</p>`;
    card.innerHTML += html;
    host.appendChild(card);
  }

  /* per-champion stats in the matching games */
  if (d.champ_stats && d.champ_stats.length && d.n > 0) {
    const card = el("div", "card");
    card.appendChild(el("h3", null, d.n === 1
      ? "What the selected champions did in that one game"
      : `What the selected champions did in those ${plural(d.n, "game")}`));
    let html = `<table class="stats"><thead><tr>
      <th>champion</th><th>n</th><th>win</th><th>K</th><th>D</th><th>A</th>
      <th>gold</th><th>dmg</th><th>CS</th><th>vision</th></tr></thead><tbody>`;
    d.champ_stats
      .slice().sort((a, b) => b.n - a.n)
      .forEach((r) => {
        const c = S.byRiot[r.riot];
        const base = S.byRiot[r.riot];
        html += `<tr>
          <td class="champ">${img(c)}${r.name}</td>
          <td>${r.n.toLocaleString()}</td>
          <td>${pct(r.win)}</td>
          <td>${num(r.kills, 1)}</td><td>${num(r.deaths, 1)}</td><td>${num(r.assists, 1)}</td>
          <td>${kfmt(r.gold)}</td><td>${kfmt(r.dmg)}</td>
          <td>${num(r.cs, 0)}</td><td>${num(r.vision, 1)}</td>
        </tr>`;
      });
    html += "</tbody></table>";
    card.innerHTML += html;
    host.appendChild(card);
  }

  /* drop-one: who is starving the sample */
  if (d.drop_one && d.drop_one.length > 1 && d.n < t.ok) {
    const card = el("div", "card");
    card.appendChild(el("h3", null, "Games you would recover by dropping one champion"));
    const chips = el("div", "chips");
    d.drop_one
      .slice().sort((a, b) => b.n_without - a.n_without)
      .forEach((x) => {
        const c = S.byRiot[x.riot];
        const gain = x.n_without - d.n;
        const chip = el("div", "chip",
          `${img(c)}<span>${x.name}</span> <b class="${gain > 0 ? "up" : "flat"}">` +
          `${x.n_without.toLocaleString()}</b>`);
        chip.title = `without ${x.name}: ${x.n_without.toLocaleString()} games`;
        chips.appendChild(chip);
      });
    card.appendChild(chips);
    card.appendChild(el("p", "tiny muted",
      "Number of games matching the rest of the draft if that champion is removed."));
    host.appendChild(card);
  }

  /* pairwise matrix */
  if (d.pairs && d.pairs.length && d.selected > 1) {
    const card = el("div", "card");
    card.appendChild(el("h3", null, "Pairwise win-rate delta inside this draft"));
    const order = [...d.blue, ...d.red];
    const sideOf = {};
    d.blue.forEach((r) => { sideOf[r] = "blue"; });
    d.red.forEach((r) => { sideOf[r] = "red"; });
    const lookup = {};
    d.pairs.forEach((p) => { lookup[`${p.champ}:${p.partner}:${p.kind}`] = p; });

    let html = `<div class="matrix-wrap"><table class="matrix"><thead><tr><th></th>`;
    order.forEach((r) => {
      html += `<th title="${S.byRiot[r].name}">${img(S.byRiot[r])}</th>`;
    });
    html += "</tr></thead><tbody>";
    order.forEach((row) => {
      html += `<tr><th title="${S.byRiot[row].name}">${img(S.byRiot[row])}</th>`;
      order.forEach((col) => {
        if (row === col) { html += `<td class="self">·</td>`; return; }
        const kind = sideOf[row] === sideOf[col] ? "with" : "vs";
        const p = lookup[`${row}:${col}:${kind}`];
        if (!p) { html += `<td>—</td>`; return; }
        const dpts = p.delta * 100;
        const a = Math.min(0.6, 0.12 + Math.abs(dpts) / 8);
        const bg = dpts >= 0 ? `rgba(79,184,106,${a})` : `rgba(217,92,72,${a})`;
        html += `<td style="background:${bg};color:#e8ecf3"
          title="${S.byRiot[row].name} ${kind} ${S.byRiot[col].name}: ` +
          `${pct(p.win)} vs base, n=${p.n}">${dpts > 0 ? "+" : ""}${dpts.toFixed(1)}</td>`;
      });
      html += "</tr>";
    });
    html += `</tbody></table></div>
      <div class="legend">
        <span><i style="background:rgba(79,184,106,.6)"></i>higher win rate than the champion's own average</span>
        <span><i style="background:rgba(217,92,72,.6)"></i>lower</span>
        <span>numbers are percentage points; hover a cell for n</span>
      </div>`;
    card.innerHTML += html;
    host.appendChild(card);
  }

  /* matching games */
  if (d.games && d.games.length) {
    const card = el("div", "card");
    card.appendChild(el("h3", null, `Matching games${d.n > d.games.length ? ` (first ${d.games.length} of ${d.n.toLocaleString()})` : ""}`));
    const wrap = el("div", "games");
    d.games.forEach((g) => {
      const row = el("div", "game");
      const blueWon = g.blue_win === 1;
      row.innerHTML = `
        <div class="side b">${g.blue.map((n) => img(S.champs.find((c) => c.name === n))).join("")}</div>
        <div class="mid"><span class="${blueWon ? "win-b" : "win-r"}">${blueWon ? "blue" : "red"} win</span>
          · ${Math.floor(g.duration / 60)}:${String(g.duration % 60).padStart(2, "0")}</div>
        <div class="side r">${g.red.map((n) => img(S.champs.find((c) => c.name === n))).join("")}</div>`;
      wrap.appendChild(row);
    });
    card.appendChild(wrap);
    host.appendChild(card);
  }
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
    // [label, value, delta, formatter, higher-is-better]
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
        <div class="d ${dcls(shown)}">${dl === null || dl === undefined ? "&nbsp;"
          : dsgn(dl, digits, scale) + " vs base"}</div>
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
