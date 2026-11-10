const ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];
let CHAMPIONS = [], BY_IDX = {}, BY_RIOT = {};
let picks = Array(10).fill(null);          // slot -> riot id | null (pick order)
let bans = { blue: [], red: [] };          // team -> [riot id]
let lastSlot = null;                       // slot just filled: wins a role
let resp = null;                           // last /api/suggest (role model)
let respSig = null;                        // board `resp` was computed for
let presp = null;                          // last /api/next-move (policy)
// The lane the coming pick is for, when the user overrides the model's guess.
// Only the reranker uses it: the head answers a lane-free "best next champion".
let NEXT_LANE = null;
let prespSig = null;                       // board `presp` was computed for
let pickerCtx = null;                      // {type, slot? , team?}
let debounceT = null, wtFilterT = null;
let ROLE_MODEL = null, POLICY_MODEL = null;   // server-side model slots
let WT = null;                             // walkthrough: {draft, step, ...}
let SERIES = null;                         // {games:[slot], cur, fearless}
                                           // slot = {firstPick, picks, bans, done}
let DRAFT_PLANS = null;                    // {0: [20 moves], 1: [...]} from /api/champions
let PATCHES = [];                          // [{id, key}] the checkpoint's patch table

async function init() {
  const d = await (await fetch('/api/champions')).json();
  CHAMPIONS = d.champions;
  CHAMPIONS.forEach(c => { BY_IDX[c.idx] = c; BY_RIOT[c.riot] = c; });
  DRAFT_PLANS = d.move_plan || null;
  applyPatchPayload(d);
  setModels(d.model, d.policy_model);
  refreshCheckpoints();
  refreshDraftList();
  buildStrips('blue');
  buildStrips('red');
  initSeries();          // restores the last series, renders and asks
  renderWalkthrough();
}

function imgTag(c, cls) {
  if (!c) return '';
  return `<img class="${cls || 'ic'}" src="${c.img}" alt="" title="${c.name}"`
    + ' onerror="this.classList.add(\'broken\')">';
}

function setModels(masked, policy) {
  ROLE_MODEL = masked || null;
  POLICY_MODEL = policy || null;
  renderModelInfo();
}

function modelText(m, what) {
  let txt = what + ' ' + m.ckpt;
  if (m.kind === 'finetune')
    txt += ` · finetuned${m.stage != null ? ' (stage ' + m.stage + ')' : ''}`;
  else if (m.kind && m.kind !== 'pretrain' && m.kind !== 'policy')
    txt += ' · ' + m.kind;
  // An ensemble has no val run of its own; the server publishes the number it
  // was measured at, on rows neither it nor the model it replaced had seen.
  const ev = m.ensemble && m.ensemble.eval;
  if (ev && ev.top5 != null) {
    txt += ` · ${ev.label || 'held-out top-5'} ${(100 * ev.top5).toFixed(1)}%`
         + ` (${ev.n_drafts} drafts · ${m.ensemble.members.length} models)`;
  } else if (m.rerank_eval && m.rerank_eval.kway != null) {
    // A reranker's number is a k-way accuracy over the head's shortlist, not a
    // top-5 over 173 champions, so it is labelled as what it is.
    txt += ` · k-way ${(100 * m.rerank_eval.kway).toFixed(1)}%`
         + ` of the head's top-${m.rerank_eval.k}`
         + (m.rerank_eval.n ? ` (${m.rerank_eval.n} states)` : '');
  } else {
    txt += m.val_top5 == null ? ' · val top-5 n/a'
      : ` · val top-5 ${(100 * m.val_top5).toFixed(1)}%`;
  }
  if (m.epoch != null) txt += ` · ep ${m.epoch}`;
  return txt;
}

function renderModelInfo() {
  // Two independent slots: the masked model answers the role board and the
  // per-role targets, the policy answers the next move. Either can be absent.
  const bits = [];
  bits.push(ROLE_MODEL ? modelText(ROLE_MODEL, 'role')
    : 'role model — none');
  if (POLICY_MODEL) bits.push(modelText(POLICY_MODEL, 'next-move'));
  document.getElementById('model-info').textContent = bits.join('   ·   ');
  document.body.classList.toggle('no-role-model', !ROLE_MODEL);
  document.body.classList.toggle('policy-mode', !!POLICY_MODEL);
  if (!POLICY_MODEL) document.getElementById('nextmove').classList.add('hidden');
}

async function refreshCheckpoints() {
  let d;
  try { d = await (await fetch('/api/checkpoints')).json(); }
  catch (e) { return; }
  // One list, two dropdowns: the server tags each checkpoint with the slots
  // it can fill (masked / policy) so a policy file never shows up under
  // "role" and vice versa.
  const fill = (id, role, noneLabel) => {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    if (noneLabel) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = noneLabel;
      sel.appendChild(o);
    }
    (d.checkpoints || []).forEach(c => {
      if (c.roles && !c.roles.includes(role)) return;
      const o = document.createElement('option');
      o.value = c.path;
      // An llm-* checkpoint needs CUDA and the LLM image. Offer it, but say so
      // instead of letting the click fail - the server already knows why.
      const off = c.available === false;
      o.textContent = off ? c.label + ' — needs the LLM image'
                          : c.label;
      o.disabled = off;
      if (c.reason) o.title = c.reason;
      sel.appendChild(o);
    });
    const cur = d.current && d.current[role];
    if (cur && ![...sel.options].some(o => o.value === cur)) {
      const o = document.createElement('option');   // current ckpt outside globs
      o.value = cur;
      o.textContent = cur;
      sel.appendChild(o);
    }
    if (cur) sel.value = cur;
    else if (noneLabel) sel.value = '';
  };
  fill('ckpt-select', 'masked', null);
  fill('policy-select', 'policy', '— none (role board only) —');
  // One line under the picker rather than a failure per click: the reranker is
  // the better model but it is a 4.2B one, and a CPU-only host cannot run it.
  const note = document.getElementById('policy-note');
  if (note) {
    const r = d.rerank || {};
    note.textContent = r.available === false ? 'LLM rerank unavailable: ' + (r.reason || '') : '';
    note.classList.toggle('hidden', r.available !== false);
  }
}

async function switchModel(url, opts, role) {
  const ids = ['btn-load-ckpt', 'btn-load-policy', 'btn-upload', 'btn-reset'];
  ids.forEach(id => { document.getElementById(id).disabled = true; });
  document.getElementById('model-info').textContent = 'loading model…';
  let ok = false, msg = '';
  try {
    const r = await fetch(url, opts);
    const d = await r.json().catch(() => ({}));
    if (r.ok && 'model' in d) {
      ok = true;
      if (d.model) {                       // null = unloaded
        if (d.role === 'policy') POLICY_MODEL = d.model;
        else ROLE_MODEL = d.model;
      } else if (role === 'policy') {
        POLICY_MODEL = null;
      } else {
        ROLE_MODEL = null;
      }
      renderModelInfo();
    } else msg = d.error || ('HTTP ' + r.status);
  } catch (e) { msg = String(e); }
  ids.forEach(id => { document.getElementById(id).disabled = false; });
  if (!ok) {
    alert('model load failed: ' + msg);
    renderModelInfo();
    return;
  }
  resp = null; presp = null; respSig = null;
  await refreshCheckpoints();
  await refreshPatches();   // the new checkpoint has its own patch table
  changed();                // re-run suggestions with the new weights
}

// ---- patch conditioning -------------------------------------------------
// The vocabulary belongs to the checkpoint: the patch embedding rows are
// indexed by the vocab the model was trained with, so offering a patch it
// never saw would silently select a different vector. The server publishes
// the loaded policy model's own table; a checkpoint without one gets a single
// disabled option instead of a control that does nothing.
function applyPatchPayload(d) {
  PATCHES = (d && d.patches) || [];
  const sel = document.getElementById('patch-select');
  if (!sel) return;
  const prevKey = sel.options.length
    ? sel.options[sel.selectedIndex].dataset.key : null;
  sel.innerHTML = '';
  if (!PATCHES.length) {
    const o = document.createElement('option');
    o.value = '0';
    o.dataset.key = '';
    o.textContent = ((d && d.default_patch) || '16.20') + ' (no patch table)';
    sel.appendChild(o);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  PATCHES.forEach(p => {
    const o = document.createElement('option');
    o.value = String(p.id);
    o.dataset.key = p.key;
    o.textContent = p.key;
    sel.appendChild(o);
  });
  const unk = document.createElement('option');
  unk.value = '0';
  unk.dataset.key = '';
  unk.textContent = 'UNK (unseen patch)';
  sel.appendChild(unk);
  // Keep the choice across a checkpoint switch when the new vocabulary has
  // the same patch; otherwise fall back to that model's default.
  const keep = prevKey != null
    ? [...sel.options].find(o => o.dataset.key === prevKey) : null;
  const dflt = [...sel.options].find(o => +o.value === +(d.default_patch_id || 0));
  sel.value = (keep || dflt || sel.options[0]).value;
}

async function refreshPatches() {
  let d;
  try { d = await (await fetch('/api/champions')).json(); }
  catch (e) { return; }
  applyPatchPayload(d);
}

function currentPatchId() {
  const sel = document.getElementById('patch-select');
  return sel && sel.value ? +sel.value : 0;
}

function patchKeyOf(s) {
  // `pipeline.patch_key` ('16.3' -> '16.03') in JS: the corpus stores both
  // spellings, the checkpoint vocabulary only the padded one.
  const m = /^\s*(\d+)\.(\d+)\s*$/.exec(String(s == null ? '' : s));
  return m ? m[1] + '.' + m[2].padStart(2, '0') : '';
}

function setPatchByKey(key) {
  // A real game carries its own patch, and that is the right conditioning
  // for it. Returns false when the checkpoint never saw that patch, so the
  // caller leaves the manual choice alone.
  const sel = document.getElementById('patch-select');
  const want = patchKeyOf(key);
  if (!want || !sel || sel.disabled) return false;
  const hit = [...sel.options].find(o => o.dataset.key === want);
  if (!hit) return false;
  sel.value = hit.value;
  return true;
}

function currentFirstPick() {
  // A loaded game dictates the order; a hand-built draft uses the toggle.
  if (WT) return WT.draft.first_pick ? 1 : 0;
  return +document.getElementById('first-pick').value ? 1 : 0;
}

function isPolicyModel() {
  return !!POLICY_MODEL;
}

function nextTurn(firstPick) {
  // Whose move is it is a property of the BOARD, not of a model answer.
  // /api/champions publishes the canonical 20-move order (built from
  // infer.move_plan, which is checked against the pro-data pipeline);
  // DraftOrder.nextTurn walks it from the front with the bans and picks the
  // board has consumed - exactly what the server's resolve_position does.
  // Keeping this synchronous is the point: /api/next-move is async, so its
  // `state` describes an older board. Trusting it here made the highlight
  // lag one move (blue banning twice at the start, a 4th ban slot showing
  // up in ban phase 1). Highlight only, never enforced - any click works.
  const nBans = bans.blue.length + bans.red.length;
  const nPicks = picks.filter(Boolean).length;
  return DraftOrder.nextTurn(DRAFT_PLANS, firstPick, nBans, nPicks);
}

function buildStrips(team) {
  const pbox = document.getElementById('slots-' + team);
  for (let i = 0; i < 5; i++) {
    const slot = team === 'blue' ? i : 5 + i;
    const el = document.createElement('div');
    el.className = 'slot empty';
    el.id = 'slot-' + slot;
    el.innerHTML =
      '<button class="clear" title="remove">✕</button>' +
      '<div class="pic"></div><div class="nm muted">pick</div>' +
      '<div class="rb"></div>';
    el.addEventListener('click', ev => {
      if (ev.target.classList.contains('clear')) {
        picks[slot] = null;
        lastSlot = slot;
        changed();
        return;
      }
      if (picks[slot] == null) openPicker({ type: 'pick', slot });
    });
    pbox.appendChild(el);
  }
  const bbox = document.getElementById('bans-' + team);
  for (let i = 0; i < 5; i++) {
    const el = document.createElement('div');
    el.className = 'bslot empty';
    el.id = 'ban-' + team + '-' + i;
    el.innerHTML = '<button class="clear" title="remove">✕</button>' +
      '<div class="pic"></div>';
    el.addEventListener('click', ev => {
      if (ev.target.classList.contains('clear')) {
        bans[team].splice(i, 1);
        changed();
        return;
      }
      if (bans[team].length < 5) openPicker({ type: 'ban', team });
    });
    bbox.appendChild(el);
  }
}

function renderTurn() {
  // Always derive the turn from the board. The next-move panel's `presp`
  // arrives asynchronously and is deliberately NOT read here: while a step
  // is being rendered it still describes the previous board.
  const t = nextTurn(currentFirstPick());
  ['blue', 'red'].forEach(name => {
    const el = document.getElementById('turn-' + name);
    el.textContent = '';
    el.title = '';
  });
  document.querySelectorAll('.turn-target')
    .forEach(el => el.classList.remove('turn-target'));
  if (!t) return;
  const side = t.side === 0 ? 'blue' : 'red';
  const label = document.getElementById('turn-' + side);
  label.textContent = t.kind === 'ban' ? '▶ to ban' : '▶ to pick';
  label.title = t.warning || '';
  const strip = document.getElementById(
    (t.kind === 'ban' ? 'bans-' : 'slots-') + side);
  const empty = strip.querySelector('.empty');
  if (empty) empty.classList.add('turn-target');
}

function renderSlots() {
  for (let s = 0; s < 10; s++) {
    const el = document.getElementById('slot-' + s);
    const riot = picks[s];
    const c = riot != null ? BY_RIOT[riot] : null;
    el.classList.toggle('empty', !c);
    el.classList.toggle('filled', !!c);
    el.querySelector('.pic').innerHTML = c ? imgTag(c, 'ic big') : '';
    const nm = el.querySelector('.nm');
    nm.textContent = c ? c.name : 'pick';
    nm.classList.toggle('muted', !c);
    const rb = el.querySelector('.rb');
    rb.className = 'rb';
    // Only an answer computed for THIS board may label a pick; otherwise
    // the badge would repeat the previous board's role for a champion that
    // has just been placed. "…" until the reply lands (a few ms).
    const fresh = !!(resp && respSig === boardSig());
    const rp = fresh && resp.roles ? resp.roles[s] : null;
    const ar = (fresh && resp.assignment && resp.assignment[s] != null)
      ? resp.assignment[s]
      : (rp && rp.length ? rp[0].role : null);
    if (c && rp && rp.length && ar != null) {
      const conf = rp.find(x => x.role === ar) || rp[0];
      rb.textContent = ROLES[ar] + ' ' + Math.round(100 * conf.prob) + '%';
      const proRole = WT && WT.actualRole ? WT.actualRole[riot] : undefined;
      rb.title = rp.slice(0, 3).map(x =>
        `${ROLES[x.role]} ${(100 * x.prob).toFixed(0)}%`).join(' · ')
        + (proRole != null ? ` · pro played ${ROLES[proRole]}` : '');
      if (proRole != null && proRole !== ar)
        rb.classList.add('mismatch');           // model vs pro disagreement
      else {
        const second = rp.find(x => x.role !== ar);
        if (second && second.prob >= 0.15) rb.classList.add('flex-rb');
      }
    } else {
      rb.textContent = '';
      rb.title = '';
    }
  }
}

function renderBans() {
  ['blue', 'red'].forEach(team => {
    for (let i = 0; i < 5; i++) {
      const el = document.getElementById('ban-' + team + '-' + i);
      const riot = bans[team][i];
      const c = riot != null ? BY_RIOT[riot] : null;
      el.classList.toggle('empty', !c);
      el.classList.toggle('filled', !!c);
      el.querySelector('.pic').innerHTML = c ? imgTag(c, 'ic small') : '';
      el.title = c ? 'ban: ' + c.name : 'add ban';
    }
  });
}

function renderRoles() {
  // Role board: five slots per team in TOP..UTILITY order. A picked champ
  // sits in its assigned role; an empty role shows the masked model's
  // target for it (the role-conditioned MASK query), with the next two
  // candidates as small icons.
  //
  // `fresh` is the whole point: an answer only describes the board it was
  // computed for. Rendering the previous assignment against the current
  // picks put a fresh pick in an old role (or nowhere) and left the role
  // it belongs to showing its old target - "I picked Vi and it still says
  // Pantheon jungle". While a newer answer is in flight the board is drawn
  // dimmed, with no targets and no confidences, i.e. it never claims
  // anything it does not know.
  const fresh = !!(resp && respSig === boardSig());
  ['blue', 'red'].forEach(team => {
    const box = document.getElementById('roles-' + team);
    box.innerHTML = '';
    const base = team === 'blue' ? 0 : 5;
    const assign = (resp && resp.assignment) || [];
    const roleSlot = {};
    for (let s = base; s < base + 5; s++)
      if (assign[s] != null) roleSlot[assign[s]] = s;
    for (let r = 0; r < 5; r++) {
      const el = document.createElement('div');
      const slot = roleSlot[r];
      const c = (slot != null && picks[slot] != null)
        ? BY_RIOT[picks[slot]] : null;
      if (c) {
        const rp = (resp && resp.roles && resp.roles[slot]) || [];
        const conf = fresh ? rp.find(x => x.role === r) : null;
        el.className = 'role-slot filled' + (fresh ? '' : ' busy');
        el.title = `${ROLES[r]} — ${c.name}`
          + (conf ? ` · ${(100 * conf.prob).toFixed(1)}%` : ' · placing…');
        el.innerHTML = `<div class="rlabel">${ROLES[r]}</div>`
          + `<div class="pic">${imgTag(c, 'ic big')}</div>`
          + `<div class="nm">${c.name}</div>`
          + `<div class="sub">${conf
            ? (100 * conf.prob).toFixed(0) + '%' : '…'}</div>`;
      } else {
        const list = fresh && resp[team] && resp[team][ROLES[r]]
          ? resp[team][ROLES[r]] : [];
        if (list.length) {
          const top = list[0];
          const rest = list.slice(1, 3);
          el.className = 'role-slot empty target';
          el.title = `${ROLES[r]} — model target: ${top.name}`
            + ` (${(100 * top.prob).toFixed(1)}%)`;
          el.innerHTML = `<div class="rlabel">${ROLES[r]}</div>`
            + `<div class="pic">${imgTag(BY_IDX[top.idx], 'ic big')}</div>`
            + `<div class="nm">${top.name}</div>`
            + `<div class="sub">${(100 * top.prob).toFixed(1)}%`
            + (rest.length
              ? `<span class="mini">${
                rest.map(x => imgTag(BY_IDX[x.idx], 'ic mini')).join('')
              }</span>`
              : '') + '</div>';
          el.addEventListener('click', () => addPick(team, top.idx));
          el.querySelectorAll('.mini img').forEach((img, i) =>
            img.addEventListener('click', ev => {
              ev.stopPropagation();
              addPick(team, rest[i].idx);
            }));
        } else {
          el.className = 'role-slot empty' + (fresh ? '' : ' busy');
          el.innerHTML = `<div class="rlabel">${ROLES[r]}</div>`
            + '<div class="pic"></div>'
            + `<div class="nm muted">${fresh ? '—' : '…'}</div>`;
        }
      }
      box.appendChild(el);
    }
  });
}

function addPick(team, idx) {
  // Clicking a target card. Refuse anything already on the board: the card
  // may come from an answer that is a moment old, and a duplicate makes the
  // server reject the whole board (which used to freeze it). A fearless
  // champion is refused too - the answer may predate the pool.
  const c = BY_IDX[idx];
  if (!c || picks.includes(c.riot) || allBansFlat().includes(c.riot)) return;
  if (isFearless(c.riot)) return;
  const base = team === 'blue' ? 0 : 5;
  for (let s = base; s < base + 5; s++) {
    if (picks[s] == null) { picks[s] = c.riot; lastSlot = s; break; }
  }
  changed();
}

function renderAll() {
  renderSlots(); renderBans(); renderRoles();
  renderTurn(); renderNextMove();
  renderSeries(); renderFearless();
}

function changed() {
  renderAll();
  renderCompare();
  requestSuggest();
}

function allBansFlat() { return bans.blue.concat(bans.red); }

// ---- series + fearless -------------------------------------------------
// A series is a list of game slots. Each game carries its own fearless pool:
// the champions picked in the games *before* it. That is how the pro corpus
// defines it - `pipeline.attach_fearless` fills `row["unavailable"]` with the
// union of the earlier games' picks, and it matches on all 10,021 fearless
// rows in `pro_drafts.pt` with zero mismatches.
//
// The pool is stored per game rather than as one running list, for a reason
// that is easy to get wrong: a running list makes a *closed* game illegal
// against itself (its own picks are in the pool), so switching back to G1
// after starting G2 would show a board the server rejects. Per-game pools
// keep every game self-consistent, and still let the user add and remove
// champions by hand.
//
// Bans are never in the pool: a champion picked in an earlier game is still
// legal to ban.
const SERIES_KEY = 'draftSeries.v1';

function newGameSlot(firstPick, fearless) {
  return { firstPick: firstPick ? 1 : 0, picks: Array(10).fill(null),
           bans: { blue: [], red: [] }, fearless: (fearless || []).slice(),
           done: false };
}

function fearlessPool() {
  return SERIES.games[SERIES.cur].fearless;
}

function initSeries() {
  SERIES = { games: [newGameSlot(0)], cur: 0 };
  try {
    const s = JSON.parse(localStorage.getItem(SERIES_KEY) || 'null');
    // Only restore a payload with the shape the renderers assume; a stale or
    // hand-edited one must not take the page down on load.
    if (s && Array.isArray(s.games) && s.games.length
        && s.cur >= 0 && s.cur < s.games.length) {
      s.games.forEach(g => {
        if (!Array.isArray(g.picks) || g.picks.length !== 10)
          g.picks = Array(10).fill(null);
        g.bans = g.bans || {};
        if (!Array.isArray(g.bans.blue)) g.bans.blue = [];
        if (!Array.isArray(g.bans.red)) g.bans.red = [];
        // v1 stored one running pool for the whole series; drop it rather
        // than reinterpreting it as game 1's, which would be wrong for
        // every later game.
        if (!Array.isArray(g.fearless)) g.fearless = [];
        g.firstPick = g.firstPick ? 1 : 0;
      });
      delete s.fearless;
      SERIES = s;
    }
  } catch (e) { /* corrupt or storage disabled: keep the fresh series */ }
  loadGame(SERIES.cur);
}

function saveSeries() {
  try { localStorage.setItem(SERIES_KEY, JSON.stringify(SERIES)); }
  catch (e) { /* private mode / quota: the session still works */ }
}

// The live `picks`/`bans` are the board of SERIES.games[SERIES.cur]; these two
// move state between the live board and the slot.
function syncGame() {
  if (!SERIES) return;
  const g = SERIES.games[SERIES.cur];
  if (!g) return;
  g.picks = picks.slice();
  g.bans = { blue: bans.blue.slice(), red: bans.red.slice() };
  g.firstPick = currentFirstPick();
  saveSeries();
}

function loadGame(i) {
  if (!SERIES || i < 0 || i >= SERIES.games.length) return;
  SERIES.cur = i;
  const g = SERIES.games[i];
  picks = g.picks.slice();
  bans = { blue: g.bans.blue.slice(), red: g.bans.red.slice() };
  lastSlot = null;
  document.getElementById('first-pick').value = g.firstPick ? '1' : '0';
  // `cur` is part of the persisted state, so switching games has to be
  // saved here - otherwise "next game in series" (which saves *before*
  // calling this) stores the previous index and a refresh lands on the
  // wrong game.
  saveSeries();
  renderAll();
  requestSuggest();
}

function addGameSlot() {
  syncGame();
  const cur = SERIES.games[SERIES.cur];
  // A fresh game inherits the pool the current one had - it is the same
  // point in the series, so the same champions are gone - plus nothing,
  // because the current game has not been closed.
  SERIES.games.push(newGameSlot(cur ? cur.firstPick : 0,
                                cur ? cur.fearless : []));
  saveSeries();
  if (WT) clearWT();
  loadGame(SERIES.games.length - 1);
}

function nextGameInSeries() {
  syncGame();
  const g = SERIES.games[SERIES.cur];
  // This game's pool + everything picked in it = the next game's pool.
  const pool = g.fearless.slice();
  g.picks.filter(Boolean).forEach(r => {
    if (!pool.includes(r)) pool.push(r);
  });
  g.done = true;
  // First-pick side is *kept*, not flipped. The teams swap sides between
  // games but the side holding first pick stays the same in 94% of pro
  // games (18,614 vs 1,139 in the corpus), so keeping it is the better
  // default - and the select above still overrides it.
  SERIES.games.splice(SERIES.cur + 1, 0, newGameSlot(g.firstPick, pool));
  if (WT) clearWT();
  saveSeries();
  loadGame(SERIES.cur + 1);
}

function clearSeries() {
  SERIES = { games: [newGameSlot(0)], cur: 0 };
  if (WT) clearWT();
  saveSeries();
  loadGame(0);
}

function addFearless(riot) {
  if (!SERIES || !riot) return false;
  const pool = fearlessPool();
  if (pool.includes(riot)) return false;
  pool.push(riot);
  saveSeries();
  return true;
}

function removeFearless(riot) {
  if (!SERIES) return false;
  const g = SERIES.games[SERIES.cur];
  const n = g.fearless.length;
  g.fearless = g.fearless.filter(r => r !== riot);
  if (g.fearless.length === n) return false;
  saveSeries();
  return true;
}

function isFearless(riot) {
  return !!(SERIES && riot != null && fearlessPool().includes(riot));
}

function renderSeries() {
  const box = document.getElementById('series-games');
  if (!box || !SERIES) return;
  box.innerHTML = '';
  SERIES.games.forEach((g, i) => {
    const n = g.picks.filter(Boolean).length;
    const fp = g.firstPick ? 'red' : 'blue';
    const b = document.createElement('button');
    b.className = 'sg' + (i === SERIES.cur ? ' cur' : '')
      + (g.done ? ' done' : '');
    b.innerHTML = `<span class="g">G${i + 1}</span>`
      + `<span class="fp" title="first pick: ${fp}">${g.firstPick ? 'R' : 'B'}</span>`
      + `<span class="n">${n}/10</span>`;
    b.title = `game ${i + 1} · first pick ${fp} · ${n}/10 picks`
      + ` · ${g.fearless.length} fearless`
      + (g.done ? ' · closed' : '');
    b.addEventListener('click', () => {
      if (i === SERIES.cur) return;
      syncGame();
      if (WT) clearWT();
      loadGame(i);
    });
    box.appendChild(b);
  });
  const cur = SERIES.games[SERIES.cur];
  document.getElementById('series-info').textContent =
    `game ${SERIES.cur + 1}/${SERIES.games.length}`
    + ` · first pick ${cur.firstPick ? 'red' : 'blue'}`
    + ` · ${cur.fearless.length} fearless`;
  // A closed game already has its successor; offering "next game" there
  // would splice a duplicate in between. Use "+ game" to insert one.
  const nextBtn = document.getElementById('series-next');
  const blocked = cur.done && SERIES.cur < SERIES.games.length - 1;
  nextBtn.disabled = blocked;
  nextBtn.title = blocked
    ? 'this game is already closed - its successor is in the series. Use'
      + ' "＋ game" to insert one, or open the last game to extend the series'
    : 'close this game and open the next one: every champion picked here'
      + ' becomes unavailable there. The pool is defined exactly that way in'
      + ' the pro corpus (verified on all 10,021 fearless rows), and the'
      + ' first-pick side is kept because it stays the same across a series'
      + ' in 94% of pro games.';
}

function renderFearless() {
  const strip = document.getElementById('fearless-strip');
  if (!strip || !SERIES) return;
  const pool = fearlessPool();
  strip.innerHTML = '';
  pool.forEach(riot => {
    const c = BY_RIOT[riot];
    if (!c) return;
    const el = document.createElement('div');
    el.className = 'fb-slot';
    el.title = `${c.name} — picked earlier in this series: cannot be picked`
      + ' again, can still be banned';
    el.innerHTML = '<button class="clear" title="remove">✕</button>'
      + `<div class="pic">${imgTag(c, 'ic')}</div>`;
    el.querySelector('.clear').addEventListener('click', ev => {
      ev.stopPropagation();
      if (removeFearless(riot)) changed();
    });
    strip.appendChild(el);
  });
  document.getElementById('fearless-bar')
    .classList.toggle('empty-pool', !pool.length);
  const note = document.getElementById('fearless-note');
  let txt = pool.length ? `${pool.length} unavailable`
    : 'this game — the pool is empty';
  // The masked models have no unavailable input at all, so the server can
  // only drop those champions from the answer - the ranking is not
  // fearless-aware. Say so instead of implying it was handled.
  const filtered = !!(resp && resp.fearless_filtered);
  if (pool.length && filtered) txt += ' · role board filters them out only';
  note.textContent = txt;
  note.title = pool.length && filtered
    ? 'The role model was trained without a fearless mask: its top-5 was'
      + ' computed as if these champions were available, then they were'
      + ' removed. The next-move model takes the pool as an input and is'
      + ' not affected by this caveat.'
    : '';
  note.classList.toggle('warn', pool.length > 0 && filtered);
}

function excludedSet() {
  const ex = new Set(picks.filter(Boolean));
  allBansFlat().forEach(b => ex.add(b));
  const ctx = pickerCtx || {};
  // Fearless-unavailable champions cannot be picked, and cannot be added to
  // the pool twice. They *stay* selectable for a ban, because banning what
  // the opponent picked last game is legal and common.
  if (ctx.type === 'pick' || ctx.type === 'fearless')
    fearlessPool().forEach(f => ex.add(f));
  if (ctx.type === 'pick' && picks[ctx.slot])
    ex.delete(picks[ctx.slot]);   // current pick stays selectable
  return ex;
}

function renderPickerGrid(q) {
  const grid = document.getElementById('picker-grid');
  grid.innerHTML = '';
  const ex = excludedSet();
  const needle = (q || '').toLowerCase();
  let n = 0;
  CHAMPIONS.forEach(c => {
    if (ex.has(c.riot)) return;
    if (needle && !c.name.toLowerCase().includes(needle)) return;
    n++;
    const b = document.createElement('button');
    b.className = 'champ';
    b.innerHTML = imgTag(c, 'ic') + `<span>${c.name}</span>`;
    b.addEventListener('click', () => {
      const ctx = pickerCtx;
      if (ctx.type === 'ban') bans[ctx.team].push(c.riot);
      else if (ctx.type === 'fearless') addFearless(c.riot);
      else { picks[ctx.slot] = c.riot; lastSlot = ctx.slot; }
      closePicker();
      changed();
    });
    grid.appendChild(b);
  });
  if (!n) grid.innerHTML = '<span class="muted">no champion matches</span>';
}

function openPicker(ctx) {
  pickerCtx = ctx;
  document.getElementById('picker-title').textContent =
    ctx.type === 'ban' ? ('Ban champion — ' + ctx.team)
      : ctx.type === 'fearless'
        ? 'Fearless — already picked in this series'
        : ('Pick champion — ' + (ctx.slot < 5 ? 'blue' : 'red'));
  const search = document.getElementById('picker-search');
  search.value = '';
  renderPickerGrid('');
  document.getElementById('picker').classList.remove('hidden');
  search.focus();
}

function closePicker() {
  document.getElementById('picker').classList.add('hidden');
  pickerCtx = null;
}

function requestSuggest() {
  clearTimeout(debounceT);
  // The lane belongs to one board state: once the board moves, the pick being
  // asked for is a different one and a lane carried over would be applied to
  // the wrong slot.
  NEXT_LANE = null;
  const sig = boardSig();
  debounceT = setTimeout(() => runSuggest(sig), 120);
}

async function postJSON(url, body) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    return await r.json();
  } catch (e) { return { error: String(e) }; }
}

async function runSuggest(sig) {
  // Both models answer the same board, independently: the masked one the
  // role board + per-role targets, the policy one the next move. A reply is
  // only accepted if the board has not moved since the request went out -
  // rendering an older assignment against newer picks is what made a fresh
  // pick look like it was ignored.
  const maskedP = ROLE_MODEL
    ? postJSON('/api/suggest',
      { picks: picks, bans: allBansFlat(), last_slot: lastSlot,
        fearless: fearlessPool() })
    : null;
  const policyP = isPolicyModel()
    ? postJSON('/api/next-move', {
      picks: picks, bans: allBansFlat(), first_pick: currentFirstPick(),
      fearless: fearlessPool(),
      patch_id: currentPatchId(),
      // Ignored by a plain nextpick-* head; the reranker needs it because the
      // prompt asks for a champion in a named lane.
      lane: NEXT_LANE,
      k: 8
    })
    : null;

  if (maskedP) {
    const m = await maskedP;
    if (sig !== boardSig()) return;            // a newer board is in flight
    if (m && m.error) {
      resp = null; respSig = null;
      boardError(m.error);
    } else {
      resp = m; respSig = sig;
      boardError('');
    }
    renderSlots(); renderRoles(); renderTurn();
    // the fearless note reads `resp.fearless_filtered`, which only exists
    // once this answer has landed
    renderFearless();
  }
  if (policyP) {
    const p = await policyP;
    if (sig !== boardSig()) return;
    if (p && !p.error) {
      presp = p; prespSig = sig;
      renderNextMove();
    } else {
      presp = null; prespSig = null;
      const msg = (p && p.error) || 'no suggestion returned';
      boardError(msg);
      renderNextMoveError(msg);   // do not overwrite with the busy state
    }
    renderExplain();       // hides itself if the board moved
  }
  renderCompare();
}

function boardError(msg) {
  const el = document.getElementById('board-error');
  if (!el) return;
  el.textContent = msg ? 'board: ' + msg : '';
  el.classList.toggle('hidden', !msg);
}

// ---- next-move policy panel --------------------------------------------

function renderNextMove() {
  const box = document.getElementById('nextmove');
  if (!isPolicyModel()) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  // Same freshness rule as the role board: an answer only describes the
  // board it was computed for. Showing the previous board's "Blue to ban"
  // beside a highlight that already moved on contradicts it, so blank the
  // panel while the new answer is in flight.
  const fresh = !!(presp && presp.state && prespSig === boardSig());
  if (!fresh) {
    document.getElementById('nm-turn').innerHTML = '<b>updating…</b>';
    document.getElementById('nm-meta').textContent = '';
    document.getElementById('nm-lane').innerHTML = '';
    document.getElementById('nm-chips').innerHTML = '';
    document.getElementById('nm-warning').classList.add('hidden');
    return;
  }
  const st = presp.state;
  const rr = presp.rerank;
  const sideName = st.side === 0 ? 'Blue' : 'Red';
  document.getElementById('nm-turn').innerHTML = st.complete
    ? '<b>Draft complete</b>'
    : `<b>${sideName} to ${st.kind}</b>`;
  document.getElementById('nm-meta').textContent = st.complete
    ? '' : `move ${st.move_idx + 1}/20 · ${st.phase}`
      + (presp.patch ? ` · patch ${presp.patch}` : '')
      + (st.n_unavailable ? ` · ${st.n_unavailable} fearless` : '');
  const warn = document.getElementById('nm-warning');
  // A failed rerank is not a failed answer: the head's list still stands, so
  // it is reported next to the chips rather than replacing them.
  const msgs = [];
  if (st.warning) msgs.push('board warning: ' + st.warning);
  if (rr && rr.error) msgs.push('rerank failed, showing the head: ' + rr.error);
  if (msgs.length) {
    warn.textContent = msgs.join(' · ');
    warn.classList.remove('hidden');
  } else {
    warn.classList.add('hidden');
  }
  const chips = document.getElementById('nm-chips');
  chips.innerHTML = '';
  renderLanePick(rr, st);
  if (!st.complete) {
    (presp.suggestions || []).forEach(c => {
      const b = document.createElement('button');
      // `llm_pick` marks the reranker's choice. The number shown is the LLM's
      // k-way probability when it scored the list - the head's own probability
      // stays in the tooltip, because 31 % of 173 champions and 31 % of five
      // candidates are not the same claim.
      b.className = 'nm-chip' + (c.llm_pick ? ' llm-pick' : '');
      const shown = c.llm_prob != null ? c.llm_prob : c.prob;
      b.innerHTML = imgTag(BY_IDX[c.idx])
        + `<span class="nm">${c.name}</span>`
        + `<span class="p">${(100 * shown).toFixed(1)}%</span>`
        + (c.llm_pick ? '<span class="nm-llm">LLM</span>' : '');
      b.title = c.llm_prob != null
        ? `${sideName} ${st.kind}: ${c.name} — head ${(100 * c.prob).toFixed(1)}%, LLM ${(100 * c.llm_prob).toFixed(1)}%`
        : `${sideName} ${st.kind}: ${c.name}`;
      b.addEventListener('click', () => applyMove(st.kind, st.side, c.idx));
      chips.appendChild(b);
    });
  }
}

// The lane of the coming pick is a decision, not a fact, until four lanes are
// filled. Choosing one now *constrains the answer*: the head's pool is masked
// to champions that play that lane, so "Bot" returns ADCs instead of the head's
// unconstrained favourites. Leaving it unset keeps the lane as a prompt hint and
// says so, because filtering on a guess would make a wrong guess unrecoverable.
function renderLanePick(rr, st) {
  const box = document.getElementById('nm-lane');
  box.innerHTML = '';
  if (!rr || rr.error || st.complete || st.kind !== 'pick') return;
  const open = rr.lanes_open || [];
  if (!open.length) return;
  const names = ['Top', 'Jungle', 'Mid', 'Bot', 'Support'];
  const lf = rr.lane_filter || {};
  const pool = lf.applied ? ` · ${lf.n_legal} champions` : '';
  if (open.length === 1) {
    // Forced: the board already decides the lane, so the pool is masked without
    // asking and there is nothing to choose.
    box.textContent = `· lane ${names[open[0]]}${pool}`;
    return;
  }
  const lab = document.createElement('span');
  lab.textContent = '· lane ';
  const sel = document.createElement('select');
  sel.className = 'nm-lane-sel';
  sel.title = 'the lane this pick is for — choosing one keeps only champions '
    + 'that play it; leaving it on "any" asks for a champion in the first open '
    + 'lane without constraining the list';
  // A placeholder, so picking the lane that happens to be first is still a
  // change event. Without it the default lane is displayed but cannot be
  // activated: selecting the option already shown fires nothing.
  const any = document.createElement('option');
  any.value = '';
  any.textContent = 'any';
  sel.appendChild(any);
  open.forEach(r => {
    const o = document.createElement('option');
    o.value = String(r);
    o.textContent = names[r];
    sel.appendChild(o);
  });
  sel.value = rr.lane_assumed ? '' : String(rr.lane);
  sel.addEventListener('change', () => {
    NEXT_LANE = sel.value === '' ? null : parseInt(sel.value, 10);
    const sig = boardSig();
    renderNextMove();               // blank the stale answer while it recomputes
    runSuggest(sig);
  });
  box.appendChild(lab);
  box.appendChild(sel);
  const hint = document.createElement('span');
  hint.className = 'nm-assumed';
  if (lf.applied) {
    hint.textContent = pool;
    hint.title = `only champions that play ${names[rr.lane]} can be suggested`;
  } else {
    hint.textContent = ' assumed — list not filtered';
    hint.title = 'the board does not force this lane yet: it is in the prompt, '
      + 'but the list is the head\'s unconstrained top-k';
  }
  box.appendChild(hint);
}

function renderNextMoveError(msg) {
  const box = document.getElementById('nextmove');
  if (!isPolicyModel()) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  document.getElementById('nm-turn').innerHTML = '<b>no suggestion</b>';
  document.getElementById('nm-meta').textContent = '';
  document.getElementById('nm-lane').innerHTML = '';
  document.getElementById('nm-chips').innerHTML = '';
  const warn = document.getElementById('nm-warning');
  warn.textContent = msg;
  warn.classList.remove('hidden');
}

// ---- explanation panel -------------------------------------------------
// The model's reasoning is requested explicitly (a `reasoning` field in the
// answer schema) and rendered here. It is the model's own account, not a
// transcript of hidden deliberation, so the panel labels it as such and puts
// the citations - the only part that is actually checked - underneath it.

let EXPLAIN = null;        // last /api/explain payload
let EXPLAIN_BUSY = false;
let EXPLAIN_SIG = null;    // board the explanation was fetched for

function boardSig() {
  // The patch is part of the answer, not of the board: switching it must
  // invalidate an in-flight reply exactly like a pick does. The lane is too -
  // the reranker's answer is "the best champion *for this lane*".
  return JSON.stringify([picks, bans.blue, bans.red, currentFirstPick(),
                         SERIES ? fearlessPool() : [], currentPatchId(),
                         NEXT_LANE]);
}

function hideExplain() {
  EXPLAIN = null;
  EXPLAIN_BUSY = false;
  document.getElementById('explain').classList.add('hidden');
}

function explainBody(force) {
  return {
    picks: picks,
    bans: allBansFlat(),
    first_pick: currentFirstPick(),
    patch_id: currentPatchId(),   // the same row the next-move model used
    // The panel's list and the prose must be the same list: same fearless
    // pool, same lane. Without the lane the reranker would assume one and
    // explain a champion the panel did not offer.
    fearless: fearlessPool(),
    lane: NEXT_LANE,
    k: 8,
    k_games: 5,
    force: !!force
  };
}

async function requestExplain(force) {
  if (!isPolicyModel() || EXPLAIN_BUSY) return;
  const sig = boardSig();
  EXPLAIN_BUSY = true;
  const box = document.getElementById('explain');
  box.classList.remove('hidden');
  document.getElementById('ex-title').textContent = 'Why this move';
  document.getElementById('ex-meta').textContent = '';
  document.getElementById('ex-body').innerHTML =
    '<div class="ex-busy">asking the model</div>';
  let d;
  try {
    d = await (await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(explainBody(force))
    })).json();
  } catch (e) {
    d = { error: 'request failed: ' + e.message };
  }
  EXPLAIN_BUSY = false;
  if (d.error) {
    document.getElementById('ex-body').innerHTML =
      `<div class="ex-drop">${esc(d.error)}</div>`;
    return;
  }
  EXPLAIN = d;
  EXPLAIN_SIG = sig;
  renderExplain();
}

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function gameById(id) {
  const g = ((EXPLAIN && EXPLAIN.retrieval) || {}).games || [];
  return g.find(x => x.gameid === id) || null;
}

function renderExplain() {
  const box = document.getElementById('explain');
  // a stale explanation is worse than none: the board moved since
  if (!EXPLAIN || !EXPLAIN_SIG || EXPLAIN_SIG !== boardSig()) {
    if (!EXPLAIN_BUSY) box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  const ex = EXPLAIN.explanation;
  const llm = EXPLAIN.llm || {};
  const patch = EXPLAIN.patch || {};
  const games = (EXPLAIN.retrieval || {}).games || [];

  document.getElementById('ex-meta').textContent = ex
    ? (ex.model || 'model') + (ex.cached ? ' · cached' : '')
      + (ex.ok ? '' : ' · unverified citations')
    : (llm.configured ? '' : 'no LLM configured');
  document.getElementById('ex-reask').classList.toggle('hidden',
    !llm.configured);

  const out = [];

  if (!ex) {
    out.push(`<div class="ex-empty">${esc(llm.reason ||
      llm.error || 'no explanation returned')}</div>`);
  } else {
    if (ex.reasoning && ex.reasoning.length) {
      out.push('<ol class="ex-steps">'
        + ex.reasoning.map(s => `<li>${esc(s)}</li>`).join('') + '</ol>');
      out.push('<div class="ex-note">the model\'s account, in its own words —'
        + ' only the citations below are checked</div>');
    }
    const cand = (presp && presp.suggestions || []).find(s => s.name === ex.recommend);
    const pick = cand
      ? `<button class="ex-pick" data-apply="${cand.idx}"
           title="apply ${esc(ex.recommend)}">${imgTag(BY_IDX[cand.idx])}`
        + `<span>${esc(ex.recommend)}</span>`
        + `<span class="p">${(100 * cand.prob).toFixed(1)}%</span></button>`
      : `<span class="ex-pick"><span>${esc(ex.recommend || '?')}</span></span>`;
    out.push('<div class="ex-answer">'
      + `<div class="k">pick</div><div class="v">${pick}</div>`
      + `<div class="k">why</div><div class="v">${esc(ex.why)}</div>`
      + `<div class="k">their answer</div><div class="v">${esc(ex.their_answer)}</div>`
      + `<div class="k">risk</div><div class="v">${esc(ex.risk)}</div>`
      + '</div>');

    // evidence - resolved against the retrieval payload so the reader can see
    // *why* it is evidence, not just an id
    out.push('<div class="ex-sub">Cited evidence</div>');
    const ev = ex.evidence || [];
    if (!ev.length) {
      out.push('<div class="ex-empty">the model cited nothing</div>');
    } else {
      out.push('<div class="ex-ev">');
      ev.forEach(item => {
        if (item.type === 'game') {
          const g = gameById(item.id);
          out.push('<div class="ex-game">'
            + `<span class="gid">${esc(item.id)}</span>`
            + (g ? `<span class="who">${esc(g.blue_team)} vs ${esc(g.red_team)}</span>`
              + `<span class="res">${esc(g.league || '')} · ${esc(g.patch || '')}`
              + ` · ${esc(g.date || '')}</span>`
              + `<span class="res">next: ${esc(g.next_side)} ${esc(g.next_kind)}`
              + ` ${esc(String(g.next_champ))}</span>`
              + `<span class="ov">overlap ${(100 * g.overlap).toFixed(0)}%</span>`
              : '<span class="res">not in the retrieved set</span>')
            + '</div>');
        } else {
          out.push(`<div class="ex-line">${esc(item.line)}</div>`);
        }
      });
      out.push('</div>');
    }

    const dropped = ex.unverified || [];
    if (dropped.length || (ex.problems || []).length) {
      out.push('<div class="ex-drop">'
        + '<b>Not shown as evidence</b><ul>'
        + dropped.map(u => `<li><code>${esc(u.id || u.line || JSON.stringify(u))}`
          + '</code></li>').join('')
        + (ex.problems || []).map(p => `<li>${esc(p)}</li>`).join('')
        + '</ul></div>');
    }
  }

  // what the model was actually shown - the audit trail for the answer above
  out.push('<div class="ex-sub">Comparable pro drafts'
    + (EXPLAIN.retrieval && EXPLAIN.retrieval.relaxed_side
      ? ' <span class="muted">(first-pick side filter relaxed)</span>' : '')
    + '</div>');
  if (!games.length) {
    out.push('<div class="ex-empty">no comparable board in the corpus</div>');
  } else {
    out.push('<div class="ex-ret">' + games.map(g =>
      `<div class="r"><span>${esc(g.blue_team)} vs ${esc(g.red_team)}`
      + ` <span class="muted">${esc(g.league || '')} ${esc(g.patch || '')}</span></span>`
      + `<span class="ov-bar"><i style="width:${(100 * g.overlap).toFixed(0)}%"></i></span>`
      + `<span class="ov-n">${(100 * g.overlap).toFixed(0)}%</span></div>`
    ).join('') + '</div>');
  }
  if (patch.available === false) {
    out.push('<div class="ex-empty" style="margin-top:10px">no patch notes on '
      + 'this machine — the model was told not to claim anything about patch '
      + 'changes</div>');
  } else if (!Object.keys(patch.notes || {}).length) {
    out.push(`<div class="ex-empty" style="margin-top:10px">patch ${esc(patch.patch || '?')}`
      + ' was read; none of the candidates changed</div>');
  }

  document.getElementById('ex-body').innerHTML = out.join('');
  const apply = document.querySelector('#ex-body [data-apply]');
  if (apply && presp && presp.state && !presp.state.complete) {
    apply.addEventListener('click', () =>
      applyMove(presp.state.kind, presp.state.side, +apply.dataset.apply));
  }
}

function applyMove(kind, side, idx) {
  const c = BY_IDX[idx];
  if (!c) return;
  if (kind === 'ban') {
    const team = side === 0 ? 'blue' : 'red';
    if (bans[team].length >= 5) return;
    if (bans[team].includes(c.riot)) return;
    bans[team].push(c.riot);
  } else {
    // The suggestion may predate a pool edit, so re-check legality here
    // rather than trusting the answer.
    if (picks.includes(c.riot) || isFearless(c.riot)) return;
    const base = side === 0 ? 0 : 5;
    for (let s = base; s < base + 5; s++) {
      if (picks[s] == null) { picks[s] = c.riot; lastSlot = s; break; }
    }
  }
  changed();
}

// ---- walkthrough of real pro drafts -------------------------------------

function optLabel(d) {
  return `${(d.date || '').slice(0, 10)} · ${d.league} · ${d.blue} vs `
    + `${d.red} (G${d.game_number})` + (d.fearless ? ' · fearless' : '');
}

function renderCompare() {
  const box = document.getElementById('wt-compare');
  if (!WT) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  const m = WT.draft.moves[WT.step];
  if (!m) {
    box.innerHTML = `<b>Draft complete</b> — all ${WT.draft.moves.length}`
      + ` moves applied · ${WT.draft.winner} side won`
      + ' · step back with ◀ or pick another game';
    return;
  }
  const sideName = m.side === 0 ? 'blue' : 'red';
  const sideLabel = m.side === 0 ? 'Blue' : 'Red';
  const head = `<b>Move ${WT.step + 1}/${WT.draft.moves.length}</b>`
    + ` — ${sideLabel} to ${m.kind}`;
  if (m.kind === 'ban') {
    box.innerHTML = `${head}: ${imgTag(BY_IDX[m.idx])} <b>${m.name}</b>`
      + (isPolicyModel()
        ? ' — the policy reads bans as input, so this is part of its state.'
        : ' — bans are a post-filter; they never change what the model'
          + ' conditions on.');
    return;
  }
  const role = ROLES[m.role];
  // The move being judged is the one the board is about to take, so the
  // answer to judge it against is the one for the *current* board - and
  // only if it actually describes this board.
  const fresh = boardSig();
  if (isPolicyModel()) {
    const list = (presp && prespSig === fresh && presp.suggestions) || [];
    const k = list.findIndex(c => c.idx === m.idx);
    let verdict;
    if (k >= 0) {
      verdict = `policy ranked it <b>#${k + 1}</b>`
        + ` (${(100 * list[k].prob).toFixed(1)}%)`;
    } else {
      verdict = '<b>not in the policy\'s top-8</b>'
        + (list[0] ? ` (its #1 was ${list[0].name})` : '');
    }
    box.innerHTML = `${head}: ${imgTag(BY_IDX[m.idx])} <b>${m.name}</b>`
      + ` <span class="muted">(${role})</span> — ${verdict}`;
    return;
  }
  const lists = (resp && respSig === fresh && resp[sideName])
    ? resp[sideName] : {};
  const inRole = lists[role] || [];
  let rank = -1, prob = 0;
  inRole.forEach((c, k) => {
    if (c.idx === m.idx) { rank = k + 1; prob = c.prob; }
  });
  const elsewhere = [];
  ROLES.forEach(rn => {
    if (rn === role) return;
    (lists[rn] || []).forEach((c, k) => {
      if (c.idx === m.idx)
        elsewhere.push(`${rn} #${k + 1} (${(100 * c.prob).toFixed(1)}%)`);
    });
  });
  let verdict;
  if (rank > 0) {
    verdict = `model ranked it <b>#${rank} for ${role}</b>`
      + ` (${(100 * prob).toFixed(1)}%)`;
  } else {
    const top1 = inRole[0] ? `model's #1 is ${inRole[0].name}` : 'no list';
    verdict = `<b>not in the model's top-5 for ${role}</b> (${top1})`;
  }
  if (elsewhere.length) verdict += ` · also ${elsewhere.join(', ')}`;
  box.innerHTML = `${head}: ${imgTag(BY_IDX[m.idx])} <b>${m.name}</b>`
    + ` <span class="muted">(${role})</span> — ${verdict}`;
}

function wtApply() {
  picks = Array(10).fill(null);
  bans = { blue: [], red: [] };
  lastSlot = null;
  if (WT) {
    WT.actualRole = {};
    const next = [0, 0];                 // next free pick slot per team
    for (let k = 0; k < WT.step; k++) {
      const m = WT.draft.moves[k];
      if (m.kind === 'ban') {
        bans[m.side === 0 ? 'blue' : 'red'].push(m.riot);
      } else if (m.side < 2 && next[m.side] < 5) {
        lastSlot = m.side * 5 + next[m.side];   // newest real pick
        picks[lastSlot] = m.riot;
        next[m.side] += 1;
        WT.actualRole[m.riot] = m.role;
      }
    }
  }
  syncGame();          // the game slot mirrors the walkthrough position
  changed();
}

async function wtLoad(i) {
  let d;
  try { d = await (await fetch('/api/draft?i=' + i)).json(); }
  catch (e) { alert('failed to load the game'); return; }
  if (d.error) { alert(d.error); return; }
  WT = { draft: d, step: 0 };
  const sel = document.getElementById('wt-select');
  if (![...sel.options].some(o => o.value === String(i))) {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = optLabel(d);
    sel.appendChild(o);
  }
  sel.value = String(i);
  const fp = document.getElementById('first-pick');
  fp.value = d.first_pick ? '1' : '0';
  fp.disabled = true;              // the game dictates the order
  setPatchByKey(d.patch);          // ... and the patch it was played on
  // Seed the pool from the game itself: `unavailable` is the real fearless
  // set for this game (champions picked in earlier games of the same
  // series), so a walkthrough of G3 starts with the 20 that are actually
  // gone instead of an empty pool.
  SERIES.games[SERIES.cur].fearless = (d.unavailable || []).slice();
  saveSeries();
  renderWalkthrough();
  wtApply();   // changed() -> suggest -> renderCompare on response
  renderCompare();
}

// Drop the walkthrough without touching the board - used when the board is
// about to be replaced by another game slot.
function clearWT() {
  WT = null;
  document.getElementById('wt-select').value = '';
  document.getElementById('first-pick').disabled = false;
  renderWalkthrough();
}

function wtExit() {
  clearWT();
  // The pool was seeded from the real game, so leaving drops it rather than
  // leaving a hand-built board behind a wall of bans it never asked for.
  SERIES.games[SERIES.cur].fearless = [];
  saveSeries();
  resetDraft();
}

function resetDraft() {
  picks = Array(10).fill(null);
  bans = { blue: [], red: [] };
  lastSlot = null;
  syncGame();          // empties the slot on the board, keeps the pool
  changed();
  renderWalkthrough();
  renderCompare();
}

function renderWalkthrough() {
  const info = document.getElementById('wt-info');
  const stepEl = document.getElementById('wt-step');
  const prev = document.getElementById('wt-prev');
  const next = document.getElementById('wt-next');
  const exit = document.getElementById('wt-exit');
  if (!WT) {
    info.innerHTML = '<span class="muted">no game loaded</span>';
    info.title = '';
    stepEl.textContent = '';
    prev.disabled = next.disabled = exit.disabled = true;
    return;
  }
  const d = WT.draft;
  const nUn = (d.unavailable || []).length;
  info.innerHTML = `<b>${d.date.slice(0, 10)}</b> ${d.league} · `
    + `${d.blue} vs ${d.red} · G${d.game_number} · patch ${d.patch} · `
    + `first pick ${d.first_pick ? 'red' : 'blue'}`
    + (d.fearless ? ` · fearless (${nUn} gone)` : ' · not fearless') + ' · '
    + `<span class="muted">${d.winner} side won</span>`;
  info.title = d.date + ' · ' + d.gameid;
  stepEl.textContent = `${WT.step}/${d.moves.length}`;
  prev.disabled = WT.step === 0;
  next.disabled = WT.step >= d.moves.length;
  exit.disabled = false;
}

async function refreshDraftList(q) {
  const sel = document.getElementById('wt-select');
  const cur = sel.value;
  let d;
  try {
    d = await (await fetch('/api/drafts?limit=100&q=' +
      encodeURIComponent(q || ''))).json();
  } catch (e) { return; }
  if (d.error) return;
  sel.innerHTML = `<option value="">— pick a game `
    + `(${d.total || 0}) —</option>`;
  (d.drafts || []).forEach(dr => {
    const o = document.createElement('option');
    o.value = String(dr.i);
    o.textContent = optLabel(dr);
    sel.appendChild(o);
  });
  if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
}

document.getElementById('btn-reset').addEventListener('click', () => {
  if (WT) { wtExit(); return; }
  resetDraft();
});
document.getElementById('series-add').addEventListener('click', addGameSlot);
document.getElementById('series-next').addEventListener('click',
  nextGameInSeries);
document.getElementById('series-clear').addEventListener('click', () => {
  if (confirm('Clear every game in the series and its fearless pool?'))
    clearSeries();
});
document.getElementById('fearless-add').addEventListener('click', () =>
  openPicker({ type: 'fearless' }));
document.getElementById('first-pick').addEventListener('change', () => {
  syncGame();                  // the value belongs to the game on the board
  if (!WT) changed();          // the whole order mirrors
});
document.getElementById('patch-select').addEventListener('change', () => {
  // A patch change re-asks both panels: the next move conditions on it and
  // the explanation's notes default to it.
  hideExplain();
  changed();
});
document.getElementById('nm-explain').addEventListener('click',
  () => requestExplain(false));
document.getElementById('ex-reask').addEventListener('click',
  () => requestExplain(true));
document.getElementById('ex-close').addEventListener('click', hideExplain);
document.getElementById('picker-close').addEventListener('click', closePicker);
document.getElementById('picker').addEventListener('click', ev => {
  if (ev.target.id === 'picker') closePicker();
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') closePicker();
  if (!WT || ev.target.tagName === 'INPUT' ||
      ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowRight') document.getElementById('wt-next').click();
  if (ev.key === 'ArrowLeft') document.getElementById('wt-prev').click();
});
document.getElementById('picker-search').addEventListener('input', ev =>
  renderPickerGrid(ev.target.value));

// model selection: server-side list + upload a file from this machine.
// Two independent slots - "role" (masked) and "next move" (policy) - so
// loading one never unloads the other.
const loadInto = (role) => {
  const id = role === 'policy' ? 'policy-select' : 'ckpt-select';
  const path = document.getElementById(id).value;
  if (!path && role !== 'policy') return;
  switchModel('/api/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: path || null, role: role })
  }, role);
};
document.getElementById('ckpt-select').addEventListener('focus',
  refreshCheckpoints);   // pick up newly trained runs without a restart
document.getElementById('policy-select').addEventListener('focus',
  refreshCheckpoints);
document.getElementById('btn-load-ckpt').addEventListener('click',
  () => loadInto('masked'));
document.getElementById('btn-load-policy').addEventListener('click',
  () => loadInto('policy'));
document.getElementById('btn-upload').addEventListener('click', () =>
  document.getElementById('file-input').click());
document.getElementById('file-input').addEventListener('change', async ev => {
  const f = ev.target.files && ev.target.files[0];
  ev.target.value = '';              // allow re-picking the same file
  if (!f) return;
  if (!/\.pt$/i.test(f.name)) {
    alert('expected a .pt checkpoint file');
    return;
  }
  const buf = await f.arrayBuffer();
  await switchModel('/api/upload?name=' + encodeURIComponent(f.name), {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: buf
  });
});

// walkthrough of real pro drafts
document.getElementById('wt-select').addEventListener('change', ev => {
  if (ev.target.value !== '') wtLoad(+ev.target.value);
});
document.getElementById('wt-filter').addEventListener('input', ev => {
  clearTimeout(wtFilterT);
  wtFilterT = setTimeout(() => refreshDraftList(ev.target.value), 300);
});
document.getElementById('wt-random').addEventListener('click', async () => {
  let d;
  try {
    d = await (await fetch('/api/drafts?random=1&limit=1')).json();
  } catch (e) { return; }
  if (d.error) { alert(d.error); return; }
  if (d.drafts && d.drafts.length) wtLoad(d.drafts[0].i);
});
document.getElementById('wt-next').addEventListener('click', () => {
  if (!WT || WT.step >= WT.draft.moves.length) return;
  WT.step++;
  wtApply();
  renderWalkthrough();
});
document.getElementById('wt-prev').addEventListener('click', () => {
  if (!WT || WT.step === 0) return;
  WT.step--;
  wtApply();
  renderWalkthrough();
});
document.getElementById('wt-exit').addEventListener('click', wtExit);

init();

