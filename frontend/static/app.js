// The UI's role vocabulary. `/api/champions` publishes the server's own list
// (`ROLE_ROWS`), and init() adopts it, so the two can never drift again: the
// role board used to be keyed by name while this file indexed it by number.
let ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"];
let CHAMPIONS = [], BY_IDX = {}, BY_RIOT = {};
let picks = Array(10).fill(null);          // slot -> riot id | null (pick order)
let bans = { blue: [], red: [] };          // team -> [riot id]
let resp = null;                           // last /api/suggest (the role board)
let respSig = null;                        // board `resp` was computed for
let presp = null;                          // last /api/next-move (the head)
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
  // One role vocabulary, published by the server. A stale client with a
  // different list would render `undefined` under a pick and leave its role
  // cell empty even though `/api/suggest` answered it correctly.
  if (Array.isArray(d.roles) && d.roles.length === 5) ROLES = d.roles.slice();
  applyPatchPayload(d);
  setModels(d.model, d.policy_model);
  refreshDraftList(wtFilterValue());
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

function renderModelInfo() {
  // The header no longer names the loaded weights - the checkpoints are chosen
  // on the command line (--ckpt / --policy-ckpt), so the header text was just a
  // long suffix string. What still has to happen here is the layout state: the
  // two slots are independent and either can be absent.
  document.body.classList.toggle('no-role-model', !ROLE_MODEL);
  document.body.classList.toggle('policy-mode', !!POLICY_MODEL);
  if (!POLICY_MODEL) document.getElementById('nextmove').classList.add('hidden');
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

function currentPatchId() {
  const sel = document.getElementById('patch-select');
  return sel && sel.value ? +sel.value : 0;
}

function currentPatchKey() {
  // The label the state will carry, so the agent's board and the state store's
  // agree about which patch the head was conditioned on. Resolved from the
  // same option list the selector shows - never from the text of the option.
  const sel = document.getElementById('patch-select');
  if (!sel || !sel.value) return null;
  const hit = [...sel.options].find(o => o.value === sel.value);
  return (hit && hit.dataset.key) || null;
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

// `/api/suggest` keys every role by its display name (`TOP`..`SUPPORT`), while
// a real pro move carries a numeric role id. Normalise both into one index.
// Indexing ROLES with the raw assignment value printed `undefined` under every
// picked champion and left that champion's role cell empty: `roleSlot["SUPPORT"]`
// was populated while the renderer looked up `roleSlot[4]`.
function roleIndex(value) {
  if (typeof value === 'number' && Number.isInteger(value)
      && value >= 0 && value < ROLES.length) return value;
  if (value == null) return null;
  const i = ROLES.indexOf(String(value).toUpperCase());
  return i < 0 ? null : i;
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
    //
    // One role and one confidence per pick now, not a five-way distribution:
    // the assignment comes from the champion-role table, which is one-to-one
    // per team, and `p_lane` is that champion's own share of its picks in the
    // role. A low number is a genuine flex, not a weak guess - the title says
    // which, so the badge never has to.
    const fresh = !!(resp && respSig === boardSig());
    const ar = fresh && resp.assignment ? resp.assignment[s] : null;
    const pl = fresh && resp.conf ? resp.conf[s] : null;
    const ai = roleIndex(ar);
    if (c && ai != null && pl != null) {
      rb.textContent = ROLES[ai] + ' ' + Math.round(100 * pl) + '%';
      const proRole = WT && WT.actualRole ? WT.actualRole[riot] : undefined;
      rb.title = `${ROLES[ai]} — ${c.name} fills this role in `
        + `${(100 * pl).toFixed(0)} % of its pro picks`
        + (pl < 0.6 ? ' (a flex pick)' : '')
        + (proRole != null ? ` · pro played ${ROLES[proRole]}` : '');
      if (proRole != null && proRole !== ai)
        rb.classList.add('mismatch');           // model vs pro disagreement
      else if (pl < 0.6)
        rb.classList.add('flex-rb');
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
  // Role board: five slots per team in TOP..SUPPORT order. A picked champ
  // sits in its assigned role; an empty role shows the champion the model
  // would take for it, with the next two candidates as small icons.
  //
  // The assignment is the champion-role table's, not a role head's: one-to-one
  // per team, so a flex champion can never appear in two lanes at once. The
  // targets are one `predict_in_lane` call per open cell, and the cell says
  // which model answered it - the head on a pick move, a masking checkpoint
  // during the ban phase. They are different models, so the title says which.
  //
  // `fresh` is the whole point: an answer only describes the board it was
  // computed for. Rendering the previous assignment against the current
  // picks put a fresh pick in an old role (or nowhere) and left the role
  // it belongs to showing its old target - "I picked Vi and it still says
  // Pantheon jungle". While a newer answer is in flight the board is drawn
  // dimmed, with no targets and no confidences, i.e. it never claims
  // anything it does not know.
  const fresh = !!(resp && respSig === boardSig());
  const methods = (fresh && resp.methods) || {};
  const notes = (fresh && resp.notes) || {};
  ['blue', 'red'].forEach(team => {
    const box = document.getElementById('roles-' + team);
    box.innerHTML = '';
    const base = team === 'blue' ? 0 : 5;
    const assign = (resp && resp.assignment) || [];
    const conf = (resp && resp.conf) || [];
    const roleSlot = {};
    for (let s = base; s < base + 5; s++) {
      const r = roleIndex(assign[s]);
      if (r != null) roleSlot[r] = s;
    }
    for (let r = 0; r < 5; r++) {
      const el = document.createElement('div');
      const slot = roleSlot[r];
      const c = (slot != null && picks[slot] != null)
        ? BY_RIOT[picks[slot]] : null;
      if (c) {
        const pl = fresh ? conf[slot] : null;
        el.className = 'role-slot filled' + (fresh ? '' : ' busy');
        el.title = `${ROLES[r]} — ${c.name}`
          + (pl != null ? ` · fills it in ${(100 * pl).toFixed(0)}% of its pro`
            + ' picks' : ' · placing…');
        el.innerHTML = `<div class="rlabel">${ROLES[r]}</div>`
          + `<div class="pic">${imgTag(c, 'ic big')}</div>`
          + `<div class="nm">${c.name}</div>`
          + `<div class="sub">${pl != null
            ? (100 * pl).toFixed(0) + '%' : '…'}</div>`;
      } else {
        const list = fresh && resp[team] && resp[team][ROLES[r]]
          ? resp[team][ROLES[r]] : [];
        const how = methods[team + '/' + ROLES[r]];
        const why = notes[team + '/' + ROLES[r]];
        if (list.length) {
          const top = list[0];
          const rest = list.slice(1, 3);
          el.className = 'role-slot empty target';
          el.title = `${ROLES[r]} — model target: ${top.name}`
            + ` (${(100 * top.prob).toFixed(1)} % of the champions that play it)`
            + (how ? ` · by the ${how === 'head' ? 'next-move head' : 'masking model'}` : '')
            + (top.flex ? ' · also listed under another role' : '');
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
          // A refused cell carries its reason: "load a masking checkpoint" is
          // actionable, an empty cell is not.
          if (why) el.title = why;
          el.innerHTML = `<div class="rlabel">${ROLES[r]}</div>`
            + '<div class="pic"></div>'
            + `<div class="nm muted">${fresh ? (why ? 'n/a' : '—') : '…'}</div>`;
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
    if (picks[s] == null) { picks[s] = c.riot; break; }
  }
  changed();
}

// Play a suggestion from a panel. Both the head's chips (/api/next-move) and
// the agent's one-click answer call this: they carry the same dense champion
// index, the same move kind and the same side. It used to be missing, so a
// click on a next-move chip raised a ReferenceError instead of playing the
// move. A ban of a fearless champion is legal (the pool only forbids picks),
// but a champion already picked or banned is not.
function applyMove(kind, side, idx) {
  const c = BY_IDX[idx];
  if (!c) return;
  const team = side === 1 || side === 'red' ? 'red' : 'blue';
  if (kind === 'ban') {
    if (allBansFlat().includes(c.riot) || picks.includes(c.riot)) return;
    bans[team].push(c.riot);
    changed();
    return;
  }
  if (kind === 'pick') addPick(team, idx);
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
  // The agent's reranked top-k is a view of this board, so it must follow every
  // move. `AgentPanel` is defined in agent.js (loaded after this file); guard so
  // a board change before it initializes is simply missed, not an error.
  if (window.AgentPanel) AgentPanel.scheduleAutoRank();
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

// The series context the board cannot carry: which game this is, how many
// games there are, and how far each one got. Every board publish sends it, so
// the state store (and therefore the agent) describes the game on screen
// rather than the last one it was told about. Numbers only - the server turns
// them into lines, and no client text reaches a model prompt.
function seriesBody() {
  if (!SERIES || !Array.isArray(SERIES.games)) return null;
  return {
    game: SERIES.cur + 1,
    of: SERIES.games.length,
    games: SERIES.games.map((g, i) => ({
      game: i + 1,
      picks: (g.picks || []).filter(Boolean).length,
      fearless: (g.fearless || []).length,
      closed: !!g.done
    }))
  };
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
  // Both answering paths mask the pool now, but not the same way: the
  // next-move head takes `unavailable` as an input and re-scores, while a
  // masking checkpoint was trained without it and can only have its answer
  // filtered and renormalised. Say which one is in play.
  const filtered = !!(resp && resp.fearless_filtered);
  const byMasker = !!(resp && resp.method === 'masker');
  if (pool.length && filtered && !byMasker) {
    txt += ' · masked out of the head\u2019s answer';
  }
  note.textContent = txt;
  note.title = pool.length && filtered
    ? (byMasker
      ? 'A masking checkpoint was trained without a fearless mask: its top-5'
        + ' was computed as if these champions were available, then they were'
        + ' removed and the rest renormalised. The next-move head takes the'
        + ' pool as an input and is not affected by this caveat.'
      : 'The next-move head takes the fearless pool as an input, so these'
        + ' champions were never in its pool.')
    : '';
  note.classList.toggle('warn', pool.length > 0 && byMasker);
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
      else picks[ctx.slot] = c.riot;
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
  // Two questions about the same board, asked independently: the role board
  // ("what does the model take for each open role") and the head ("what does
  // it take next"). A reply is only accepted if the board has not moved since
  // the request went out - rendering an older assignment against newer picks
  // is what made a fresh pick look like it was ignored.
  const rolesP = ROLE_MODEL
    ? postJSON('/api/suggest',
      { picks: picks, bans: allBansFlat(), first_pick: currentFirstPick(),
        fearless: fearlessPool(), series: seriesBody() })
    : null;
  const policyP = isPolicyModel()
    ? postJSON('/api/next-move', {
      picks: picks, bans: allBansFlat(), first_pick: currentFirstPick(),
      fearless: fearlessPool(), series: seriesBody(),
      patch_id: currentPatchId(),
      k: 8
    })
    : null;

  if (rolesP) {
    const m = await rolesP;
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
  }
  renderCompare();
}

function boardError(msg) {
  const el = document.getElementById('board-error');
  if (!el) return;
  el.textContent = msg ? 'board: ' + msg : '';
  el.classList.toggle('hidden', !msg);
}

// ---- next-move panel ----------------------------------------------------
// One call, one list: `predict_next_move` on the live state. This is the tool
// the agent calls, rendered instead of narrated - the head's ranking for the
// move on the clock, each champion with its share of the legal pool.
//
// There is no reranker any more. The local 4B model that used to re-order this
// list is gone: the agent's own model reads the list and decides, in the chat,
// where its reasoning is visible. A second ordering rendered beside the head's
// would be a second opinion with no mechanism behind it - and it could not say
// which of the two it was.

function renderNextMove() {
  const box = document.getElementById('nextmove');
  if (!isPolicyModel()) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  // Same freshness rule as the role board: an answer only describes the board
  // it was computed for. Showing the previous board's "Blue to pick" beside a
  // highlight that already moved on contradicts it, so blank the panel while
  // the new answer is in flight.
  const fresh = !!(presp && presp.state && prespSig === boardSig());
  const turn = document.getElementById('nm-turn');
  const meta = document.getElementById('nm-meta');
  const chips = document.getElementById('nm-chips');
  const warn = document.getElementById('nm-warning');
  if (!fresh) {
    turn.innerHTML = '<b>updating\u2026</b>';
    meta.textContent = '';
    chips.innerHTML = '';
    warn.classList.add('hidden');
    return;
  }
  const st = presp.state;
  const sideName = st.side === 0 ? 'Blue' : 'Red';
  turn.innerHTML = st.complete
    ? '<b>Draft complete</b>'
    : `<b>${sideName} to ${st.kind}</b>`;
  meta.textContent = st.complete ? ''
    : `move ${st.move_idx + 1}/20 \u00b7 ${st.n_legal} legal`
      + (presp.patch ? ` \u00b7 patch ${presp.patch}` : '');
  // The tool's own note, not a label of ours: it says which pool the numbers
  // are a share of, and whether the patch it asked for was the one it got.
  warn.textContent = presp.note || '';
  warn.classList.toggle('hidden', !presp.note);
  chips.innerHTML = '';
  if (st.complete) return;
  (presp.suggestions || []).forEach(c => {
    const b = document.createElement('button');
    b.className = 'nm-chip';
    b.innerHTML = imgTag(BY_IDX[c.idx])
      + `<span class="nm">${esc(c.name)}</span>`
      + `<span class="p">${(100 * c.prob).toFixed(1)}%</span>`;
    b.title = `${sideName} ${st.kind}: ${c.name}`;
    b.addEventListener('click', () => applyMove(st.kind, st.side, c.idx));
    chips.appendChild(b);
  });
}

function renderNextMoveError(msg) {
  const box = document.getElementById('nextmove');
  if (!isPolicyModel()) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  document.getElementById('nm-turn').innerHTML = '<b>no suggestion</b>';
  document.getElementById('nm-meta').textContent = '';
  document.getElementById('nm-chips').innerHTML = '';
  const warn = document.getElementById('nm-warning');
  warn.textContent = msg;
  warn.classList.remove('hidden');
}

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function boardSig() {
  // The patch is part of the answer, not of the board: switching it must
  // invalidate an in-flight reply exactly like a pick does.
  return JSON.stringify([picks, bans.blue, bans.red, currentFirstPick(),
                         SERIES ? fearlessPool() : [], currentPatchId()]);
}

// ---- walkthrough of real pro drafts -------------------------------------

function optLabel(d) {
  return `${(d.date || '').slice(0, 10)} · ${d.league} · ${d.blue} vs `
    + `${d.red} (G${d.game_number})` + (d.fearless ? ' · fearless' : '');
}

function renderCompare() {
  const box = document.getElementById('wt-compare');
  // Collapsed stays collapsed: loading a game must not reopen the section the
  // user hid. Its bar is the only way in.
  if (!WT || !walkthroughVisible()) { box.classList.add('hidden'); return; }
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
  if (WT) {
    WT.actualRole = {};
    const next = [0, 0];                 // next free pick slot per team
    for (let k = 0; k < WT.step; k++) {
      const m = WT.draft.moves[k];
      if (m.kind === 'ban') {
        bans[m.side === 0 ? 'blue' : 'red'].push(m.riot);
      } else if (m.side < 2 && next[m.side] < 5) {
        picks[m.side * 5 + next[m.side]] = m.riot;
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

function wtFilterValue() {
  // The filter box is pre-filled with WLDs in index.html, so the dropdown opens
  // on Worlds instead of the newest 100 games of every league. Reading it from
  // the DOM (rather than a second constant) keeps the box and the list agreeing
  // the moment the user types.
  const el = document.getElementById('wt-filter');
  return el ? el.value : '';
}

async function refreshDraftList(q) {
  const sel = document.getElementById('wt-select');
  const cur = sel.value;
  if (q == null) q = wtFilterValue();
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
  // A patch change re-asks both panels: the next move conditions on it, and
  // the role board's answers are computed against it.
  changed();
});
document.getElementById('picker-close').addEventListener('click', closePicker);
document.getElementById('picker').addEventListener('click', ev => {
  if (ev.target.id === 'picker') closePicker();
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') closePicker();
  if (!WT || !walkthroughVisible() || ev.target.tagName === 'INPUT' ||
      ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowRight') document.getElementById('wt-next').click();
  if (ev.key === 'ArrowLeft') document.getElementById('wt-prev').click();
});
document.getElementById('picker-search').addEventListener('input', ev =>
  renderPickerGrid(ev.target.value));

// Models are chosen at startup (--ckpt / --policy-ckpt / --no-policy). There
// is deliberately no browser control for them: the header only reports which
// two are loaded, and the patch selector is built from the live checkpoint's
// own vocabulary, so a swap mid-draft would invalidate every answer in flight.

// The "Real draft" walkthrough is collapsed behind one button. Hidden means
// hidden for both halves: the bar and the per-move comparison below it.
function walkthroughVisible() {
  return !document.getElementById('wt-bar').classList.contains('hidden');
}

function setWalkthrough(open) {
  const bar = document.getElementById('wt-bar');
  const btn = document.getElementById('btn-wt');
  bar.classList.toggle('hidden', !open);
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  btn.textContent = open ? 'Real draft ▴' : 'Real draft ▾';
  btn.classList.toggle('open', open);
  renderCompare();       // the comparison follows the bar, never pops out alone
}
document.getElementById('btn-wt').addEventListener('click', () =>
  setWalkthrough(!walkthroughVisible()));

// walkthrough of real pro drafts
document.getElementById('wt-select').addEventListener('change', ev => {
  if (ev.target.value !== '') wtLoad(+ev.target.value);
});
document.getElementById('wt-filter').addEventListener('input', ev => {
  clearTimeout(wtFilterT);
  wtFilterT = setTimeout(() => refreshDraftList(ev.target.value), 300);
});
document.getElementById('wt-random').addEventListener('click', async () => {
  // Same filter as the list: a 🎲 that dropped you into an LCP game while the
  // dropdown said WLDs would be a different corpus than the one on screen.
  let d;
  try {
    d = await (await fetch('/api/drafts?random=1&limit=1&q=' +
      encodeURIComponent(wtFilterValue() || ''))).json();
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

