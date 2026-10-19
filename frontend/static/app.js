const ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];
let CHAMPIONS = [], BY_IDX = {}, BY_RIOT = {};
let picks = Array(10).fill(null);          // slot -> riot id | null (pick order)
let bans = { blue: [], red: [] };          // team -> [riot id]
let resp = null;                           // last /api/suggest payload
let pickerCtx = null;                      // {type, slot? , team?}
let debounceT = null, wtFilterT = null;
let CURRENT_MODEL = null;                  // last model info from the server
let WT = null;                             // walkthrough: {draft, step, ...}

async function init() {
  const d = await (await fetch('/api/champions')).json();
  CHAMPIONS = d.champions;
  CHAMPIONS.forEach(c => { BY_IDX[c.idx] = c; BY_RIOT[c.riot] = c; });
  setModelInfo(d.model);
  refreshCheckpoints();
  refreshDraftList();
  buildStrips('blue');
  buildStrips('red');
  renderAll();
  renderWalkthrough();
  requestSuggest();
}

function imgTag(c, cls) {
  if (!c) return '';
  return `<img class="${cls || 'ic'}" src="${c.img}" alt="" title="${c.name}"`
    + ' onerror="this.classList.add(\'broken\')">';
}

function setModelInfo(m) {
  CURRENT_MODEL = m;
  renderModelInfo();
}

function renderModelInfo() {
  const m = CURRENT_MODEL;
  if (!m) return;
  let txt = `model ${m.ckpt} · val top-5 ` +
    (m.val_top5 >= 0 ? `${(100 * m.val_top5).toFixed(1)}%` : 'n/a');
  if (m.kind === 'finetune')
    txt += ` · finetuned${m.stage != null ? ' (stage ' + m.stage + ')' : ''}`;
  else if (m.kind) txt += ` · ${m.kind}`;
  if (m.epoch != null) txt += ` · ep ${m.epoch}`;
  document.getElementById('model-info').textContent = txt;
}

async function refreshCheckpoints() {
  let d;
  try { d = await (await fetch('/api/checkpoints')).json(); }
  catch (e) { return; }
  const sel = document.getElementById('ckpt-select');
  sel.innerHTML = '';
  (d.checkpoints || []).forEach(c => {
    const o = document.createElement('option');
    o.value = c.path;
    o.textContent = c.label;
    sel.appendChild(o);
  });
  if (d.current && ![...sel.options].some(o => o.value === d.current)) {
    const o = document.createElement('option');   // current ckpt outside globs
    o.value = d.current;
    o.textContent = d.current;
    sel.appendChild(o);
  }
  if (d.current) sel.value = d.current;
}

async function switchModel(url, opts) {
  const ids = ['btn-load-ckpt', 'btn-upload', 'btn-reset'];
  ids.forEach(id => { document.getElementById(id).disabled = true; });
  document.getElementById('model-info').textContent = 'loading model…';
  let ok = false, msg = '';
  try {
    const r = await fetch(url, opts);
    const d = await r.json().catch(() => ({}));
    if (r.ok && d.model) { ok = true; setModelInfo(d.model); }
    else msg = d.error || ('HTTP ' + r.status);
  } catch (e) { msg = String(e); }
  ids.forEach(id => { document.getElementById(id).disabled = false; });
  if (!ok) {
    alert('model load failed: ' + msg);
    renderModelInfo();
    return;
  }
  resp = null;
  await refreshCheckpoints();
  changed();   // re-run suggestions with the new weights
}

function nextTurn(firstPick) {
  // Standard pro draft order (root README): bans1 B,R,B,R,B,R -
  // picks1 B,R,R,B,B,R - bans2 R,B,R,B - picks2 R,B,B,R.
  // Mirrored wholesale when red has first pick. Highlight only, never
  // enforced - clicks on any slot always work.
  const nb = [bans.blue.length, bans.red.length];
  const np = [0, 1].map(t => picks.slice(t * 5, t * 5 + 5)
    .filter(Boolean).length);
  const M = s => (firstPick === 1 ? 1 - s : s);
  if (nb[0] < 3 || nb[1] < 3)
    return { kind: 'ban', side: M((nb[0] + nb[1]) % 2) };
  if (np[0] < 3 || np[1] < 3)
    return { kind: 'pick', side: M([0, 1, 1, 0, 0, 1][np[0] + np[1]]) };
  if (nb[0] < 5 || nb[1] < 5)
    return { kind: 'ban', side: M([1, 0, 1, 0][(nb[0] - 3) + (nb[1] - 3)]) };
  if (np[0] < 5 || np[1] < 5)
    return { kind: 'pick', side: M([1, 0, 0, 1][(np[0] - 3) + (np[1] - 3)]) };
  return null;
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
  const t = nextTurn(WT ? WT.draft.first_pick : 0);
  ['blue', 'red'].forEach(name => {
    document.getElementById('turn-' + name).textContent = '';
  });
  document.querySelectorAll('.turn-target')
    .forEach(el => el.classList.remove('turn-target'));
  if (!t) return;
  const side = t.side === 0 ? 'blue' : 'red';
  document.getElementById('turn-' + side).textContent =
    t.kind === 'ban' ? '▶ to ban' : '▶ to pick';
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
    const rp = resp && resp.roles ? resp.roles[s] : null;
    if (c && rp && rp.length) {
      rb.textContent = ROLES[rp[0].role] + ' ' +
        Math.round(100 * rp[0].prob) + '%';
      const proRole = WT && WT.actualRole ? WT.actualRole[riot] : undefined;
      rb.title = rp.slice(0, 3).map(x =>
        `${ROLES[x.role]} ${(100 * x.prob).toFixed(0)}%`).join(' · ')
        + (proRole != null ? ` · pro played ${ROLES[proRole]}` : '');
      if (proRole != null && proRole !== rp[0].role)
        rb.classList.add('mismatch');           // model vs pro disagreement
      else if (rp[1] && rp[1].prob >= 0.15)
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
  ['blue', 'red'].forEach(team => {
    const box = document.getElementById('roles-' + team);
    box.innerHTML = '';
    let any = false;
    for (let s = 0; s < 5; s++) {
      const slot = team === 'blue' ? s : 5 + s;
      const c = picks[slot] != null ? BY_RIOT[picks[slot]] : null;
      if (!c) continue;
      any = true;
      const rp = (resp && resp.roles && resp.roles[slot]) || [];
      const el = document.createElement('div');
      el.className = 'role-ass';
      let html = imgTag(c) + `<span class="nm">${c.name}</span>`;
      if (rp.length) {
        html += `<span class="rb hi">${ROLES[rp[0].role]}`
          + ` ${(100 * rp[0].prob).toFixed(0)}%</span>`;
        if (rp[1] && rp[1].prob >= 0.15)
          html += `<span class="rb alt">${ROLES[rp[1].role]}`
            + ` ${(100 * rp[1].prob).toFixed(0)}%</span>`;
        el.title = rp.slice(0, 3).map(x =>
          `${ROLES[x.role]} ${(100 * x.prob).toFixed(1)}%`).join(' · ');
      } else {
        html += '<span class="rb muted-r">…</span>';
      }
      el.innerHTML = html;
      box.appendChild(el);
    }
    if (!any) box.innerHTML = '<span class="muted">no picks yet</span>';
  });
}

function renderSuggestions() {
  ['blue', 'red'].forEach(team => {
    const box = document.getElementById('sugs-' + team);
    box.innerHTML = '';
    ROLES.forEach(role => {
      const row = document.createElement('div');
      row.className = 'sug-row';
      const list = (resp && resp[team] && resp[team][role]) || [];
      let html = `<span class="rt">${role}</span><div class="chips">`;
      if (!list.length) html += '<span class="muted">—</span>';
      list.forEach(cand => {
        html += `<button class="sug${cand.flex ? ' flex' : ''}"`
          + ` data-team="${team}" data-idx="${cand.idx}">`
          + imgTag(BY_IDX[cand.idx])
          + `<span class="nm">${cand.name}</span>`
          + `<span class="p">${(100 * cand.prob).toFixed(1)}%</span>`
          + (cand.flex ? '<span class="fb">flex</span>' : '')
          + '</button>';
      });
      row.innerHTML = html + '</div>';
      box.appendChild(row);
    });
    box.querySelectorAll('.sug').forEach(b =>
      b.addEventListener('click', () =>
        addPick(b.dataset.team, +b.dataset.idx)));
  });
}

function addPick(team, idx) {
  const base = team === 'blue' ? 0 : 5;
  for (let s = base; s < base + 5; s++) {
    if (picks[s] == null) { picks[s] = BY_IDX[idx].riot; break; }
  }
  changed();
}

function renderAll() {
  renderSlots(); renderBans(); renderRoles();
  renderSuggestions(); renderTurn();
}

function changed() {
  renderAll();
  renderCompare();
  requestSuggest();
}

function allBansFlat() { return bans.blue.concat(bans.red); }

function excludedSet() {
  const ex = new Set(picks.filter(Boolean));
  allBansFlat().forEach(b => ex.add(b));
  if (pickerCtx && pickerCtx.type === 'pick' && picks[pickerCtx.slot])
    ex.delete(picks[pickerCtx.slot]);   // current pick stays selectable
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
      if (pickerCtx.type === 'ban') bans[pickerCtx.team].push(c.riot);
      else picks[pickerCtx.slot] = c.riot;
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
  debounceT = setTimeout(async () => {
    let r;
    try {
      r = await (await fetch('/api/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ picks: picks, bans: allBansFlat() })
      })).json();
    } catch (e) { return; }
    if (r.error) { console.warn(r.error); return; }
    resp = r;
    renderSlots(); renderRoles(); renderSuggestions(); renderTurn();
    renderCompare();
  }, 200);
}

// ---- walkthrough of real pro drafts -------------------------------------

function optLabel(d) {
  return `${(d.date || '').slice(0, 10)} · ${d.league} · ${d.blue} vs `
    + `${d.red} (G${d.game_number})`;
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
      + ' — bans are a post-filter; they never change what the model'
      + ' conditions on.';
    return;
  }
  const role = ROLES[m.role];
  const lists = resp && resp[sideName] ? resp[sideName] : {};
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
        picks[m.side * 5 + next[m.side]++] = m.riot;
        WT.actualRole[m.riot] = m.role;
      }
    }
  }
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
  renderWalkthrough();
  wtApply();   // changed() -> suggest -> renderCompare on response
  renderCompare();
}

function wtExit() {
  WT = null;
  document.getElementById('wt-select').value = '';
  resetDraft();
}

function resetDraft() {
  picks = Array(10).fill(null);
  bans = { blue: [], red: [] };
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
  info.innerHTML = `<b>${d.date.slice(0, 10)}</b> ${d.league} · `
    + `${d.blue} vs ${d.red} · G${d.game_number} · patch ${d.patch} · `
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

// model selection: server-side list + upload a file from this machine
document.getElementById('ckpt-select').addEventListener('focus',
  refreshCheckpoints);   // pick up newly trained runs without a restart
document.getElementById('btn-load-ckpt').addEventListener('click', () => {
  const path = document.getElementById('ckpt-select').value;
  if (!path) return;
  switchModel('/api/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: path })
  });
});
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

