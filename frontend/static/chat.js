// Chat panel: a message about the board, answered from the model's own numbers.
//
// The panel is deliberately read-only with respect to the draft. It reports a
// ranking and the evidence behind it; it never places a pick. That is the
// difference between "the model suggested X" and "X is now on the board", and
// the second one is the user's decision, not the chat's.
//
// Every turn shows three things, because each answers a different question:
//   the reply   what the model and the corpus say, in sentences
//   the delta   where the named champion moved, and what it would have taken to
//               move it further - this is the part that makes "incredibly good"
//               and "alright" visibly different instead of a claim
//   evidence    the tool results with their n, collapsible
(function () {
  const el = id => document.getElementById(id);
  let busy = false;
  let turns = 0;

  function boardBody() {
    return {
      picks: picks,
      bans: allBansFlat(),
      first_pick: currentFirstPick(),
      fearless: fearlessPool(),
      patch_id: currentPatchId()
    };
  }

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function pct(x) {
    return x == null ? 'n/a' : (100 * x).toFixed(2) + ' %';
  }

  function open() {
    el('chat').classList.remove('hidden');
    if (!turns) greet();
    el('chat-input').focus();
  }

  function greet() {
    const n = el('chat-meta');
    if (n) n.textContent = 'answers from the next-move head; nothing here edits the board';
    add('system', 'Name a champion and say how strongly you rate it. The reply '
      + 'shows the head\u2019s rank, where your wording moves it, and the '
      + 'nearest boards that agree or disagree.');
  }

  function add(kind, text) {
    const d = document.createElement('div');
    d.className = 'chat-msg chat-' + kind;
    d.textContent = text;
    el('chat-log').appendChild(d);
    el('chat-log').scrollTop = el('chat-log').scrollHeight;
    return d;
  }

  function evidenceHtml(ev) {
    if (!ev || !Object.keys(ev).length) return '';
    const rows = Object.keys(ev).map(k => {
      const t = ev[k];
      const flag = t.reliable ? '' : ' <span class="chat-thin">thin</span>';
      return '<div class="chat-ev-row"><b>' + esc(k) + '</b>'
        + ' <span class="muted">n=' + esc(t.n) + '</span>' + flag
        + (t.note ? '<div class="chat-ev-note">' + esc(t.note) + '</div>' : '')
        + '</div>';
    }).join('');
    return '<details class="chat-ev"><summary>evidence (' + rows.length
      + ')</summary>' + rows + '</details>';
  }

  function render(res) {
    const d = add('model', '');
    let html = '<p>' + esc(res.reply) + '</p>';

    const m = res.moved;
    if (m && m.before_rank != null) {
      const moved = m.after_rank !== m.before_rank;
      html += '<div class="chat-facts">'
        + '<span class="chat-chip' + (m.after_rank === 1 ? ' win' : '') + '">'
        + esc(m.name) + ' ' + m.before_rank + ' \u2192 ' + m.after_rank
        + ' of ' + (res.board ? res.board.n_legal : '?')
        + '</span>'
        + '<span class="chat-chip">\u03bb ' + (res.lambda > 0 ? '+' : '')
        + res.lambda + ' nats</span>'
        + (m.lam_needed != null && m.after_rank !== 1
          ? '<span class="chat-chip muted">needs ' + m.lam_needed.toFixed(1)
            + '</span>' : '')
        + (moved ? '' : '<span class="chat-chip muted">no change</span>')
        + '</div>';
    }
    if (res.pick) {
      const c = BY_IDX[res.pick.idx];
      html += '<div class="chat-pick">'
        + (c ? '<img class="ic" src="' + esc(c.img) + '" alt="">' : '')
        + '<span>' + esc(res.pick.name) + '</span>'
        + '<span class="muted">' + pct(res.pick.prob) + '</span></div>';
    }
    html += evidenceHtml(res.evidence);
    d.innerHTML = html;
    el('chat-log').scrollTop = el('chat-log').scrollHeight;
  }

  async function send(text) {
    if (busy) return;
    busy = true;
    el('chat-send').disabled = true;
    add('user', text);
    const pending = add('model pending', 'thinking\u2026');
    try {
      const body = Object.assign(boardBody(), { message: text });
      const r = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const res = await r.json();
      pending.remove();
      if (!r.ok || res.error) { add('error', res.error || ('HTTP ' + r.status)); }
      else { turns++; render(res); }
    } catch (e) {
      pending.remove();
      add('error', String(e));
    } finally {
      busy = false;
      el('chat-send').disabled = false;
      el('chat-input').focus();
    }
  }

  function init() {
    if (!el('chat')) return;
    el('nm-chat').addEventListener('click', open);
    el('chat-close').addEventListener('click',
      () => el('chat').classList.add('hidden'));
    el('chat-clear').addEventListener('click', () => {
      el('chat-log').innerHTML = '';
      turns = 0;
      greet();
    });
    el('chat-form').addEventListener('submit', ev => {
      ev.preventDefault();
      const box = el('chat-input');
      const text = box.value.trim();
      if (!text) return;
      box.value = '';
      send(text);
    });
    // Deep link: #chat opens the panel on load, so a screenshot or a bookmark
    // lands on the same place instead of on the board.
    if (location.hash === '#chat') open();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
