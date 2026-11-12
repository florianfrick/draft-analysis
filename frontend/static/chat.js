// Chat panel: a message about the board, answered from the model's own numbers.
//
// The panel is deliberately read-only with respect to the draft. It reports a
// ranking and the evidence behind it; it never places a pick. That is the
// difference between "the model suggested X" and "X is now on the board", and
// the second one is the user's decision, not the chat's.
//
// Two answerers, one box. The default is `head`: the server computes the
// numbers and a template (or a remote model asked to phrase, never to decide)
// says which of them matters. `agent` hands the same board to an LLM that can
// call the tools itself - it reads the live state, clones it to test a
// hypothetical, and writes the sentence. The agent costs a key and a round trip
// per step, which is why it is a toggle and not the default.
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
  let mode = 'head';          // 'head' -> /api/chat, 'agent' -> /api/agent

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
    if (n) {
      n.textContent = mode === 'agent'
        ? 'an LLM drives the tools; it reads the live board and cannot edit it'
        : 'answers from the next-move head; nothing here edits the board';
    }
    add('system', mode === 'agent'
      ? 'Ask about the board, or ask what happens if a champion is taken. The '
        + 'assistant can clone the board to try it, then answer from the head\u2019s '
        + 'ranking on the clone.'
      : 'Name a champion and say how strongly you rate it. The reply '
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

  // ---- the agent's turn: an answer, plus the tool calls it is built on ----
  function summariseArgs(args) {
    if (!args || !Object.keys(args).length) return 'no arguments';
    const text = Object.keys(args).map(k => {
      const v = args[k];
      if (Array.isArray(v)) return k + '=[' + v.length + ']';
      return k + '=' + (typeof v === 'object' ? JSON.stringify(v) : v);
    }).join(', ');
    return text.length > 90 ? text.slice(0, 90) + '\u2026' : text;
  }

  function renderAgent(res) {
    const d = add('model', '');
    let html = '';
    if (res.answer) {
      html += '<p>' + esc(res.answer) + '</p>';
    } else {
      html += '<p class="muted">no answer \u2014 ' + esc(res.stop || 'unknown')
        + '</p>';
    }
    if (res.error) {
      html += '<div class="chat-ev-note">' + esc(res.error) + '</div>';
    }
    const calls = res.calls || [];
    if (calls.length) {
      const rows = calls.map(c => '<div class="chat-ev-row"><b>'
        + esc(c.name) + '</b> <span class="muted">'
        + esc(summariseArgs(c.args)) + '</span>'
        + (c.ok ? '' : ' <span class="chat-thin">refused</span>')
        + '</div>').join('');
      html += '<details class="chat-ev"><summary>tools (' + calls.length
        + ')</summary>' + rows + '</details>';
    }
    const s = res.summary || {};
    const facts = ['<span class="chat-chip">' + esc(res.model || 'no model')
      + '</span>',
      '<span class="chat-chip muted">stop: ' + esc(res.stop || '?') + '</span>'];
    if (s.calls != null) {
      facts.push('<span class="chat-chip muted">' + s.calls + ' calls</span>');
    }
    if (s.sql_share != null) {
      // The number the two-tier design is judged on: mostly SQL means the
      // high-level tools are under-specified, not that SQL is too good.
      facts.push('<span class="chat-chip muted">sql '
        + Math.round(100 * s.sql_share) + ' %</span>');
    }
    html += '<div class="chat-facts">' + facts.join('') + '</div>';
    d.innerHTML = html;
    el('chat-log').scrollTop = el('chat-log').scrollHeight;
  }

  async function send(text) {
    if (busy) return;
    busy = true;
    el('chat-send').disabled = true;
    add('user', text);
    const pending = add('model pending', 'thinking\u2026');
    // Taken before the request, not after: the reply describes the board that
    // was sent. If a pick lands while it is in flight the signature no longer
    // matches and the panel's row stays hidden, instead of showing numbers
    // computed for the previous move under the new one.
    const sig = chatSig();
    try {
      const body = Object.assign(boardBody(), { message: text });
      const agent = mode === 'agent';
      const r = await fetch(agent ? '/api/agent' : '/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const res = await r.json();
      pending.remove();
      if (!r.ok || res.error) { add('error', res.error || ('HTTP ' + r.status)); }
      else if (agent) {
        // No CHAT_ROW: the agent's answer is prose about a state, and the row
        // under the header compares the *head's* ranking with the chat's. An
        // agent turn has neither a single ranking nor a lambda to publish.
        turns++;
        renderAgent(res);
      } else {
        turns++;
        render(res);
        // The same ranking, published to the next-move panel: the chat log is
        // where the reasoning is, the row under the header is where the numbers
        // get compared with the head's own.
        CHAT_ROW = { sig: sig, res: res, message: text };
        renderChatRow();
      }
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
      // The row in the panel is a record of a turn, and its provenance is the
      // message it quotes. Forget the turns and the record has to go with them.
      CHAT_ROW = null;
      renderChatRow();
      greet();
    });
    el('chat-mode').addEventListener('click', () => {
      mode = mode === 'agent' ? 'head' : 'agent';
      el('chat-mode').textContent = mode;
      // A turn in the other mode answered a different question, so the log
      // cannot stay: "head" and "agent" disagreeing on screen with no label
      // between them reads as a model changing its mind.
      el('chat-log').innerHTML = '';
      turns = 0;
      CHAT_ROW = null;
      renderChatRow();
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
