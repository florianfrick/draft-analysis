// The agent panel: one question about the board, answered by a model that
// holds the draft as an object.
//
// What changed, and why there is no toggle any more. The old chat had two
// answerers: a rule-based composer that read the message and a "tool" mode that
// handed the board to an LLM. Only one of them could reason, and only the other
// needed no key, so the panel asked the user to choose. The composer is gone.
// The agent reads the live state, calls the models and the corpus itself, and
// says what it found - so there is one answerer and one question.
//
// What it is NOT allowed to do: edit the board. It reads `real` and clones it
// into a sandbox to try a hypothetical. That is the difference between "the
// model likes X" and "X is now on the board", and the second one is the user's
// decision. The panel shows the sandboxes it left behind so an exploration is
// visible rather than implied.
//
// Every turn shows three things, because each answers a different question:
//   the answer   what the model and the corpus say, in sentences
//   the tools    what it actually called, with the arguments - the only audit
//                trail there is, since nothing here is a template
//   the mix      how many calls, and what share were SQL: above half means the
//                high-level tools are under-specified, not that SQL is good
//
// The header's "Suggest" button is a fourth thing: a fixed question to the
// same agent, run through /api/agent-suggest with SQL ablated and the explained
// prompt. It is the reranking arm - the model reads the held state, asks the
// head for a shortlist and commits to one champion - so its answer is shown as
// a move (parsed from the final PICK:/BAN: line) next to the head's own #1.
// "Explain" asks the Android app's composition-archetype question through the
// ordinary chat route; it is prose and a tool log, not a parsed move.
(function () {
  const el = id => document.getElementById(id);
  let busy = false;
  let turns = 0;
  // The Android panel's Explain button asks this exact question.
  const EXPLAIN_QUESTION =
    'What team composition archetype is each team aiming toward?';

  // ---- agent suggestions (the reranked top-k) ---------------------------
  // A second column under the head's own #nextmove list, filled by
  // /api/agent-rank. It reruns on every board change (debounced) and after any
  // chat turn, so it mirrors the head's list but with the agent's reordering.
  // `convoSeed` carries the recent chat into the rerank, so a conversation that
  // changes the agent's mind changes this list too. `lastSeen` drives the chat
  // line that records what the board did between reranks.
  let featuresOn = true;
  let autoTimer = null;
  let activeCalls = new Set();
  let convoSeed = [];        // [{role, content}] recent chat, for the rerank
  let lastSeen = null;       // {picks, bans, fearless} the last rank described
  let rankedOnce = false;    // so the first (initial) rerank also logs to chat

  function boardBody() {
    // The same board the panels are showing. The server publishes it to the
    // state store, so the agent answers about what is on screen rather than
    // about a transcription of it.
    return {
      picks: picks,
      bans: allBansFlat(),
      bans_blue: bans.blue,
      bans_red: bans.red,
      first_pick: currentFirstPick(),
      fearless: fearlessPool(),
      // Which game of the series this board belongs to. The pool above is its
      // effect on legality; this is the rest of the context, and the state
      // store renders it into the board the agent reads.
      series: (typeof seriesBody === 'function' ? seriesBody() : null),
      patch_id: currentPatchId(),
      patch: (typeof currentPatchKey === 'function' ? currentPatchKey() : null)
    };
  }

  // Every LLM route goes through here. Feature-off is a hard stop, and an
  // in-flight call is abortable so switching the agent off mid-turn really
  // stops the work rather than only hiding its result.
  function isSilentCancel(e) {
    return !featuresOn || (e && e.name === 'AbortError');
  }

  function agentFetch(url, body) {
    if (!featuresOn) return Promise.reject(new Error('agent features off'));
    const controller = new AbortController();
    activeCalls.add(controller);
    return fetch(apiUrl(url), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: controller.signal
    }).finally(() => activeCalls.delete(controller));
  }

  function abortAgentCalls() {
    activeCalls.forEach(c => c.abort());
    activeCalls.clear();
  }

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // Minimal markdown for chat text. Escapes first (XSS-safe), then inline code,
  // bold (**), italic (*), and newlines -> <br>. Underscore emphasis is
  // deliberately omitted so snake_case tokens (first_pick, patch_id) are not
  // mangled. Keep it tiny on purpose.
  function md(s) {
    let t = esc(s == null ? '' : s);
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
    t = t.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    t = t.replace(/\n/g, '<br>');
    return t;
  }

  function open() {
    el('agent').classList.remove('hidden');
    if (!turns) greet();
    el('agent-input').focus();
  }

  function greet() {
    const n = el('agent-meta');
    if (n) {
      n.textContent = 'an LLM drives the tools; it reads the live board and '
        + 'cannot edit it';
    }
    add('system', 'Ask about the board, ask what happens if a champion is '
      + 'taken, press "Suggest" for one recommended pick, or "Explain" for a '
      + 'composition read. The assistant can clone the board to try it, then '
      + 'answer from the model\u2019s ranking on the clone.');
  }

  function add(kind, text) {
    const d = document.createElement('div');
    d.className = 'chat-msg chat-' + kind;
    d.innerHTML = md(text);
    el('agent-log').appendChild(d);
    el('agent-log').scrollTop = el('agent-log').scrollHeight;
    return d;
  }

  // A "Board changed:" system line, tinted slightly differently from the plain
  // system lines so the board events stand out in the log.
  function addBoardChange(text) {
    const d = add('system', text);
    d.classList.add('board-change');
    return d;
  }

  function summariseArgs(args) {
    if (!args || !Object.keys(args).length) return 'no arguments';
    const text = Object.keys(args).map(k => {
      const v = args[k];
      if (Array.isArray(v)) return k + '=[' + v.length + ']';
      return k + '=' + (typeof v === 'object' ? JSON.stringify(v) : v);
    }).join(', ');
    return text.length > 90 ? text.slice(0, 90) + '\u2026' : text;
  }

  function render(res) {
    const d = add('model', '');
    let html = '';
    // A suggestion turn carries one parsed move. It is a claim about the board
    // on screen, so an illegal champion (a ban, a pick, a fearless one) is shown
    // but cannot be played, and neither can an answer that names the wrong kind
    // of move. Everything below - the prose, the tools, the sandboxes - is the
    // same for a free-form turn and for this one.
    const sug = res.suggestion || null;
    if (sug) {
      if (sug.champion) {
        const canPlay = sug.legal && sug.kind_matches_clock !== false
          && sug.idx != null && !!(res.state && !res.state.complete)
          && typeof applyMove === 'function';
        html += '<div class="chat-suggest' + (sug.legal ? '' : ' bad') + '">'
          + '<span class="cs-kind">' + esc((sug.kind || '?').toUpperCase())
          + '</span><b>' + esc(sug.champion) + '</b>'
          + (canPlay ? '<button class="cs-play" data-play="1" title="play this '
            + 'move on the board - it is the user\u2019s decision, not the '
            + 'agent\u2019s">play it</button>' : '')
          + (sug.note ? '<span class="cs-note">' + esc(sug.note) + '</span>' : '')
          + (sug.kind_matches_clock === false
            ? '<span class="cs-note">the answer names a different move kind '
              + 'than the clock</span>' : '')
          + '</div>';
      } else {
        html += '<div class="chat-ev-note">no champion could be read from the '
          + 'answer (' + esc(sug.parse || '?') + ')</div>';
      }
    }
    if (res.complete) {
      html += '<p class="muted">the draft is complete - there is no move to '
        + 'recommend</p>';
    }
    if (res.answer) {
      html += '<p>' + md(res.answer) + '</p>';
    } else {
      // No answer is an outcome, not a crash: "no client", "step budget",
      // "no state". The reason is the useful part, so it is not hidden behind
      // a generic failure.
      html += '<p class="muted">no answer \u2014 ' + esc(res.stop || 'unknown')
        + '</p>';
    }
    if (res.error) {
      html += '<div class="chat-ev-note">' + esc(res.error) + '</div>';
    }

    // The sandboxes a turn left behind. An exploration the user cannot see is
    // an answer they cannot check - the count now sits next to the tool
    // dropdown, not in a chip below the message.
    const extra = (res.states || []).filter(x => x.state !== 'real');
    const calls = res.calls || [];
    if (calls.length) {
      const rows = calls.map(c => '<div class="chat-ev-row"><b>'
        + esc(c.name) + '</b> <span class="muted">'
        + esc(summariseArgs(c.args)) + '</span>'
        + ' <span class="muted">' + esc(c.ms) + ' ms</span>'
        + (c.ok ? '' : ' <span class="chat-thin">refused</span>')
        + (c.error ? '<div class="chat-ev-note">' + esc(c.error) + '</div>' : '')
        + '</div>').join('');
      const sandboxNote = ' sandboxes (' + extra.length + ')';
      html += '<details class="chat-ev"><summary>tools (' + calls.length
        + ')' + sandboxNote + '</summary>' + rows + '</details>';
    }

    // The model that answered this turn is shown by the panel title (white, not
    // the green accent) rather than as a tag below the message.
    const modelEl = el('agent-model');
    if (modelEl) modelEl.textContent = res.model || '';
    d.innerHTML = html;
    // The agent never edits the board; this button is the user's decision. It
    // calls the same helper the head's own chips use, which refuses a champion
    // already on the board (an answer can be a moment old).
    const play = d.querySelector('[data-play]');
    if (play) {
      play.addEventListener('click', () => {
        if (typeof applyMove !== 'function') return;
        const s = res.state || {};
        applyMove(res.suggestion.kind, s.side === 'red' ? 1 : 0,
                  res.suggestion.idx);
        play.disabled = true;
        play.textContent = 'played';
      });
    }
    el('agent-log').scrollTop = el('agent-log').scrollHeight;
  }

  function setBusy(flag) {
    busy = flag;
    el('agent-send').disabled = flag;
    el('agent-suggest').disabled = flag;
    el('agent-explain').disabled = flag;
  }

  // ---- agent suggestions list --------------------------------------------
  // The reranked top-k, rendered as a column beside the head's own #nextmove
  // chips. Each chip is a champion with its confidence; `source` says whether
  // the number is the agent's own `(NN%)` or the head's `p_head` it fell
  // back to ("head"). The list is honest by construction - the endpoint only
  // emits champions the head listed.
  function renderRank(res) {
    const box = el('agentsuggest');
    const chips = el('as-chips');
    const meta = el('as-meta');
    const warn = el('as-warning');
    if (!box || !chips) return;
    if (res.complete) {
      box.classList.remove('hidden');
      meta.textContent = 'draft complete';
      chips.innerHTML = '';
      warn.classList.add('hidden');
      return;
    }
    const sugs = res.suggestions || [];
    if (res.error && !sugs.length) {
      // The agent could not answer; the head fallback should still be here, but
      // if even that failed, say so rather than leaving the column stale.
      meta.textContent = res.stop || 'no agent';
      warn.textContent = res.error;
      warn.classList.remove('hidden');
      return;
    }
    box.classList.remove('hidden');
    meta.textContent = res.model ? ('via ' + res.model)
      : (res.stop === 'no client' ? 'head fallback (no key)'
        : (res.stop || 'agent'));
    warn.classList.toggle('hidden', !res.note);
    warn.textContent = res.note || '';
    chips.innerHTML = '';
    sugs.forEach(s => {
      const b = document.createElement('button');
      b.className = 'nm-chip as-chip ' + (s.source === 'head' ? 'as-head' : 'as-model')
        + (s.legal ? '' : ' bad');
      const c = (typeof s.idx === 'number' && BY_IDX[s.idx]) ? BY_IDX[s.idx] : null;
      b.innerHTML = (c ? imgTag(c, 'ic') : '')
        + '<span class="nm">' + esc(s.name) + '</span>'
        + '<span class="p">' + (100 * s.conf).toFixed(0) + '%</span>';
      b.title = s.name + ' - '
        + (s.source === 'head' ? 'head p_head' : 'agent confidence') + ' '
        + (100 * s.conf).toFixed(1) + '%'
        + (s.legal ? '' : ' - on the board or in the fearless pool');
      if (s.legal && typeof applyMove === 'function' && res.state) {
        const kind = res.state.kind || 'pick';
        const side = res.state.side === 'red' ? 1 : 0;
        b.addEventListener('click', () => {
          if (typeof s.idx === 'number') applyMove(kind, side, s.idx);
        });
      }
      chips.appendChild(b);
    });
  }

  function pushSeed(role, content) {
    if (!content) return;
    convoSeed.push({ role: role, content: content });
    if (convoSeed.length > 8) convoSeed = convoSeed.slice(-8);
  }

  // One line per move, derived from the gap between this board and the last we
  // ranked. `lastSeen` is only advanced here, so several moves inside one
  // debounce window collapse into a single accurate description.
  function describeChange(cur) {
    if (!lastSeen) { lastSeen = cur; return null; }
    const notes = [];
    const oldP = new Set(lastSeen.picks.filter(Boolean));
    for (const r of cur.picks.filter(Boolean))
      if (!oldP.has(r)) notes.push('picked ' + pretty(r));
    const oldB = new Set(lastSeen.bans);
    for (const b of cur.bans) if (!oldB.has(b)) notes.push('banned ' + pretty(b));
    const oldF = new Set(lastSeen.fearless);
    for (const f of cur.fearless)
      if (!oldF.has(f)) notes.push('fearless: ' + pretty(f));
    lastSeen = cur;
    return notes.length ? notes.join('; ') : null;
  }

  function pretty(riot) {
    const c = (BY_RIOT && BY_RIOT[riot]) || (BY_IDX && BY_IDX[riot]);
    return c ? c.name : String(riot);
  }

  function logBoardChange(note, res) {
    addBoardChange('Board changed: ' + note);
    const sugs = res.suggestions || [];
    if (res.complete) {
      add('system', 'the draft is now complete');
    } else if (sugs.length) {
      const top = sugs.slice(0, 3)
        .map(s => s.name + ' ' + (100 * s.conf).toFixed(0) + '%').join(', ');
      add('system', 'agent ranks: ' + top
        + (sugs.length > 3 ? ' …' : ''));
    }
  }

  function scheduleAutoRank() {
    if (!featuresOn) return;
    clearTimeout(autoTimer);
    autoTimer = setTimeout(runAutoRank, 200);
  }

  async function runAutoRank() {
    if (!featuresOn) return;
    const cur = { picks: picks.slice(), bans: allBansFlat(),
                  fearless: fearlessPool() };
    const note = describeChange(cur);
    const initial = !rankedOnce;
    // A board change (or the first rank on load) is a visible event: show the
    // agent is working before the fetch returns, so the chat reflects that the
    // rerank is in flight rather than appearing all at once. A pure chip
    // refresh (no board change, no initial) needs no banner.
    const thinking = (note || initial)
      ? add('model pending', 'Agent is thinking\u2026') : null;
    try {
      const r = await agentFetch('/api/agent-rank',
        Object.assign(boardBody(), { messages: convoSeed.slice() }));
      const res = await r.json();
      if (thinking) thinking.remove();
      if (!r.ok || res.error) {
        renderRank(res);
        if (note) addBoardChange('Board changed: ' + note + ' (agent ranking '
          + (res.error || 'failed') + ')');
        return;
      }
      renderRank(res);
      // The first board the page loads (move 1) and every later board move get a
      // chat line - the initial rerank used to be silent. A chat-driven rerank
      // (no board change) is already represented by the turn above it, so it only
      // refreshes the chips, not the log.
      if (note || initial) {
        if (note) addBoardChange('Board changed: ' + note);
        else addBoardChange('Initial board - agent rerank');
        // The transcript: the LLM's answer (its chain of thought + the ranking
        // it committed to) and the tool calls it made to get there.
        render(res);
      }
      rankedOnce = true;
    } catch (e) {
      if (thinking) thinking.remove();
      if (isSilentCancel(e)) return;
      if (note) addBoardChange('Board changed: ' + note + ' (agent ranking error)');
    }
  }

  // Whether a question is about the draft (and so should rerank the
  // suggestions) or about the agent itself. A question like "which tools do you
  // have available" is about the assistant, not the board, and must not rerank -
  // the user explicitly wanted those to leave the column untouched.
  function boardIntent(text) {
    const t = ' ' + (text || '').toLowerCase() + ' ';
    const aboutAgent = [' tool', 'what can you do', 'what are you capable',
      'who are you', 'your name', 'your capabilities', 'how do you work',
      'how does this work', 'how are you', 'hello', ' hey ', ' hi '];
    if (aboutAgent.some(m => t.includes(m))) return false;
    // Default to true: this panel only knows the draft, so almost every real
    // question is about picks, bans, teams or matchups. A rerank that changes
    // nothing just refreshes the chips, which is harmless.
    return true;
  }

  async function send(text) {
    if (!featuresOn || busy) return;
    setBusy(true);
    add('user', text);
    const pending = add('model pending', 'thinking\u2026');
    try {
      const r = await agentFetch('/api/agent',
        Object.assign(boardBody(), { message: text }));
      const res = await r.json();
      pending.remove();
      if (!r.ok || res.error) {
        add('error', res.error || ('HTTP ' + r.status));
      } else {
        turns++;
        render(res);
        // The chat changed the agent's mind: carry it into the rerank so the
        // suggestions column reflects the conversation, not the last board move.
        pushSeed('user', text);
        pushSeed('assistant', res.answer || '');
        // Only rerank when the question was about the draft. A question about the
        // agent itself (its tools, its limits) changes nothing on the board, so
        // rerunning the rank is noise and must leave the suggestions alone.
        if (boardIntent(text)) scheduleAutoRank();
      }
    } catch (e) {
      pending.remove();
      if (isSilentCancel(e)) return;
      add('error', String(e));
    } finally {
      setBusy(false);
      if (featuresOn) el('agent-input').focus();
    }
  }

  async function suggest() {
    if (!featuresOn || busy) return;
    setBusy(true);
    add('user', 'Suggest and justify the optimal next move (Blue/Red to Pick/Ban).');
    const pending = add('model pending',
      'reading the state and the head\u2019s ranking\u2026');
    try {
      const r = await agentFetch('/api/agent-suggest', boardBody());
      const res = await r.json();
      pending.remove();
      if (!r.ok) {
        add('error', res.error || ('HTTP ' + r.status));
      } else {
        turns++;
        // A missing key is an outcome, not a crash: the answer is empty but
        // the state, the head's #1 and the "no client" reason still say what
        // happened. render() shows that reason rather than hiding it.
        render(res);
        pushSeed('user', 'Suggest and justify the optimal next move (Blue/Red to Pick/Ban).');
        pushSeed('assistant', res.answer || '');
        scheduleAutoRank();
      }
    } catch (e) {
      pending.remove();
      if (isSilentCancel(e)) return;
      add('error', String(e));
    } finally {
      setBusy(false);
      if (featuresOn) el('agent-input').focus();
    }
  }

  function explain() {
    if (!featuresOn || busy) return;
    open();
    send(EXPLAIN_QUESTION);
  }

  function init() {
    if (!el('agent')) return;
    el('agent-suggest').addEventListener('click', suggest);
    el('agent-explain').addEventListener('click', explain);
    // The chat form submits natively (the "ask" button is type=submit), which
    // would reload the whole page. Intercept it so it runs the chat turn (the
    // "thinking" pending line + the transcript) in place.
    const form = el('agent-form');
    if (form) {
      form.addEventListener('submit', e => {
        e.preventDefault();
        const input = el('agent-input');
        const text = (input.value || '').trim();
        if (!text || busy) return;
        input.value = '';
        send(text);
      });
    }
    el('agent-close').addEventListener('click',
      () => el('agent').classList.add('hidden'));
    el('agent-clear').addEventListener('click', () => {
      el('agent-log').innerHTML = '';
      turns = 0;
      greet();
      // Clear chat = reset the agent to its fresh top-k on this board: drop the
      // conversation context that was seeding the rerank and rerank from the
      // head only.
      convoSeed = [];
      lastSeen = null;
      scheduleAutoRank();
    });
    // The Agent toggle: off hides both the chat and the rerank column
    // and stops the agent from running at all (only the base model's own
    // suggestions and board updates remain).
    const feat = el('agent-features');
    if (feat) {
      feat.checked = featuresOn;
      feat.addEventListener('change', e => setEnabled(e.target.checked));
    }
    // Deep link: #agent opens the panel on load, so a screenshot or a bookmark
    // lands on the same place instead of on the board.
    if (location.hash === '#agent') open();
    // Seed the rerank column on load so it is not empty beside the head's list.
    if (featuresOn) scheduleAutoRank();
  }

  function setEnabled(on) {
    featuresOn = on;
    if (el('agent')) el('agent').classList.toggle('hidden', !on);
    if (el('agentsuggest')) el('agentsuggest').classList.toggle('hidden', !on);
    if (on) {
      scheduleAutoRank();
    } else {
      // Stop the timer and anything already in flight. `featuresOn` is false
      // first, so the abandoned catch blocks stay silent.
      clearTimeout(autoTimer);
      abortAgentCalls();
      setBusy(false);
    }
  }

  // Exposed so app.js's board-change hook (changed()) can drive an auto rerank
  // without knowing this module's internals.
  window.AgentPanel = { scheduleAutoRank, setEnabled, renderRank };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
