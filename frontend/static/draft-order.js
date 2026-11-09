'use strict';
/* The canonical 20-move draft order, as a pure board -> next-move function.
 *
 * The plan itself is NOT duplicated here. GET /api/champions publishes
 * `move_plan` for both first-pick sides, built from
 * solo_queue/model/policy/infer.py:move_plan (which is checked against
 * data/pro play/pipeline.py:interleaved_moves). The client keeps one source
 * of truth and computes the next turn synchronously from the board.
 *
 * Why synchronous matters: /api/next-move is an async model call. Its
 * `state` describes the board the request was sent for, so an answer that
 * is still in flight belongs to a *previous* board. Reading it for the
 * green highlight made the highlight lag one move behind - blue banning
 * twice at the start, and a fourth ban slot in ban phase 1 after all six
 * bans. The turn is a property of the board, not of a model answer.
 *
 * `position` mirrors infer.resolve_position: walk the plan from the front,
 * consuming one ban or pick for each move of that kind, and stop at the
 * first move whose kind has been used up. Boards that do not fit the order
 * report the leftovers instead of being silently reordered.
 */
(function (global) {
  const BAN = 'ban';
  const PLAN_LEN = 20;

  function valid(plan) {
    return Array.isArray(plan) && plan.length === PLAN_LEN
      && plan.every(m => Array.isArray(m) && m.length === 2
        && (m[0] === BAN || m[0] === 'pick')
        && (m[1] === 0 || m[1] === 1));
  }

  function position(plan, nBans, nPicks) {
    let bans = Math.max(0, nBans | 0), picks = Math.max(0, nPicks | 0);
    let k = 0;
    for (; k < plan.length; k++) {
      const kind = plan[k][0];
      if (kind === BAN) {
        if (bans <= 0) break;
        bans -= 1;
      } else {
        if (picks <= 0) break;
        picks -= 1;
      }
    }
    return { move_idx: k, leftover_bans: bans, leftover_picks: picks };
  }

  /* plans: {"0": [[kind, side] x20], "1": [...]} from /api/champions.
   * Returns {kind, side, move_idx, warning} or null (no plan / complete). */
  function nextTurn(plans, firstPick, nBans, nPicks) {
    const plan = plans && plans[firstPick ? '1' : '0'];
    if (!valid(plan)) return null;
    const pos = position(plan, nBans, nPicks);
    if (pos.move_idx >= plan.length) return null;
    const left = pos.leftover_bans + pos.leftover_picks;
    return {
      kind: plan[pos.move_idx][0],
      side: plan[pos.move_idx][1],
      move_idx: pos.move_idx,
      warning: left ? left + ' move(s) do not fit the draft order' : null
    };
  }

  const api = { position, nextTurn, valid };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.DraftOrder = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
