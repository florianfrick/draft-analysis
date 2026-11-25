package com.draftassistant.data

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive

enum class MoveKind { BAN, PICK }

/** One step of the draft order. [side] is 0 for blue, 1 for red. */
data class DraftMove(val kind: MoveKind, val side: Int)

/** A resolved next move: the step plus its position in the 20-move plan. */
data class Turn(val kind: MoveKind, val side: Int, val index: Int)

/**
 * The board -> next-move function, ported from `webapp/static/draft-order.js`.
 *
 * Two things here are easy to get wrong, and both are deliberate in the
 * original:
 *
 *  1. The plan's second element is the SIDE, not a slot index. A pick's slot
 *     is not in the plan at all - the client assigns the first empty slot on
 *     that side.
 *  2. Position is found by CONSUMING counts (n bans, n picks) against the
 *     kinds in order, not by testing which slots look filled. A board that
 *     does not fit the order therefore reports the leftovers rather than
 *     silently reordering itself.
 */
object DraftOrder {

    const val PLAN_LEN = 20

    /** Parses `/api/champions`' `move_plan`. Malformed plans are dropped, not patched. */
    fun parse(plans: Map<String, List<List<JsonElement>>>): Map<Int, List<DraftMove>> {
        val out = mutableMapOf<Int, List<DraftMove>>()
        for ((key, rows) in plans) {
            val side = key.toIntOrNull() ?: continue
            if (side != 0 && side != 1) continue
            if (rows.size != PLAN_LEN) continue

            val moves = ArrayList<DraftMove>(PLAN_LEN)
            var ok = true
            for (row in rows) {
                if (row.size < 2) { ok = false; break }

                val kind = when (row[0].jsonPrimitive.contentOrNull) {
                    "ban" -> MoveKind.BAN
                    "pick" -> MoveKind.PICK
                    else -> { ok = false; break }
                }

                val s = row[1].jsonPrimitive.contentOrNull?.toIntOrNull()
                if (s != 0 && s != 1) { ok = false; break }

                moves.add(DraftMove(kind, s!!))
            }
            if (ok) out[side] = moves
        }
        return out
    }

    /** Mirrors `position()`: walk the plan, consuming one move of each kind. */
    fun position(plan: List<DraftMove>, nBans: Int, nPicks: Int): Int {
        var bans = nBans.coerceAtLeast(0)
        var picks = nPicks.coerceAtLeast(0)
        var k = 0
        while (k < plan.size) {
            if (plan[k].kind == MoveKind.BAN) {
                if (bans <= 0) break
                bans--
            } else {
                if (picks <= 0) break
                picks--
            }
            k++
        }
        return k
    }

    /** The move on the clock, or null when the plan is spent (draft complete). */
    fun nextTurn(plan: List<DraftMove>?, nBans: Int, nPicks: Int): Turn? {
        if (plan == null) return null
        val k = position(plan, nBans, nPicks)
        val move = plan.getOrNull(k) ?: return null
        return Turn(move.kind, move.side, k)
    }
}
