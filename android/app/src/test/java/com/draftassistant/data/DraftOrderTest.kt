package com.draftassistant.data

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Guards the port of `webapp/static/draft-order.js` against
 * `solo_queue/model/policy/infer.py:move_plan`.
 *
 * The plan is rebuilt here from the same four sequences the server uses, so if
 * the canonical order ever changes, this test keeps asserting the *old* shape
 * and fails - which is the point: the client must not silently keep a stale
 * copy of the draft order.
 */
class DraftOrderTest {

    /** Mirrors infer.move_plan: 6 bans, 6 picks, 4 bans, 4 picks. */
    private fun planJson(firstPick: Int): String {
        val fp = firstPick
        val sp = 1 - fp
        val seq = mutableListOf<Pair<String, Int>>()
        listOf(fp, sp, fp, sp, fp, sp).forEach { seq += "ban" to it }
        listOf(fp, sp, sp, fp, fp, sp).forEach { seq += "pick" to it }
        listOf(sp, fp, sp, fp).forEach { seq += "ban" to it }
        listOf(sp, fp, fp, sp).forEach { seq += "pick" to it }
        assertEquals("plan length", 20, seq.size)

        val rows = seq.joinToString(",") { (kind, side) -> "[\"$kind\",$side]" }
        return """{"$firstPick":[$rows]}"""
    }

    private fun parse(firstPick: Int): List<DraftMove> {
        val map: Map<String, List<List<JsonElement>>> =
            Json.parseToJsonElement(planJson(firstPick)).jsonObject
                .mapValues { (_, value) -> value.jsonArray.map { row -> row.jsonArray.toList() } }
        return DraftOrder.parse(map).getValue(firstPick)
    }

    @Test
    fun `plan parses to twenty moves for both first-pick sides`() {
        assertEquals(20, parse(0).size)
        assertEquals(20, parse(1).size)
    }

    @Test
    fun `first-pick side mirrors the order rather than reordering it`() {
        val blue = parse(0)
        val red = parse(1)
        for (i in blue.indices) {
            assertEquals("kind at $i", blue[i].kind, red[i].kind)
            assertEquals("side at $i", 1 - blue[i].side, red[i].side)
        }
    }

    @Test
    fun `turn walks ban phase then pick phase then the second ban phase`() {
        val plan = parse(0)

        assertEquals(Turn(MoveKind.BAN, 0, 0), DraftOrder.nextTurn(plan, 0, 0))
        // Six bans consumed -> pick phase 1 opens with the first-pick side.
        assertEquals(Turn(MoveKind.PICK, 0, 6), DraftOrder.nextTurn(plan, 6, 0))
        // Six bans and six picks -> ban phase 2 opens with the *other* side.
        assertEquals(Turn(MoveKind.BAN, 1, 12), DraftOrder.nextTurn(plan, 6, 6))
        // All ten bans and six picks -> the last pick block.
        assertEquals(Turn(MoveKind.PICK, 1, 16), DraftOrder.nextTurn(plan, 10, 6))
        assertNull(DraftOrder.nextTurn(plan, 10, 10))
    }

    @Test
    fun `red first pick opens with red banning`() {
        val plan = parse(1)
        assertEquals(Turn(MoveKind.BAN, 1, 0), DraftOrder.nextTurn(plan, 0, 0))
        assertEquals(Turn(MoveKind.PICK, 1, 6), DraftOrder.nextTurn(plan, 6, 0))
    }

    @Test
    fun `a pick before any ban does not advance the position`() {
        // One pick and no bans cannot be laid onto the order from the front.
        // The reference client reports the leftover instead of reordering, so
        // the board still sits on move 0 - not on the pick phase.
        val plan = parse(0)
        assertEquals(Turn(MoveKind.BAN, 0, 0), DraftOrder.nextTurn(plan, 0, 1))
    }

    @Test
    fun `a malformed plan is dropped rather than half-parsed`() {
        val map: Map<String, List<List<JsonElement>>> =
            Json.parseToJsonElement("""{"0":[["ban",0],["pick",7]]}""").jsonObject
                .mapValues { (_, value) -> value.jsonArray.map { row -> row.jsonArray.toList() } }
        assertEquals(0, DraftOrder.parse(map).size)
    }
}
