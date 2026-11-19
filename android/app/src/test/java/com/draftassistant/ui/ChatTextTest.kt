package com.draftassistant.ui

import com.draftassistant.data.Champion
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatTextTest {

    private val aatrox = Champion(idx = 0, riot = 1, name = "Aatrox")
    private val vi = Champion(idx = 1, riot = 2, name = "Vi")
    private val viktor = Champion(idx = 2, riot = 3, name = "Viktor")

    private fun board(picks: List<Int?> = List(10) { null }) = UiState(
        champions = listOf(aatrox, vi, viktor),
        byRiot = mapOf(1 to aatrox, 2 to vi, 3 to viktor),
        picks = picks,
    )

    @Test
    fun unpicked_champion_is_green() {
        val spans = richChatText("Aatrox is strong.", board()).spanStyles
        assertTrue(spans.any { it.item.color == Accent })
    }

    @Test
    fun picked_champions_take_their_side_colour_and_longest_name_wins() {
        val picks = List<Int?>(10) { null }.toMutableList().also {
            it[0] = 1   // blue Aatrox
            it[5] = 2   // red Vi
        }
        val spans = richChatText("Aatrox, Vi and Viktor.", board(picks)).spanStyles
        val colours = spans.map { it.item.color }
        assertTrue("blue Aatrox", BlueSide in colours)
        assertTrue("red Vi", RedSide in colours)
        assertTrue("unpicked Viktor", Accent in colours)
    }
}
