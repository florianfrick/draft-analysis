package com.draftassistant.ui

import com.draftassistant.data.Champion
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatTextTest {

    private val aatrox = Champion(idx = 0, riot = 1, name = "Aatrox")
    private val vi = Champion(idx = 1, riot = 2, name = "Vi")
    private val viktor = Champion(idx = 2, riot = 3, name = "Viktor")
    private val kaiSa = Champion(idx = 3, riot = 4, name = "Kai'Sa")
    private val kSante = Champion(idx = 4, riot = 5, name = "K'Sante")
    private val velKoz = Champion(idx = 5, riot = 6, name = "Vel'Koz")

    private fun board(picks: List<Int?> = List(10) { null }) = UiState(
        champions = listOf(aatrox, vi, viktor, kaiSa, kSante, velKoz),
        byRiot = mapOf(
            1 to aatrox,
            2 to vi,
            3 to viktor,
            4 to kaiSa,
            5 to kSante,
            6 to velKoz,
        ),
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

    @Test
    fun apostrophe_names_match_possessive_and_curly_apostrophes() {
        val text = "Kai'Sa's poke beats K’Sante; Vel'Koz zones."
        val mentions = championMentions(text, board())
        assertEquals(
            listOf("Kai'Sa", "K’Sante", "Vel'Koz"),
            mentions.map { text.substring(it.start, it.end) },
        )
    }

    @Test
    fun apostrophes_around_name_are_boundaries() {
        val text = "'Kai'Sa' and Vel'Koz's poke."
        val mentions = championMentions(text, board())
        assertEquals(
            listOf("Kai'Sa", "Vel'Koz"),
            mentions.map { text.substring(it.start, it.end) },
        )
    }

    @Test
    fun apostrophe_names_keep_side_colours() {
        val picks = List<Int?>(10) { null }.toMutableList().also {
            it[0] = 4   // blue Kai'Sa
            it[5] = 6   // red Vel'Koz
        }
        val spans = richChatText("Kai'Sa, K'Sante and Vel'Koz.", board(picks)).spanStyles
        val colours = spans.map { it.item.color }
        assertTrue("blue Kai'Sa", BlueSide in colours)
        assertTrue("unpicked K'Sante", Accent in colours)
        assertTrue("red Vel'Koz", RedSide in colours)
    }
}
