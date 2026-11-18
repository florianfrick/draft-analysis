package com.draftassistant.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** A scripted series used only for the on-device walkthrough recording. */
@Serializable
data class DemoSeries(
    val title: String = "",
    val games: List<DemoGame> = emptyList(),
)

@Serializable
data class DemoGame(
    val blue: String = "",
    val red: String = "",
    @SerialName("firstPick") val firstPick: Int = 0,
    val moves: List<DemoMove> = emptyList(),
)

@Serializable
data class DemoMove(
    val k: String,
    val s: Int,
    val c: Int,
    val name: String = "",
)
