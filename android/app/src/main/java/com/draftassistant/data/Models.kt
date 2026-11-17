package com.draftassistant.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement

/**
 * Wire models for the Python backend. Field names mirror `frontend/server.py`
 * exactly, because this client is a consumer of that contract, not a second
 * implementation of it.
 *
 * `ignoreUnknownKeys` matters more than usual here: `model_info` grows keys
 * (stage, ensemble numbers) without notice, and a strict decoder would turn a
 * cosmetic server change into a blank screen.
 */
val ApiJson: Json = Json {
    ignoreUnknownKeys = true
    isLenient = true
    coerceInputValues = true
    explicitNulls = true
    // Required, not cosmetic: without it a field equal to its default is
    // omitted, so `fearless = emptyList()` would never reach the wire and the
    // server would read the missing key as "keep the existing pool" instead of
    // "this series has no pool".
    encodeDefaults = true
}

/** One champion as `/api/champions` publishes it. `img` is a dDragon CDN URL. */
@Serializable
data class Champion(
    val idx: Int,
    val riot: Int,
    val name: String,
    val img: String? = null,
)

/** Subset of `model_info()`; every field is optional on purpose. */
@Serializable
data class ModelInfo(
    val ckpt: String? = null,
    val path: String? = null,
    val kind: String? = null,
    @SerialName("val_top5") val valTop5: Double? = null,
    val epoch: Int? = null,
)

/**
 * `GET /api/champions`.
 *
 * `movePlan` is the canonical 20-move order, published by the server precisely
 * so the client does not carry a copy. Shape: {"0": [["ban", 0], ...], "1": ...}
 * where the second element is the SIDE (0 blue, 1 red) - not a slot index.
 */
@Serializable
data class ChampionsResponse(
    val champions: List<Champion> = emptyList(),
    val roles: List<String> = emptyList(),
    @SerialName("move_plan") val movePlan: Map<String, List<List<JsonElement>>> = emptyMap(),
    val patches: List<String> = emptyList(),
    @SerialName("default_patch") val defaultPatch: String? = null,
    @SerialName("default_patch_id") val defaultPatchId: Int = 0,
    val model: ModelInfo? = null,
    @SerialName("policy_model") val policyModel: ModelInfo? = null,
)

/** A ranked champion from either route. `idx` is null if the server could not resolve it. */
@Serializable
data class Suggestion(
    val idx: Int? = null,
    val name: String,
    val prob: Double = 0.0,
    val flex: Boolean? = null,
)

/** The board `state` block on `/api/next-move`. */
@Serializable
data class MoveState(
    @SerialName("move_idx") val moveIdx: Int = 0,
    val kind: String? = null,
    val side: Int? = null,
    @SerialName("n_legal") val nLegal: Int = 0,
    val complete: Boolean = false,
    @SerialName("n_unavailable") val nUnavailable: Int = 0,
    @SerialName("first_pick") val firstPick: Int = 0,
)

@Serializable
data class NextMoveResponse(
    val model: ModelInfo? = null,
    val state: MoveState? = null,
    val patch: String? = null,
    val suggestions: List<Suggestion> = emptyList(),
    val note: String? = null,
    val error: String? = null,
)

/**
 * `POST /api/suggest` - the role board.
 *
 * `assignment` and `conf` are indexed by SLOT (0-4 blue, 5-9 red), matching
 * `picks`. The role keys inside `blue`/`red` are the UI's TOP..SUPPORT strings
 * from `/api/champions`' `roles`, never numeric indices.
 */
@Serializable
data class SuggestResponse(
    val blue: Map<String, List<Suggestion>> = emptyMap(),
    val red: Map<String, List<Suggestion>> = emptyMap(),
    val assignment: List<String?> = emptyList(),
    val conf: List<Double?> = emptyList(),
    @SerialName("open_roles") val openRoles: Map<String, List<String>> = emptyMap(),
    val complete: Boolean = false,
    val method: String? = null,
    /**
     * How many pool champions the answer was conditioned on. The head takes the
     * pool as an input and re-scores; a masking checkpoint was trained without
     * it and can only have its answer filtered, so `method == "masker"` with a
     * non-zero count is a weaker statement than it looks.
     */
    @SerialName("fearless_filtered") val fearlessFiltered: Int = 0,
    val model: ModelInfo? = null,
    val error: String? = null,
)

/**
 * The series context the board cannot carry: which game this is, how many there
 * are, and how far each got. Numbers only - the server renders these into lines
 * itself, and no client text reaches a model prompt.
 */
@Serializable
data class SeriesBody(
    val game: Int,
    val of: Int,
    val games: List<SeriesGameBody>,
)

@Serializable
data class SeriesGameBody(
    val game: Int,
    val picks: Int,
    val fearless: Int,
    val closed: Boolean,
)

/**
 * The body both board routes accept.
 *
 * `picks` is always 10 entries, null meaning empty - the server indexes it by
 * slot. `bans` is the flat list (the split `bans_blue`/`bans_red` pair is an
 * alternative the web client also uses; sending the flat list alone is the
 * unambiguous form).
 *
 * `fearless` is the pool for *this game*: champions picked in earlier games of
 * the series. It is sent as an explicit list even when empty, because the
 * server treats *absent* as "keep the existing pool" and `[]` as "no pool" -
 * and `ApiJson` sets `encodeDefaults` so an empty list is not dropped on the
 * way out.
 */
@Serializable
data class BoardRequest(
    val picks: List<Int?>,
    val bans: List<Int>,
    @SerialName("first_pick") val firstPick: Int,
    val fearless: List<Int> = emptyList(),
    val series: SeriesBody? = null,
    @SerialName("patch_id") val patchId: Int? = null,
    val k: Int? = null,
)

@Serializable
data class ErrorResponse(val error: String? = null)
