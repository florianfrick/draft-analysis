package com.draftassistant.ui

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.draftassistant.data.ApiJson
import com.draftassistant.data.BoardRequest
import com.draftassistant.data.Champion
import com.draftassistant.data.ChatClient
import com.draftassistant.data.ChatMessage
import com.draftassistant.data.ChatRequest
import com.draftassistant.data.DemoSeries
import com.draftassistant.data.DraftMove
import com.draftassistant.data.DraftOrder
import com.draftassistant.data.LocalEngine
import com.draftassistant.data.MoveKind
import com.draftassistant.data.PatchOption
import com.draftassistant.data.SeriesBody
import com.draftassistant.data.SeriesGameBody
import com.draftassistant.data.Suggestion
import com.draftassistant.data.Turn
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.roundToInt

/**
 * One game of a series.
 *
 * `fearless` lives on the slot rather than being derived from earlier games,
 * because that is how the reference client models it: the pool is an explicit,
 * editable list per game (a champion can be added or removed by hand), and
 * "next game" folds the closed game's picks into a fresh copy. Deriving it from
 * the picks alone would make the manual edits unrepresentable.
 */
data class GameSlot(
    val firstPick: Int = 0,
    val picks: List<Int?> = List(10) { null },
    val bansBlue: List<Int> = emptyList(),
    val bansRed: List<Int> = emptyList(),
    val fearless: List<Int> = emptyList(),
    val done: Boolean = false,
)

data class SeriesState(
    val games: List<GameSlot> = listOf(GameSlot()),
    val current: Int = 0,
)

data class UiState(
    val loading: Boolean = true,
    val busy: Boolean = false,
    val error: String? = null,

    val champions: List<Champion> = emptyList(),
    val byRiot: Map<Int, Champion> = emptyMap(),
    val byIdx: Map<Int, Champion> = emptyMap(),
    val byName: Map<String, Champion> = emptyMap(),
    val roles: List<String> = emptyList(),

    val plans: Map<Int, List<DraftMove>> = emptyMap(),
    val patches: List<PatchOption> = emptyList(),
    val patchId: Int = 0,

    val firstPick: Int = 0,
    val picks: List<Int?> = List(10) { null },
    val bansBlue: List<Int> = emptyList(),
    val bansRed: List<Int> = emptyList(),

    val series: SeriesState = SeriesState(),
    val turn: Turn? = null,

    val suggestions: List<Suggestion> = emptyList(),
    val assignment: List<String?> = List(10) { null },
    val conf: List<Double?> = List(10) { null },

    val note: String? = null,

    // Opt-in chat. Nothing here is updated by a board change: a chat turn is
    // sent only when the user asks a question.
    val chatEnabled: Boolean = false,
    val chatMessages: List<ChatMessage> = emptyList(),
    val chatBusy: Boolean = false,
    val chatError: String? = null,
    val chatRemaining: Int? = null,
) {
    val plan: List<DraftMove>? get() = plans[firstPick]
    val allBans: List<Int> get() = bansBlue + bansRed
    val banCount: Int get() = bansBlue.size + bansRed.size
    val pickCount: Int get() = picks.count { it != null }

    val currentGame: GameSlot? get() = series.games.getOrNull(series.current)

    /** Champions picked in earlier games: unpickable, but still legal to ban. */
    val pool: List<Int> get() = currentGame?.fearless.orEmpty()

    /** A closed game that already has a successor cannot be closed again. */
    val canAdvance: Boolean
        get() = currentGame?.let { !(it.done && series.current < series.games.size - 1) } ?: false

    fun bans(side: Int): List<Int> = if (side == 0) bansBlue else bansRed

    fun contains(riot: Int): Boolean =
        picks.contains(riot) || bansBlue.contains(riot) || bansRed.contains(riot)

    fun firstEmptySlot(side: Int): Int? =
        (side * 5 until side * 5 + 5).firstOrNull { picks[it] == null }

    /** Everything that changes the model's answer. Used to reject stale replies. */
    fun sig(): String = buildString {
        append(picks.joinToString(",") { it?.toString() ?: "-" })
        append('|').append(bansBlue.joinToString(","))
        append('|').append(bansRed.joinToString(","))
        append('|').append(firstPick)
        append('|').append(patchId)
        append('|').append(pool.joinToString(","))
        append('|').append(series.current)
    }

    /** Facts the chat Worker may use. It never sees a DeepSeek key or our UI code. */
    fun chatEvidence(): List<String> {
        val lines = ArrayList<String>()
        fun names(ids: List<Int>): String =
            ids.joinToString(", ") { byRiot[it]?.name ?: "champion $it" }
                .ifEmpty { "none" }

        lines += "Board: blue [${names(picks.take(5).filterNotNull())}]; " +
            "red [${names(picks.drop(5).filterNotNull())}]"
        lines += "Bans: blue [${names(bansBlue)}]; red [${names(bansRed)}]"
        lines += "Move on the clock: " + when (val t = turn) {
            null -> "draft complete"
            else -> "${sideName(t.side)} to ${if (t.kind == MoveKind.PICK) "pick" else "ban"}, " +
                "move ${t.index + 1}/20"
        }
        lines += "Fearless pool (blocks picks, not bans): ${names(pool)}"
        lines += "Series: game ${series.current + 1}/${series.games.size}, ${pool.size} fearless"
        val patchKey = patches.firstOrNull { it.id == patchId }?.key ?: "unknown"
        lines += "Patch: $patchKey (id $patchId)"

        if (suggestions.isEmpty()) {
            lines += "predict_next_move: no answer for this board"
        } else {
            lines += "predict_next_move (share is a fraction of the legal pool, NOT correctness):"
            suggestions.take(8).forEachIndexed { i, s ->
                lines += "  #${i + 1} ${s.name}: ${(s.prob * 100).roundToInt()}% share"
            }
        }

        val roleRows = picks.mapIndexedNotNull { slot, riot ->
            val role = assignment.getOrNull(slot) ?: return@mapIndexedNotNull null
            val name = byRiot[riot]?.name ?: "champion $riot"
            val pct = conf.getOrNull(slot)?.let { " (${(it * 100).roundToInt()}% of its own picks)" }
                ?: ""
            "${sideName(slot / 5)} $name: ${role.lowercase(Locale.ROOT)}$pct"
        }
        if (roleRows.isNotEmpty()) {
            lines += "predict_roles (one-to-one per side): ${roleRows.joinToString("; ")}"
        }
        val allRoles = roles.map { it.lowercase(Locale.ROOT) }
        fun openFor(side: Int): String {
            val from = side * 5
            val taken = (from until from + 5)
                .mapNotNull { assignment.getOrNull(it)?.lowercase(Locale.ROOT) }
                .toSet()
            return allRoles.filterNot { it in taken }.joinToString(", ").ifEmpty { "none" }
        }
        lines += "open_roles: blue [${openFor(0)}]; red [${openFor(1)}]"

        note?.takeIf { it.isNotBlank() }?.let { lines += "Model note: $it" }
        return lines
    }

    fun seriesBody(): SeriesBody = SeriesBody(
        game = series.current + 1,
        of = series.games.size,
        games = series.games.mapIndexed { index, game ->
            SeriesGameBody(
                game = index + 1,
                picks = game.picks.count { it != null },
                fearless = game.fearless.size,
                closed = game.done,
            )
        },
    )
}

class BoardViewModel : ViewModel() {

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var refreshJob: Job? = null

    init {
        reload()
    }

    // --------------------------------------------------------------- demo

    /**
     * Replays a real series from `assets/demo/` for the walkthrough video.
     * Each move goes through the same `placePick`/`placeBan` path a tap uses,
     * so the suggestions on screen are the live engine's answers, not canned
     * rows. Games are separated with `nextGame(game.firstPick)`: the 2026
     * Worlds final is fearless, so the previous game's picks must become the
     * next pool, and the real series alternates the first-pick side.
     */
    fun playDemo(context: Context, name: String) {
        viewModelScope.launch {
            val series = withContext(Dispatchers.IO) {
                context.assets.open("demo/$name.json").use { input ->
                    ApiJson.decodeFromString(
                        DemoSeries.serializer(),
                        input.readBytes().toString(Charsets.UTF_8),
                    )
                }
            }
            Log.i("BoardViewModel", "demo: ${series.title} (${series.games.size} games)")
            clearSeries()
            delay(DEMO_LEAD_MS)
            series.games.forEachIndexed { index, game ->
                if (index > 0) {
                    nextGame(game.firstPick)
                    delay(DEMO_GAME_PAUSE_MS)
                }
                for (move in game.moves) {
                    when (move.k) {
                        "ban" -> placeBan(move.s, move.c)
                        "pick" -> _state.value.firstEmptySlot(move.s)?.let {
                            placePick(it, move.c)
                        }
                    }
                    delay(DEMO_STEP_MS)
                }
                delay(DEMO_GAME_PAUSE_MS)
            }
            Log.i("BoardViewModel", "demo finished")
        }
    }

    // ------------------------------------------------------------------- load

    fun reload() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true, error = null) }
            // viewModelScope runs on Dispatchers.Main. The first call also
            // copies 36 MB of .pte weights out of assets, so it must be off
            // the main thread here, not just as a courtesy to OkHttp.
            val result = withContext(Dispatchers.IO) {
                runCatching { LocalEngine.get().champions() }
            }

            result.onSuccess { response ->
                val sorted = response.champions.sortedBy { it.name.lowercase() }
                _state.update { current ->
                    current.copy(
                        loading = false,
                        champions = sorted,
                        byRiot = sorted.associateBy { c -> c.riot },
                        byIdx = sorted.associateBy { c -> c.idx },
                        byName = sorted.associateBy { c -> c.name },
                        roles = response.roles,
                        plans = DraftOrder.parse(response.movePlan),
                        patches = response.patches,
                        patchId = response.defaultPatchId,
                    ).withTurn()
                }
                refresh()
            }.onFailure { failure ->
                _state.update { it.copy(loading = false, error = friendly(failure)) }
            }
        }
    }

    // ------------------------------------------------------------- board edits

    fun setFirstPick(side: Int) = mutate { it.copy(firstPick = side) }

    fun setPatch(id: Int) = mutate { it.copy(patchId = id) }

    fun placePick(slot: Int, riot: Int) = mutate { state ->
        val blocked = state.contains(riot) || state.pool.contains(riot)
        if (slot !in 0..9 || blocked || state.picks[slot] != null) {
            state
        } else {
            state.copy(picks = state.picks.toMutableList().also { it[slot] = riot })
        }
    }

    fun placeBan(side: Int, riot: Int) = mutate { state ->
        // A champion picked in an earlier game is still legal to ban - only the
        // pick path is closed to the pool. The pool is not a "taken" list.
        val current = state.bans(side)
        if (state.contains(riot) || current.size >= 5) {
            state
        } else {
            state.copy(
                bansBlue = if (side == 0) current + riot else state.bansBlue,
                bansRed = if (side == 1) current + riot else state.bansRed,
            )
        }
    }

    fun clearPick(slot: Int) = mutate { state ->
        if (state.picks.getOrNull(slot) == null) state
        else state.copy(picks = state.picks.toMutableList().also { it[slot] = null })
    }

    fun clearBan(side: Int, position: Int) = mutate { state ->
        val current = state.bans(side)
        if (position !in current.indices) {
            state
        } else {
            val next = current.toMutableList().also { it.removeAt(position) }
            state.copy(
                bansBlue = if (side == 0) next else state.bansBlue,
                bansRed = if (side == 1) next else state.bansRed,
            )
        }
    }

    fun reset() = mutate {
        it.copy(
            picks = List(10) { null },
            bansBlue = emptyList(),
            bansRed = emptyList(),
            suggestions = emptyList(),
        )
    }

    // ------------------------------------------------------------------ chat

    /** Chat is opt-in; toggling it never sends a request. */
    fun setChatEnabled(enabled: Boolean) =
        _state.update { it.copy(chatEnabled = enabled, chatError = null) }

    fun clearChat() = _state.update {
        it.copy(chatMessages = emptyList(), chatError = null, chatRemaining = null)
    }

    /**
     * The only path that calls the hosted model. A board edit calls `refresh()`
     * instead, so the LLM is never woken up by a pick, ban, patch or series tap.
     */
    fun askChat(question: String) {
        val text = question.trim()
        val snapshot = _state.value
        if (text.isBlank() || snapshot.chatBusy || !snapshot.chatEnabled) return

        val evidence = snapshot.chatEvidence()
        val history = snapshot.chatMessages.takeLast(6)
        val userTurn = ChatMessage("user", text)
        _state.update {
            it.copy(chatBusy = true, chatError = null, chatMessages = it.chatMessages + userTurn)
        }

        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    ChatClient.get().ask(
                        ChatRequest(question = text, evidence = evidence, history = history))
                }
            }
            result.onSuccess { response ->
                _state.update { current ->
                    current.copy(
                        chatBusy = false,
                        chatError = null,
                        chatRemaining = response.remaining ?: current.chatRemaining,
                        chatMessages = current.chatMessages + ChatMessage("assistant", response.answer),
                    )
                }
            }.onFailure { failure ->
                Log.e("BoardViewModel", "chat request failed", failure)
                _state.update {
                    it.copy(chatBusy = false, chatError = failure.message?.takeIf(String::isNotBlank)
                        ?: "Chat request failed.")
                }
            }
        }
    }

    // ----------------------------------------------------------------- series

    fun openGame(index: Int) = mutate { state ->
        if (index == state.series.current) state else state.synced().loaded(index)
    }

    fun nextGame(firstPick: Int? = null) = mutate { state ->
        val base = state.synced()
        val index = base.series.current
        val current = base.series.games.getOrNull(index) ?: return@mutate state
        if (current.done && index < base.series.games.size - 1) return@mutate state

        // This game's pool + everything picked in it = the next game's pool.
        // Bans never enter it: a banned champion is legal again next game.
        val pool = LinkedHashSet(current.fearless)
        current.picks.filterNotNull().forEach { pool.add(it) }

        val games = base.series.games.toMutableList()
        games[index] = current.copy(done = true)
        // Optional override lets a scripted series follow the real first-pick
        // side; the default keeps the side that had it, as the UI expects.
        val nextFirstPick = firstPick?.coerceIn(0, 1) ?: current.firstPick
        games.add(index + 1,
                   GameSlot(firstPick = nextFirstPick, fearless = pool.toList()))

        base.copy(series = base.series.copy(games = games)).loaded(index + 1)
    }

    fun clearSeries() = mutate { state -> state.copy(series = SeriesState()).loaded(0) }

    fun addFearless(riot: Int) =
        mutate { it.withPool { pool -> if (riot in pool) pool else pool + riot } }

    fun removeFearless(riot: Int) = mutate { it.withPool { pool -> pool - riot } }

    /** Plays a suggested champion into the slot the current turn implies. */
    fun applySuggestion(suggestion: Suggestion) {
        val state = _state.value
        val champion = championFor(suggestion) ?: return
        val turn = state.turn ?: return

        when (turn.kind) {
            MoveKind.PICK -> state.firstEmptySlot(turn.side)?.let { placePick(it, champion.riot) }
            MoveKind.BAN -> placeBan(turn.side, champion.riot)
        }
    }

    private fun championFor(suggestion: Suggestion): Champion? {
        val state = _state.value
        return suggestion.idx?.let { state.byIdx[it] } ?: state.byName[suggestion.name]
    }

    // ----------------------------------------------------- slot <-> live board

    /**
     * Writes the live board into the current game slot.
     *
     * The live `picks`/`bans` are the board on screen; the slot is where it is
     * kept. Switching games moves state between the two, exactly as the web
     * client's `syncGame`/`loadGame` pair does - so an edit made in one game is
     * still there when you come back to it.
     */
    /**
     * The move on the clock is a property of the board, not of a model answer,
     * so it is recomputed on every state change - including the first load,
     * where there is no edit and `mutate` never runs. Missing that left the
     * fresh board reading "draft complete".
     */
    private fun UiState.withTurn(): UiState =
        copy(turn = DraftOrder.nextTurn(plan, banCount, pickCount))

    private fun UiState.synced(): UiState {
        val index = series.current
        val slot = series.games.getOrNull(index) ?: return this
        val updated = slot.copy(
            firstPick = firstPick,
            picks = picks,
            bansBlue = bansBlue,
            bansRed = bansRed,
        )
        if (updated == slot) return this
        val games = series.games.toMutableList().also { it[index] = updated }
        return copy(series = series.copy(games = games))
    }

    private fun UiState.loaded(index: Int): UiState {
        val slot = series.games.getOrNull(index) ?: return this
        return copy(
            series = series.copy(current = index),
            firstPick = slot.firstPick,
            picks = slot.picks,
            bansBlue = slot.bansBlue,
            bansRed = slot.bansRed,
            suggestions = emptyList(),
        )
    }

    private fun UiState.withPool(transform: (List<Int>) -> List<Int>): UiState {
        val index = series.current
        val slot = series.games.getOrNull(index) ?: return this
        val next = transform(slot.fearless).distinct()
        if (next == slot.fearless) return this
        val games = series.games.toMutableList().also { it[index] = slot.copy(fearless = next) }
        return copy(series = series.copy(games = games))
    }

    // ---------------------------------------------------------------- refresh

    private fun mutate(block: (UiState) -> UiState) {
        val before = _state.value
        val draft = block(before)
        if (draft === before) return

        val after = draft.synced().withTurn()
        _state.update {
            after.copy(
                // Role badges belong to the board they were computed for. A
                // badge carried onto a board it never saw reads as an answer,
                // so they go blank until this board's reply lands.
                assignment = List(10) { null },
                conf = List(10) { null },
            )
        }
        refresh()
    }

    /**
     * Asks both board routes about the current board.
     *
     * Debounced because a run of quick taps would otherwise fan out into a
     * model call per tap, and every reply but the last is wasted inference.
     * Both calls go out together: they are independent questions about the
     * same board, and the web client's "one board, two answers" rule depends
     * on them describing the same snapshot.
     */
    fun refresh() {
        val snapshot = _state.value
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            delay(DEBOUNCE_MS)

            val sig = snapshot.sig()
            _state.update { it.copy(busy = true) }

            val request = BoardRequest(
                picks = snapshot.picks,
                bans = snapshot.allBans,
                firstPick = snapshot.firstPick,
                fearless = snapshot.pool,
                series = snapshot.seriesBody(),
                patchId = snapshot.patchId,
                k = 8,
            )

            val (nextResult, roleResult) = coroutineScope {
                val next = async(Dispatchers.IO) {
                    runCatching { LocalEngine.get().nextMove(request) }
                }
                val roles = async(Dispatchers.IO) {
                    runCatching { LocalEngine.get().suggest(request.copy(k = 5)) }
                }
                next.await() to roles.await()
            }

            if (!isActive) return@launch
            // A reply computed for a board that has since moved must never be
            // drawn on the current one: it is indistinguishable from a good
            // answer, and that is exactly how a fresh pick looks ignored.
            if (_state.value.sig() != sig) return@launch

            val next = nextResult.getOrNull()
            val roles = roleResult.getOrNull()
            val failure = nextResult.exceptionOrNull() ?: roleResult.exceptionOrNull()

            _state.update { state ->
                state.copy(
                    busy = false,
                    suggestions = next?.suggestions.orEmpty(),
                    assignment = roles?.assignment?.takeIf { it.size == 10 } ?: state.assignment,
                    conf = roles?.conf?.takeIf { it.size == 10 } ?: state.conf,
                    note = next?.note ?: next?.error ?: roles?.error,
                    error = failure?.let(::friendly),
                )
            }
        }
    }

    /** Inference failures are local: no backend to blame or retry. */
    private fun friendly(failure: Throwable): String {
        Log.e("BoardViewModel", "local inference failed", failure)
        return when (failure) {
            is UnsatisfiedLinkError ->
                "The on-device runtime could not load. Reinstall the app."
            is OutOfMemoryError ->
                "Not enough memory to load the models. Restart the app."
            else -> failure.message ?: failure::class.java.simpleName
        }
    }

    private companion object {
        const val DEBOUNCE_MS = 140L
        const val DEMO_LEAD_MS = 2_000L
        const val DEMO_STEP_MS = 450L
        const val DEMO_GAME_PAUSE_MS = 1_400L
    }
}
