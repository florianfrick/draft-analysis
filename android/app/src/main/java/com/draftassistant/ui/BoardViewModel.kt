package com.draftassistant.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.draftassistant.BuildConfig
import com.draftassistant.data.ApiException
import com.draftassistant.data.BoardRequest
import com.draftassistant.data.Champion
import com.draftassistant.data.DraftApi
import com.draftassistant.data.DraftMove
import com.draftassistant.data.DraftOrder
import com.draftassistant.data.MoveKind
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
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

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
    val patches: List<String> = emptyList(),
    val patchId: Int = 0,

    val firstPick: Int = 0,
    val picks: List<Int?> = List(10) { null },
    val bansBlue: List<Int> = emptyList(),
    val bansRed: List<Int> = emptyList(),

    val turn: Turn? = null,

    val suggestions: List<Suggestion> = emptyList(),
    val suggestionIndex: Int = -1,
    val assignment: List<String?> = List(10) { null },
    val conf: List<Double?> = List(10) { null },

    val note: String? = null,
    val modelLabel: String? = null,
) {
    val plan: List<DraftMove>? get() = plans[firstPick]
    val allBans: List<Int> get() = bansBlue + bansRed
    val banCount: Int get() = bansBlue.size + bansRed.size
    val pickCount: Int get() = picks.count { it != null }

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
    }
}

class BoardViewModel : ViewModel() {

    private val api = DraftApi()
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var refreshJob: Job? = null

    init {
        reload()
    }

    // ------------------------------------------------------------------ load

    fun reload() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true, error = null) }
            val result = runCatching { api.champions() }

            result.onSuccess { response ->
                val sorted = response.champions.sortedBy { it.name.lowercase() }
                _state.update {
                    it.copy(
                        loading = false,
                        champions = sorted,
                        byRiot = sorted.associateBy { c -> c.riot },
                        byIdx = sorted.associateBy { c -> c.idx },
                        byName = sorted.associateBy { c -> c.name },
                        roles = response.roles,
                        plans = DraftOrder.parse(response.movePlan),
                        patches = response.patches,
                        patchId = response.defaultPatchId,
                        modelLabel = response.policyModel?.ckpt ?: response.policyModel?.path,
                    )
                }
                refresh()
            }.onFailure { failure ->
                _state.update { it.copy(loading = false, error = friendly(failure)) }
            }
        }
    }

    // --------------------------------------------------------------- board edits

    fun setFirstPick(side: Int) = mutate { it.copy(firstPick = side) }

    fun setPatch(id: Int) = mutate { it.copy(patchId = id) }

    fun placePick(slot: Int, riot: Int) = mutate { state ->
        if (slot !in 0..9 || state.contains(riot) || state.picks[slot] != null) {
            state
        } else {
            state.copy(picks = state.picks.toMutableList().also { it[slot] = riot })
        }
    }

    fun placeBan(side: Int, riot: Int) = mutate { state ->
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
        if (state.picks[slot] == null) state
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
            suggestionIndex = -1,
        )
    }

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

    // ---------------------------------------------------------------- refresh

    private fun mutate(block: (UiState) -> UiState) {
        val before = _state.value
        val after = block(before)
        if (after === before) return

        _state.update {
            after.copy(
                turn = DraftOrder.nextTurn(after.plan, after.banCount, after.pickCount),
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
                fearless = emptyList(),
                patchId = snapshot.patchId,
                k = 8,
            )

            val (nextResult, roleResult) = coroutineScope {
                val next = async(Dispatchers.IO) { runCatching { api.nextMove(request) } }
                val roles = async(Dispatchers.IO) { runCatching { api.suggest(request.copy(k = 5)) } }
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
                    suggestionIndex = next?.state?.moveIdx ?: -1,
                    assignment = roles?.assignment?.takeIf { it.size == 10 } ?: state.assignment,
                    conf = roles?.conf?.takeIf { it.size == 10 } ?: state.conf,
                    note = next?.note ?: next?.error ?: roles?.error,
                    modelLabel = next?.model?.ckpt ?: state.modelLabel,
                    error = failure?.let(::friendly),
                )
            }
        }
    }

    private fun friendly(failure: Throwable): String = when (failure) {
        is ApiException -> failure.message ?: "The backend refused the request"
        is ConnectException, is UnknownHostException ->
            "Cannot reach ${BuildConfig.API_BASE}. Is frontend/serve.sh running?"
        is SocketTimeoutException ->
            "The backend did not answer in time. The first model call can be slow."
        else -> failure.message ?: failure::class.java.simpleName
    }

    private companion object {
        const val DEBOUNCE_MS = 140L
    }
}
