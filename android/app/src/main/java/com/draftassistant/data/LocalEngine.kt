package com.draftassistant.data

import android.content.Context
import android.util.Log
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln

/**
 * Offline backend: runs the shipped checkpoints on the phone through
 * ExecuTorch and returns the *same* DTOs the server's board routes publish.
 *
 * The whole point of this class is contract fidelity. Every quirk that is
 * easy to get wrong on a second implementation is deliberate here:
 *
 *  - champion ids on the wire are Riot ids, model tensors use 0..172 indices;
 *  - `picks` is by SLOT (blue 0-4, red 5-9), not by turn order;
 *  - the fearless pool blocks picks but not bans: a champion from an earlier
 *    game may still be suggested as a ban;
 *  - legality is applied to logits with -1e4 *before* the top-k, never by
 *    filtering the top-k afterwards;
 *  - patch ids are 1-based vocabulary rows (0 = UNK), not list positions;
 *  - `pad_mask` 1 means "this slot is empty", because that is what the
 *    transformer's `src_key_padding_mask` expects.
 *
 * Tensor inputs are all int64 because ExecuTorch's Android API has no
 * `fromBlob(BooleanArray, ...)`; the exported graphs cast the 0/1 masks to
 * bool internally (see `tools/export_offline.py`).
 */
class LocalEngine private constructor(
    private val meta: OfflineMeta,
    private val policy: Module,
    private val masker: Module,
) {
    private val lock = Any()

    /** Model index -> Riot id is not needed on the answer path; names are the key. */
    private val riotToIdx = HashMap<Int, Int>()
    private val idxToName = Array(NUM_CHAMPS) { "" }

    /** Model header labels, shaped like `model_info()` on the server. */
    val policyInfo: ModelInfo
    val maskerInfo: ModelInfo

    init {
        for (c in meta.champions) {
            if (c.idx !in 0 until NUM_CHAMPS) continue
            idxToName[c.idx] = c.name
            riotToIdx[c.riot] = c.idx
        }
        policyInfo = meta.models.policy.asInfo()
        maskerInfo = meta.models.masker.asInfo()
    }

    // -------------------------------------------------------------- metadata

    /** `/api/champions` without a network. Image paths point at copied files. */
    fun champions(): ChampionsResponse = ChampionsResponse(
        champions = meta.champions,
        roles = meta.roles,
        movePlan = meta.movePlan,
        patches = meta.patches,
        defaultPatch = meta.defaultPatch,
        defaultPatchId = meta.defaultPatchId,
        model = maskerInfo,
        policyModel = policyInfo,
    )

    // --------------------------------------------------------------- nextMove

    /** The ranked move on the clock, straight from the policy head. */
    fun nextMove(request: BoardRequest): NextMoveResponse {
        val board = buildBoard(request)
        val turn = DraftOrder.nextTurn(board.plan, board.nBans, board.nPicks)

        // A complete draft has no distribution to rank. The server still
        // reports how much of the pool was blocked, so keep that here too.
        if (turn == null || turn.index >= DraftOrder.PLAN_LEN) {
            return completeNextMove(board)
        }

        val pid = patchIdFor(request.patchId)
        val k = (request.k ?: 8).coerceIn(1, 20)
        val (top, legal) = synchronized(lock) {
            val logits = runPolicy(board, turn.index, turn.kind, turn.side, pid)
            // `taken` is always illegal. The fearless pool is illegal only
            // on picks - a champion picked in an earlier game may still be
            // banned this game (the training rule, kept by the serving head).
            val p = softmax(applyLegality(logits, board, board.taken,
                poolOnPicks = turn.kind == MoveKind.PICK))
            topK(p, k) to legalCount(board)
        }

        return NextMoveResponse(
            model = policyInfo,
            state = MoveState(
                moveIdx = turn.index,
                kind = kindName(turn.kind),
                side = turn.side,
                nLegal = legal,
                complete = false,
                nUnavailable = board.poolCount,
                firstPick = request.firstPick,
            ),
            patch = patchKey(pid),
            suggestions = toSuggestions(top),
            note = "",
        ).also { response ->
            Log.d(TAG, "nextMove move=${turn.index} ${kindName(turn.kind)} " +
                "side=${turn.side} patch=${response.patch} pool=${board.poolCount} " +
                "top=" + response.suggestions.joinToString(" ") {
                    "${it.name}:${String.format(Locale.ROOT, "%.4f", it.prob)}"
                })
        }
    }

    private fun completeNextMove(board: Board): NextMoveResponse = NextMoveResponse(
        model = policyInfo,
        state = MoveState(
            moveIdx = DraftOrder.PLAN_LEN,
            kind = null,
            side = null,
            nLegal = legalCount(board),
            complete = true,
            nUnavailable = board.poolCount,
            firstPick = board.firstPick,
        ),
        patch = patchKey(patchIdFor(board.patchId)),
        suggestions = emptyList(),
        note = "",
    )

    // ---------------------------------------------------------------- suggest

    /**
     * Role labels and confidence for PLACED PICKS only. The Android app has no
     * role-board grid: the UI uses `assignment`/`conf` for the role badges on
     * picks and for chat evidence, and uses `nextMove` for the suggestion list.
     * Both come from the champion-role table via a joint one-to-one search;
     * no model forward is needed here.
     */
    fun suggest(request: BoardRequest): SuggestResponse {
        val board = buildBoard(request)
        val turn = DraftOrder.nextTurn(board.plan, board.nBans, board.nPicks)

        val lanes = assignLanes(board.picks)
        val assignment = arrayOfNulls<String>(N_SLOTS)
        val conf = arrayOfNulls<Double>(N_SLOTS)
        for (slot in 0 until N_SLOTS) {
            val champ = board.picks[slot]
            if (champ < 0 || lanes[slot] < 0) continue
            assignment[slot] = roleName(lanes[slot])
            conf[slot] = laneConfidence(champ, lanes[slot])
        }

        val openRoles = HashMap<String, List<String>>()
        for (side in 0 until 2) {
            val taken = HashSet<String>()
            for (r in 0 until 5) {
                val lane = lanes[side * 5 + r]
                if (lane >= 0) taken.add(roleName(lane))
            }
            openRoles[if (side == 0) "blue" else "red"] =
                meta.roles.filter { it !in taken }
        }

        Log.d(TAG, "roles assignment=${assignment.toList()} (no role board in UI)")
        return SuggestResponse(
            blue = emptyMap(),
            red = emptyMap(),
            assignment = assignment.toList(),
            conf = conf.toList(),
            openRoles = openRoles,
            complete = turn == null,
            method = "none",
            fearlessFiltered = board.poolCount,
            model = maskerInfo,
        )
    }

    private fun markFlex(cells: MutableMap<String, List<Suggestion>>) {
        val seen = HashMap<String, Int>()
        for (row in cells.values) {
            for (s in row) seen[s.name] = (seen[s.name] ?: 0) + 1
        }
        for (key in cells.keys.toList()) {
            cells[key] = cells[key].orEmpty().map {
                if ((seen[it.name] ?: 0) > 1) it.copy(flex = true) else it
            }
        }
    }

    // ----------------------------------------------------------------- boards

    private data class Board(
        val plan: List<DraftMove>,
        val firstPick: Int,
        val patchId: Int?,
        val picks: IntArray,          // model indices, -1 = empty, by SLOT
        val bans: IntArray,           // model indices in draft order, -1 = empty
        val taken: BooleanArray,      // picks + bans
        val pool: BooleanArray,       // fearless: unpickable, still bannable
        val nBans: Int,
        val nPicks: Int,
    ) {
        val poolCount: Int get() = pool.count { it }
    }

    private fun buildBoard(request: BoardRequest): Board {
        val plan = DraftOrder.parse(meta.movePlan)[request.firstPick]
            ?: throw IllegalArgumentException("no draft plan for first_pick=${request.firstPick}")
        val picks = IntArray(N_SLOTS) { -1 }
        val bans = IntArray(N_BANS) { -1 }
        val taken = BooleanArray(NUM_CHAMPS)
        val pool = BooleanArray(NUM_CHAMPS)

        for (slot in 0 until N_SLOTS) {
            val riot = request.picks.getOrNull(slot) ?: continue
            val idx = riotToIdx[riot]
                ?: throw IllegalArgumentException("unknown champion id $riot")
            if (taken[idx]) throw IllegalArgumentException("champion $riot appears twice")
            picks[slot] = idx
            taken[idx] = true
        }
        var nBans = 0
        for (riot in request.bans) {
            val idx = riotToIdx[riot]
                ?: throw IllegalArgumentException("unknown champion id $riot")
            if (taken[idx]) throw IllegalArgumentException("champion $riot picked and banned")
            if (nBans >= N_BANS) throw IllegalArgumentException("more than $N_BANS bans")
            bans[nBans++] = idx
            taken[idx] = true
        }
        for (riot in request.fearless) {
            val idx = riotToIdx[riot] ?: continue
            pool[idx] = true
        }
        return Board(
            plan = plan,
            firstPick = request.firstPick,
            patchId = request.patchId,
            picks = picks,
            bans = bans,
            taken = taken,
            pool = pool,
            nBans = nBans,
            nPicks = picks.count { it >= 0 },
        )
    }

    private fun legalCount(board: Board): Int =
        NUM_CHAMPS - board.taken.count { it } - board.poolCount

    // -------------------------------------------------------------- inference

    /**
     * TEMP EXPERIMENT: pro-only Siamese policy instead of the shipped naive
     * transformer. It takes the masks/state batch from
     * `solo_queue.model.policy.infer._policy_batch` and returns raw [1,173]
     * logits; legality is applied inside the graph and again by the caller.
     */
    private fun runPolicy(
        board: Board,
        moveIdx: Int,
        kind: MoveKind,
        team: Int,
        patchId: Int,
    ): FloatArray {
        val pad = CHAMP_PAD_ID.toLong()
        val champIds = LongArray(N_SLOTS) { pad }
        val roleIds = LongArray(N_SLOTS) { ROLE_FLEX.toLong() }
        val padMask = LongArray(N_SLOTS) { 1L } // 1 = empty/padded
        for (slot in 0 until N_SLOTS) {
            val champ = board.picks[slot]
            if (champ >= 0) {
                champIds[slot] = champ.toLong()
                padMask[slot] = 0L
            }
        }

        val banIds = LongArray(N_BANS) { pad }
        for (i in 0 until board.nBans) banIds[i] = board.bans[i].toLong()

        val fearless = LongArray(FEARLESS_SLOTS) { pad }
        var p = 0
        for (i in 0 until NUM_CHAMPS) {
            if (!board.pool[i]) continue
            if (p >= FEARLESS_SLOTS) break
            fearless[p++] = i.toLong()
        }

        // The serving boundary folds the fearless pool into taken_mask so a
        // pool champion cannot be suggested on a ban either.
        val taken = LongArray(NUM_CHAMPS) { 0L }
        val unavailable = LongArray(NUM_CHAMPS) { 0L }
        for (i in 0 until NUM_CHAMPS) {
            if (board.taken[i] || board.pool[i]) taken[i] = 1L
            if (board.pool[i]) unavailable[i] = 1L
        }

        val inputs = arrayOf(
            EValue.from(Tensor.fromBlob(champIds, longArrayOf(1, N_SLOTS.toLong()))),
            EValue.from(Tensor.fromBlob(roleIds, longArrayOf(1, N_SLOTS.toLong()))),
            EValue.from(Tensor.fromBlob(padMask, longArrayOf(1, N_SLOTS.toLong()))),
            EValue.from(Tensor.fromBlob(banIds, longArrayOf(1, N_BANS.toLong()))),
            EValue.from(Tensor.fromBlob(fearless, longArrayOf(1, FEARLESS_SLOTS.toLong()))),
            EValue.from(Tensor.fromBlob(taken, longArrayOf(1, NUM_CHAMPS.toLong()))),
            EValue.from(Tensor.fromBlob(unavailable, longArrayOf(1, NUM_CHAMPS.toLong()))),
            EValue.from(Tensor.fromBlob(longArrayOf(moveIdx.toLong()), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(longArrayOf(PHASE_OF[moveIdx].toLong()), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(longArrayOf(team.toLong()), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(longArrayOf(board.firstPick.toLong()), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(longArrayOf(if (kind == MoveKind.BAN) 1L else 0L), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(longArrayOf(patchId.toLong()), longArrayOf(1))),
        )
        val outputs = policy.forward(*inputs)
        return outputs[0].toTensor().dataAsFloatArray
    }

    /** Masker forward: hide [target], ask it to complete the slot at [lane]. */
    private fun runMasker(board: Board, lanes: IntArray, target: Int, lane: Int): FloatArray {
        val mask = NUM_CHAMPS.toLong()  // the [MASK] token id
        val champIds = LongArray(N_SLOTS) { mask }
        val roleIds = LongArray(N_SLOTS) { ROLE_FLEX.toLong() }
        for (slot in 0 until N_SLOTS) {
            val champ = board.picks[slot]
            if (champ >= 0) {
                champIds[slot] = champ.toLong()
                if (lanes[slot] >= 0) roleIds[slot] = lanes[slot].toLong()
            }
        }
        champIds[target] = mask
        roleIds[target] = lane.toLong()
        // SiameseBase's pad_mask means "this slot is padding". A live board
        // has ten real slots; the empty ones are [MASK] tokens, not padding.
        val padMask = LongArray(N_SLOTS) { 0L }

        val outputs = masker.forward(
            EValue.from(Tensor.fromBlob(champIds, longArrayOf(1, N_SLOTS.toLong()))),
            EValue.from(Tensor.fromBlob(roleIds, longArrayOf(1, N_SLOTS.toLong()))),
            EValue.from(Tensor.fromBlob(padMask, longArrayOf(1, N_SLOTS.toLong()))),
        )
        val all = outputs[0].toTensor().dataAsFloatArray // [1,10,173]
        val from = target * NUM_CHAMPS
        return FloatArray(NUM_CHAMPS) { all[from + it] }
    }

    /**
     * The training-time legality rule: taken (picks + bans) is always illegal;
     * the fearless pool is illegal on picks but not on bans. The server's
     * role cells are always picks, hence `poolOnPicks = true` there too.
     */
    private fun applyLegality(
        logits: FloatArray,
        board: Board,
        taken: BooleanArray,
        poolOnPicks: Boolean,
    ): FloatArray {
        val out = logits.copyOf()
        for (i in 0 until NUM_CHAMPS) {
            if (taken[i] || (poolOnPicks && board.pool[i])) out[i] = ILLEGAL_LOGIT
        }
        return out
    }

    // ------------------------------------------------------------ role table

    /** P(lane | champion) table, keyed by MODEL index (not Riot id). */
    private val laneCounts: Map<Int, IntArray> by lazy {
        meta.laneCounts.mapValues { (_, row) -> row.toIntArray() }
            .mapKeys { (key, _) -> key.toIntOrNull() ?: -1 }
    }

    private fun laneLogp(champ: Int, lane: Int): Double {
        val counts = laneCounts[champ]
        if (counts == null || counts.size != 5) return ln(1.0 / 5.0)
        val n = counts.sum()
        if (n == 0) return ln(1.0 / 5.0)
        return ln((counts[lane] + ALPHA) / (n + ALPHA * 5.0))
    }

    /**
     * Lane per slot for a 10-slot board, -1 for empty. Maximises the summed
     * log-likelihood over the <=120 injective placements per team, exactly as
     * `lanes.assign_lanes` does: two flex champions listing jungle get split
     * by whichever second choice is likelier. A per-champion argmax would put
     * both of them in jungle and disagree with the server on every flex pick.
     */
    private fun assignLanes(picks: IntArray): IntArray {
        val out = IntArray(N_SLOTS) { -1 }
        for (team in 0 until 2) {
            val slots = ArrayList<Pair<Int, Int>>()
            for (r in 0 until 5) {
                val slot = team * 5 + r
                if (picks[slot] >= 0) slots.add(slot to picks[slot])
            }
            if (slots.isEmpty()) continue

            var bestScore = Double.NEGATIVE_INFINITY
            val bestLanes = IntArray(slots.size) { -1 }
            val used = BooleanArray(5)
            val current = IntArray(slots.size)

            fun search(i: Int, score: Double) {
                if (i == slots.size) {
                    if (score > bestScore) {
                        bestScore = score
                        System.arraycopy(current, 0, bestLanes, 0, current.size)
                    }
                    return
                }
                val champ = slots[i].second
                for (lane in 0 until 5) {
                    if (used[lane]) continue
                    used[lane] = true
                    current[i] = lane
                    search(i + 1, score + laneLogp(champ, lane))
                    used[lane] = false
                }
            }
            search(0, 0.0)
            for (j in slots.indices) out[slots[j].first] = bestLanes[j]
        }
        return out
    }

    /** Champions that play [lane] often enough to be legal for a lane query. */
    private fun legalInLane(lane: Int): IntArray {
        val out = ArrayList<Int>()
        for (champ in 0 until NUM_CHAMPS) {
            val counts = laneCounts[champ] ?: continue
            val n = counts.sum()
            if (n > 0 && counts[lane] >= meta.laneFloor &&
                counts[lane].toDouble() / n >= meta.laneShare
            ) out.add(champ)
        }
        return out.toIntArray()
    }

    private fun laneConfidence(champ: Int, lane: Int): Double = exp(laneLogp(champ, lane))

    // ----------------------------------------------------------------- utils

    private fun patchIdFor(requested: Int?): Int {
        val pid = requested ?: return PATCH_UNK
        // 0 is UNK; 1..107 are the published rows. n_patches is 108, so the
        // last published id (107) is still in range.
        return if (pid in 0..meta.patches.size) pid else PATCH_UNK
    }

    private fun patchKey(pid: Int): String? =
        meta.patches.firstOrNull { it.id == pid }?.key

    private fun roleName(lane: Int): String =
        meta.roles.getOrNull(lane) ?: "TOP"

    private fun kindName(kind: MoveKind): String =
        if (kind == MoveKind.BAN) "ban" else "pick"

    private fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return logits
        val max = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size) { exp(logits[it] - max) }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }

    /** Sorted descending, skipping zero mass. The server filters `v > 0`. */
    private fun topK(probs: FloatArray, k: Int): List<Pair<Int, Float>> {
        val indices = (0 until probs.size).filter { probs[it] > 0f }.toMutableList()
        indices.sortByDescending { probs[it] }
        return indices.take(k).map { it to probs[it] }
    }

    private fun toSuggestions(top: List<Pair<Int, Float>>): List<Suggestion> =
        // `idx` is the MODEL index (0..172): the UI resolves it through
        // `byIdx`. `name` is the corpus spelling the server publishes.
        top.map { (idx, p) -> Suggestion(idx = idx, name = idxToName[idx], prob = p.toDouble()) }

    private fun OfflineModel.asInfo(): ModelInfo {
        val last = ckpt.substringAfterLast('/', ckpt)
        val parent = ckpt.substringBeforeLast('/', "")
        return ModelInfo(
            ckpt = if (parent.isEmpty()) last else "$parent/$last",
            valTop5 = valTop5,
        )
    }

    companion object {
        private const val TAG = "LocalEngine"
        private const val NUM_CHAMPS = 173
        private const val N_SLOTS = 10
        private const val N_BANS = 10
        private const val FEARLESS_SLOTS = 40
        private const val CHAMP_PAD_ID = NUM_CHAMPS + 1
        private const val ROLE_FLEX = 5
        private const val PATCH_UNK = 0
        private const val ALPHA = 0.02
        private const val ILLEGAL_LOGIT = -1e4f
        private const val NEG_INF = -1e9f
        // 6 bans, 6 picks, 4 bans, 4 picks.
        private val PHASE_OF = intArrayOf(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)

        @Volatile
        private var cached: LocalEngine? = null

        @Volatile
        private var appContext: Context? = null

        /** Called from MainActivity before the first composition. */
        fun initialize(context: Context) {
            appContext = context.applicationContext
        }

        /** The process-wide engine. Loading happens on the caller's thread. */
        fun get(): LocalEngine {
            val context = appContext
                ?: error("LocalEngine.initialize() must run before the ViewModel is created")
            return cached ?: synchronized(this) {
                cached ?: load(context).also { cached = it }
            }
        }

        private fun load(context: Context): LocalEngine {
            val json = context.assets.open("offline.json").use { it.readBytes().toString(Charsets.UTF_8) }
            val meta = ApiJson.decodeFromString(OfflineMeta.serializer(), json)

            // Module.load needs a real path, and Coil needs real files for the
            // bundled portraits. Copy once; a size match means "already there".
            val policyPath = materialize(context, "policy.pte")
            val maskerPath = materialize(context, "masker.pte")
            val iconDir = File(context.filesDir, "icons")
            val icons = context.assets.list("icons").orEmpty()
            if (!iconDir.isDirectory || iconDir.listFiles { f -> f.extension == "png" }?.size != icons.size) {
                iconDir.mkdirs()
                for (name in icons) {
                    context.assets.open("icons/$name").use { input ->
                        File(iconDir, name).outputStream().use { input.copyTo(it) }
                    }
                }
            }
            val withFiles = meta.copy(
                champions = meta.champions.map { champ ->
                    val file = File(iconDir, champ.img.orEmpty().substringAfterLast('/'))
                    if (file.isFile) champ.copy(img = file.toURI().toString()) else champ
                },
            )

            // Fail loudly here rather than on the first board tap: a missing
            // .pte or a JNI mismatch would otherwise look like an empty app.
            val policy = Module.load(policyPath)
            val masker = Module.load(maskerPath)
            Log.i(TAG, "loaded ExecuTorch modules from ${policyPath.length} / ${maskerPath.length} bytes; " +
                "${withFiles.champions.size} champions, ${withFiles.patches.size} patches")
            return LocalEngine(withFiles, policy, masker)
        }

        private fun materialize(context: Context, name: String): String {
            val target = File(context.filesDir, name)
            val size = context.assets.open(name).use { it.available().toLong() }
            if (!target.isFile || target.length() != size) {
                context.assets.open(name).use { input ->
                    target.outputStream().use { input.copyTo(it) }
                }
            }
            return target.absolutePath
        }
    }
}

// ------------------------------------------------------------------ metadata

@Serializable
private data class OfflineMeta(
    val champions: List<Champion> = emptyList(),
    val roles: List<String> = emptyList(),
    @SerialName("move_plan") val movePlan: Map<String, List<List<kotlinx.serialization.json.JsonElement>>> = emptyMap(),
    val patches: List<PatchOption> = emptyList(),
    @SerialName("default_patch_id") val defaultPatchId: Int = 0,
    @SerialName("default_patch") val defaultPatch: String? = null,
    @SerialName("lane_counts") val laneCounts: Map<String, List<Int>> = emptyMap(),
    @SerialName("lane_share") val laneShare: Double = 0.1,
    @SerialName("lane_floor") val laneFloor: Int = 5,
    val models: OfflineModels = OfflineModels(),
)

@Serializable
private data class OfflineModels(
    val policy: OfflineModel = OfflineModel(),
    val masker: OfflineModel = OfflineModel(),
)

@Serializable
private data class OfflineModel(
    val ckpt: String = "",
    val arch: String = "",
    @SerialName("val_top5") val valTop5: Double? = null,
)
