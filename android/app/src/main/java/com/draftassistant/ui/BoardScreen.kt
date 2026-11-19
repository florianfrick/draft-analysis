package com.draftassistant.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.draftassistant.data.Champion
import com.draftassistant.data.ChatMessage
import com.draftassistant.data.MoveKind
import com.draftassistant.data.Suggestion
import kotlin.math.roundToInt

@Composable
fun BoardScreen(viewModel: BoardViewModel = viewModel(), demo: String? = null) {
    val state by viewModel.state.collectAsState()
    var target by remember { mutableStateOf<PickerTarget?>(null) }
    val context = LocalContext.current.applicationContext
    LaunchedEffect(demo) {
        if (demo != null) viewModel.playDemo(context, demo)
    }

    Scaffold(
        containerColor = Bg,
        topBar = { DraftTopBar(state, viewModel) },
    ) { insets ->
        Box(
            Modifier
                .fillMaxSize()
                .padding(insets),
        ) {
            when {
                state.loading -> CenteredMessage("loading champions…")
                state.champions.isEmpty() -> ErrorPane(state.error, viewModel::reload)
                else -> DraftContent(state, viewModel, { target = it }, demo = demo != null)
            }
        }
    }

    target?.let { picked ->
        ChampionPickerSheet(
            state = state,
            target = picked,
            takenRiot = { riot ->
                when (picked) {
                    // The pool blocks a pick but never a ban: a champion picked
                    // in an earlier game is still legal to ban in this one.
                    is PickerTarget.Ban -> state.contains(riot)
                    else -> state.contains(riot) || state.pool.contains(riot)
                }
            },
            onDismiss = { target = null },
            onChoose = { champion ->
                when (picked) {
                    is PickerTarget.Slot -> viewModel.placePick(picked.slot, champion.riot)
                    is PickerTarget.Ban -> viewModel.placeBan(picked.side, champion.riot)
                    PickerTarget.Fearless -> viewModel.addFearless(champion.riot)
                }
                target = null
            },
        )
    }
}

// --------------------------------------------------------------------- chrome

@Composable
private fun DraftTopBar(state: UiState, viewModel: BoardViewModel) {
    Surface(color = Panel) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Draft Assistant",
                    color = TextPrimary,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Medium,
                )
                Spacer(Modifier.weight(1f))
                if (state.busy) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(15.dp),
                        strokeWidth = 2.dp,
                        color = Accent,
                    )
                    Spacer(Modifier.width(10.dp))
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Chat", color = Muted, fontSize = 11.sp)
                    Spacer(Modifier.width(5.dp))
                    Switch(
                        checked = state.chatEnabled,
                        onCheckedChange = viewModel::setChatEnabled,
                        modifier = Modifier.height(28.dp),
                    )
                }
                Spacer(Modifier.width(8.dp))
                TextButton(onClick = viewModel::reset) {
                    Text("Reset Game", color = Muted, fontSize = 13.sp)
                }
            }

            Spacer(Modifier.height(6.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                PatchMenu(state, viewModel::setPatch)
                Spacer(Modifier.width(14.dp))
                FirstPickToggle(state.firstPick, viewModel::setFirstPick)
            }
        }
    }
}

@Composable
private fun PatchMenu(state: UiState, onSelect: (Int) -> Unit) {
    var open by remember { mutableStateOf(false) }
    // Selected by id, not by position: `patch_id` is a row of the checkpoint's
    // vocabulary and the list arrives newest-first, so index != id.
    val label = state.patches.firstOrNull { it.id == state.patchId }?.key
        ?: "patch ${state.patchId}"

    Box {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .clip(RoundedCornerShape(6.dp))
                .background(PanelSunken)
                .border(1.dp, Line, RoundedCornerShape(6.dp))
                .clickable(enabled = state.patches.isNotEmpty()) { open = true }
                .padding(horizontal = 9.dp, vertical = 5.dp),
        ) {
            Text("patch", color = Muted, fontSize = 11.sp)
            Spacer(Modifier.width(6.dp))
            Text(label, color = TextPrimary, fontSize = 12.sp)
        }

        DropdownMenu(
            expanded = open,
            onDismissRequest = { open = false },
            containerColor = PanelRaised,
        ) {
            state.patches.forEach { patch ->
                DropdownMenuItem(
                    text = {
                        Text(
                            text = patch.key,
                            color = if (patch.id == state.patchId) Accent else TextPrimary,
                            fontSize = 13.sp,
                        )
                    },
                    onClick = {
                        onSelect(patch.id)
                        open = false
                    },
                )
            }
        }
    }
}

/**
 * One control, not two side buttons: a switch whose track carries the label
 * "First Pick" and whose thumb slides to the side that will pick first.
 * value 0 = blue, 1 = red (the same convention the request DTO uses).
 */
@Composable
private fun FirstPickToggle(value: Int, onChange: (Int) -> Unit) {
    val accent = if (value == 1) RedSide else BlueSide
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .width(94.dp)
            .height(28.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(accent.copy(alpha = 0.16f))
            .border(1.dp, accent, RoundedCornerShape(14.dp))
            .clickable { onChange(if (value == 0) 1 else 0) }
            .padding(horizontal = 5.dp),
    ) {
        if (value == 0) {
            SwitchThumb(accent)
            Spacer(Modifier.width(6.dp))
            Text("First Pick", color = TextPrimary, fontSize = 11.sp)
        } else {
            Text("First Pick", color = TextPrimary, fontSize = 11.sp)
            Spacer(Modifier.width(6.dp))
            SwitchThumb(accent)
        }
    }
}

@Composable
private fun SwitchThumb(accent: Color) {
    Box(
        Modifier
            .size(16.dp)
            .clip(CircleShape)
            .background(accent),
    )
}

// -------------------------------------------------------------------- content

@Composable
private fun DraftContent(
    state: UiState,
    viewModel: BoardViewModel,
    onOpenPicker: (PickerTarget) -> Unit,
    demo: Boolean = false,
) {
    val scroll = rememberScrollState()
    // The walkthrough recording wants the board and the live suggestions in
    // one frame; the normal app starts at the top, where both panels fit only
    // after a scroll.
    LaunchedEffect(demo) {
        if (demo) scroll.animateScrollTo(560)
    }
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(scroll)
            .padding(horizontal = 12.dp, vertical = 12.dp),
    ) {
        SeriesPanel(state, viewModel)
        Spacer(Modifier.height(10.dp))
        FearlessPanel(state, viewModel, onOpenPicker)
        Spacer(Modifier.height(10.dp))
        TurnBar(state)
        Spacer(Modifier.height(10.dp))
        SidePanel(0, state, viewModel, onOpenPicker)
        Spacer(Modifier.height(10.dp))
        SidePanel(1, state, viewModel, onOpenPicker)
        Spacer(Modifier.height(16.dp))
        SuggestionsPanel(state, viewModel)
        if (state.chatEnabled) {
            Spacer(Modifier.height(16.dp))
            ChatPanel(state, viewModel)
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun TurnBar(state: UiState) {
    val turn = state.turn
    val accent = when (turn?.side) {
        0 -> BlueSide
        1 -> RedSide
        else -> Muted
    }
    val label = when {
        turn == null -> "draft complete"
        else -> {
            val verb = if (turn.kind == MoveKind.PICK) "pick" else "ban"
            "${sideName(turn.side)} to $verb · move ${turn.index + 1}/20"
        }
    }

    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(accent),
        )
        Spacer(Modifier.width(8.dp))
        Text(label, color = TextPrimary, fontSize = 13.sp)
    }
}

@Composable
private fun SidePanel(
    side: Int,
    state: UiState,
    viewModel: BoardViewModel,
    onOpenPicker: (PickerTarget) -> Unit,
) {
    val accent = if (side == 0) BlueSide else RedSide
    val bans = state.bans(side)
    val isTurn = state.turn?.side == side

    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Panel)
            .border(
                width = 1.dp,
                color = if (isTurn) accent.copy(alpha = 0.55f) else Line,
                shape = RoundedCornerShape(12.dp),
            )
            .padding(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = sideName(side),
                color = accent,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.width(8.dp))
            if (isTurn) {
                Text("on the clock", color = accent.copy(alpha = 0.85f), fontSize = 11.sp)
            }
            Spacer(Modifier.weight(1f))
            Text("${bans.size}/5 bans", color = Muted, fontSize = 11.sp)
        }

        Spacer(Modifier.height(9.dp))

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "ban",
                color = Muted,
                fontSize = 10.sp,
                modifier = Modifier.width(26.dp),
            )
            bans.forEachIndexed { index, riot ->
                BanSlot(
                    champion = state.byRiot[riot],
                    onClear = { viewModel.clearBan(side, index) },
                )
                Spacer(Modifier.width(6.dp))
            }
            if (bans.size < 5) {
                AddSlot(accent) { onOpenPicker(PickerTarget.Ban(side)) }
            }
        }

        Spacer(Modifier.height(10.dp))

        Row(verticalAlignment = Alignment.Top) {
            (0 until 5).forEach { column ->
                val slot = side * 5 + column
                PickSlot(
                    champion = state.picks[slot]?.let { state.byRiot[it] },
                    role = state.assignment.getOrNull(slot),
                    conf = state.conf.getOrNull(slot),
                    highlight = isTurn &&
                        state.turn?.kind == MoveKind.PICK &&
                        state.picks[slot] == null,
                    accent = accent,
                    onOpen = { onOpenPicker(PickerTarget.Slot(slot)) },
                    onClear = { viewModel.clearPick(slot) },
                    modifier = Modifier.weight(1f),
                )
                if (column < 4) Spacer(Modifier.width(6.dp))
            }
        }
    }
}

@Composable
private fun PickSlot(
    champion: Champion?,
    role: String?,
    conf: Double?,
    highlight: Boolean,
    accent: Color,
    onOpen: () -> Unit,
    onClear: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(8.dp))
                .background(if (champion != null) PanelRaised else PanelSunken)
                .border(
                    width = 1.dp,
                    color = if (highlight) accent else Line,
                    shape = RoundedCornerShape(8.dp),
                )
                .clickable(enabled = champion == null, onClick = onOpen),
            contentAlignment = Alignment.Center,
        ) {
            if (champion != null) {
                AsyncImage(
                    model = champion.img,
                    contentDescription = champion.name,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
                // Explicit remove affordance rather than tap-to-clear: wiping a
                // pick by mis-tapping mid-draft is not recoverable by undo.
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .size(17.dp)
                        .clip(CircleShape)
                        .background(Color(0xCC0B0D10))
                        .clickable(onClick = onClear),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("×", color = TextPrimary, fontSize = 10.sp)
                }
            } else {
                Text(
                    text = "+",
                    color = if (highlight) accent else Muted,
                    fontSize = 17.sp,
                )
            }
        }

        Spacer(Modifier.height(3.dp))

        Text(
            text = champion?.name ?: "—",
            color = if (champion != null) TextPrimary else Muted,
            fontSize = 10.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )

        if (champion != null && role != null) {
            val percent = conf?.let { " ${(it * 100).roundToInt()}%" } ?: ""
            Text(
                text = role.take(3) + percent,
                color = Muted,
                fontSize = 9.sp,
                maxLines = 1,
            )
        }
    }
}

@Composable
private fun BanSlot(champion: Champion?, onClear: () -> Unit) {
    Box(
        modifier = Modifier
            .size(30.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(PanelRaised)
            .border(1.dp, Line, RoundedCornerShape(6.dp))
            .clickable(onClick = onClear),
        contentAlignment = Alignment.Center,
    ) {
        if (champion != null) {
            AsyncImage(
                model = champion.img,
                contentDescription = champion.name,
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }
}

@Composable
private fun AddSlot(accent: Color, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(30.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(PanelSunken)
            .border(1.dp, Line, RoundedCornerShape(6.dp))
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text("+", color = accent, fontSize = 14.sp)
    }
}

// ------------------------------------------------------------ series / pool

@Composable
private fun SeriesPanel(state: UiState, viewModel: BoardViewModel) {
    val series = state.series
    val current = series.games.getOrNull(series.current) ?: return

    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Panel)
            .border(1.dp, Line, RoundedCornerShape(12.dp))
            .padding(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Series", color = Muted, fontSize = 11.sp)
            Spacer(Modifier.width(8.dp))
            Row(
                modifier = Modifier
                    .weight(1f)
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                series.games.forEachIndexed { index, game ->
                    GameChip(
                        game = game,
                        selected = index == series.current,
                        onClick = { viewModel.openGame(index) },
                    )
                }
            }
        }

        Spacer(Modifier.height(9.dp))

        Row(verticalAlignment = Alignment.CenterVertically) {
            SmallButton("next game ▶", enabled = state.canAdvance) { viewModel.nextGame() }
            Spacer(Modifier.width(6.dp))
            SmallButton("clear") { viewModel.clearSeries() }
            Spacer(Modifier.weight(1f))
            Text(
                text = "${series.current + 1}/${series.games.size} · " +
                    "${current.fearless.size} fearless",
                color = Muted,
                fontSize = 10.sp,
            )
        }
    }
}

@Composable
private fun GameChip(game: GameSlot, selected: Boolean, onClick: () -> Unit) {
    val picks = game.picks.count { it != null }
    val accent = if (game.firstPick == 1) RedSide else BlueSide

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clip(RoundedCornerShape(7.dp))
            .background(if (selected) PanelRaised else PanelSunken)
            .border(1.dp, if (selected) Accent else Line, RoundedCornerShape(7.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 5.dp),
    ) {
        Text(
            text = (if (game.firstPick == 1) "R" else "B") + " $picks/10" +
                if (game.done) " ✓" else "",
            color = accent,
            fontSize = 9.sp,
        )
    }
}

@Composable
private fun FearlessPanel(
    state: UiState,
    viewModel: BoardViewModel,
    onOpenPicker: (PickerTarget) -> Unit,
) {
    val pool = state.pool

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Panel)
            .border(1.dp, Line, RoundedCornerShape(12.dp))
            .padding(10.dp),
    ) {
        Text("Fearless", color = Muted, fontSize = 11.sp)
        Spacer(Modifier.width(8.dp))
        Row(
            modifier = Modifier
                .weight(1f)
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (pool.isEmpty()) {
                Text("this game — the pool is empty", color = Muted, fontSize = 10.sp)
            }
            pool.forEach { riot ->
                state.byRiot[riot]?.let { champion ->
                    PoolSlot(champion) { viewModel.removeFearless(riot) }
                }
            }
        }
        Spacer(Modifier.width(8.dp))
        SmallButton("＋") { onOpenPicker(PickerTarget.Fearless) }
    }
}

@Composable
private fun PoolSlot(champion: Champion, onRemove: () -> Unit) {
    Box(Modifier.size(28.dp)) {
        AsyncImage(
            model = champion.img,
            contentDescription = champion.name,
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxSize()
                .clip(RoundedCornerShape(6.dp))
                .background(PanelRaised)
                .border(1.dp, Line, RoundedCornerShape(6.dp)),
        )
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .size(14.dp)
                .clip(CircleShape)
                .background(Color(0xCC0B0D10))
                .clickable(onClick = onRemove),
            contentAlignment = Alignment.Center,
        ) {
            Text("×", color = TextPrimary, fontSize = 8.sp)
        }
    }
}

@Composable
private fun SmallButton(label: String, enabled: Boolean = true, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(PanelSunken)
            .border(1.dp, Line, RoundedCornerShape(6.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 9.dp, vertical = 5.dp),
    ) {
        Text(
            text = label,
            color = if (enabled) TextPrimary else Muted.copy(alpha = 0.5f),
            fontSize = 11.sp,
        )
    }
}

// ---------------------------------------------------------------- suggestions

@Composable
private fun SuggestionsPanel(state: UiState, viewModel: BoardViewModel) {
    Column(Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "Model suggests",
                color = TextPrimary,
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.width(8.dp))
            state.turn?.let {
                Text(
                    text = if (it.kind == MoveKind.PICK) "next pick" else "next ban",
                    color = Muted,
                    fontSize = 11.sp,
                )
            }
        }

        Spacer(Modifier.height(8.dp))

        when {
            state.suggestions.isEmpty() && state.busy ->
                Text("asking the model…", color = Muted, fontSize = 12.sp)

            state.suggestions.isEmpty() ->
                Text(
                    text = state.error ?: "no suggestions",
                    color = if (state.error != null) MaterialTheme.colorScheme.error else Muted,
                    fontSize = 12.sp,
                )

            else -> Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                state.suggestions.forEach { suggestion ->
                    SuggestionRow(suggestion, state) { viewModel.applySuggestion(suggestion) }
                }
            }
        }

        state.note?.takeIf { it.isNotBlank() }?.let {
            Spacer(Modifier.height(8.dp))
            Text(it, color = Muted, fontSize = 10.sp)
        }
    }
}

@Composable
private fun SuggestionRow(suggestion: Suggestion, state: UiState, onClick: () -> Unit) {
    val champion = suggestion.idx?.let { state.byIdx[it] } ?: state.byName[suggestion.name]

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(Panel)
            .border(1.dp, Line, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        AsyncImage(
            model = champion?.img,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .size(30.dp)
                .clip(RoundedCornerShape(6.dp))
                .background(PanelRaised),
        )

        Spacer(Modifier.width(10.dp))

        Column(Modifier.weight(1f)) {
            Text(
                text = suggestion.name,
                color = TextPrimary,
                fontSize = 13.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Spacer(Modifier.height(5.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(3.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(Line),
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(suggestion.prob.toFloat().coerceIn(0.02f, 1f))
                        .fillMaxHeight()
                        .background(Accent),
                )
            }
        }

        Spacer(Modifier.width(10.dp))

        Text(
            text = "${(suggestion.prob * 100).roundToInt()}%",
            color = Muted,
            fontSize = 12.sp,
        )
    }
}

// ----------------------------------------------------------------------- chat

@Composable
private fun ChatPanel(state: UiState, viewModel: BoardViewModel) {
    val scroll = rememberScrollState()
    var draft by remember { mutableStateOf("") }

    LaunchedEffect(state.chatMessages.size) {
        if (state.chatMessages.isNotEmpty()) scroll.animateScrollTo(scroll.maxValue)
    }

    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Panel)
            .border(1.dp, Line, RoundedCornerShape(12.dp))
            .padding(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                "Ask about this board",
                color = TextPrimary,
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.weight(1f))
            Text("off by default", color = Muted, fontSize = 10.sp)
        }

        Spacer(Modifier.height(4.dp))
        Text(
            "Answers use the board at the moment you send. No call is made on a board change.",
            color = Muted,
            fontSize = 10.sp,
        )

        Spacer(Modifier.height(8.dp))
        QuickQuestion("Justify the optimal next move.") {
            viewModel.askChat("Justify the optimal next move.")
        }
        QuickQuestion("What team composition archetype is each team aiming toward?") {
            viewModel.askChat("What team composition archetype is each team aiming toward?")
        }

        Spacer(Modifier.height(8.dp))
        if (state.chatMessages.isEmpty()) {
            Text("No questions yet. The local model above is unchanged.", color = Muted, fontSize = 11.sp)
        } else {
            Column(
                Modifier
                    .fillMaxWidth()
                    .heightIn(max = 340.dp)
                    .verticalScroll(scroll),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                state.chatMessages.forEach { ChatBubble(it) }
            }
        }

        state.chatError?.let { error ->
            Spacer(Modifier.height(6.dp))
            Text(error, color = MaterialTheme.colorScheme.error, fontSize = 11.sp)
        }

        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask a question…", color = Muted, fontSize = 12.sp) },
                singleLine = true,
                maxLines = 1,
                enabled = !state.chatBusy,
            )
            Spacer(Modifier.width(8.dp))
            SmallButton(
                label = if (state.chatBusy) "…" else "send",
                enabled = draft.isNotBlank() && !state.chatBusy,
            ) {
                viewModel.askChat(draft)
                draft = ""
            }
        }

        state.chatRemaining?.let { remaining ->
            Spacer(Modifier.height(6.dp))
            Text("$remaining questions left today", color = Muted, fontSize = 10.sp)
        }
    }
}

@Composable
private fun QuickQuestion(label: String, onClick: () -> Unit) {
    TextButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        colors = ButtonDefaults.textButtonColors(contentColor = Accent),
    ) {
        Text(label, fontSize = 12.sp)
    }
}

@Composable
private fun ChatBubble(message: ChatMessage) {
    val user = message.role == "user"
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (user) PanelSunken else PanelRaised)
            .border(1.dp, Line, RoundedCornerShape(8.dp))
            .padding(8.dp),
    ) {
        Text(
            text = if (user) "You" else "Assistant",
            color = if (user) Accent else BlueSide,
            fontSize = 10.sp,
            fontWeight = FontWeight.Medium,
        )
        Spacer(Modifier.height(4.dp))
        Text(message.content, color = TextPrimary, fontSize = 12.sp)
    }
}

// --------------------------------------------------------------------- states

@Composable
private fun CenteredMessage(message: String) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text(message, color = Muted, fontSize = 13.sp)
    }
}

@Composable
private fun ErrorPane(error: String?, onRetry: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(28.dp),
        ) {
            Text(
                text = "Cannot load the draft board",
                color = TextPrimary,
                fontSize = 15.sp,
                fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = error ?: "unknown error",
                color = Muted,
                fontSize = 12.sp,
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = onRetry,
                colors = ButtonDefaults.buttonColors(
                    containerColor = PanelRaised,
                    contentColor = TextPrimary,
                ),
            ) {
                Text("Retry", fontSize = 13.sp)
            }
        }
    }
}
