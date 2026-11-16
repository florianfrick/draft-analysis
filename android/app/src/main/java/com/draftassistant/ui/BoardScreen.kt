package com.draftassistant.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.draftassistant.data.Champion
import com.draftassistant.data.MoveKind
import com.draftassistant.data.Suggestion
import kotlin.math.roundToInt

@Composable
fun BoardScreen(viewModel: BoardViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()
    var target by remember { mutableStateOf<PickerTarget?>(null) }

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
                else -> DraftContent(state, viewModel) { target = it }
            }
        }
    }

    target?.let { picked ->
        ChampionPickerSheet(
            state = state,
            target = picked,
            onDismiss = { target = null },
            onChoose = { champion ->
                when (picked) {
                    is PickerTarget.Slot -> viewModel.placePick(picked.slot, champion.riot)
                    is PickerTarget.Ban -> viewModel.placeBan(picked.side, champion.riot)
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
                TextButton(onClick = viewModel::reset) {
                    Text("Reset", color = Muted, fontSize = 13.sp)
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
    val label = state.patches.getOrNull(state.patchId) ?: "patch ${state.patchId}"

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
            state.patches.forEachIndexed { index, patch ->
                DropdownMenuItem(
                    text = {
                        Text(
                            text = patch,
                            color = if (index == state.patchId) Accent else TextPrimary,
                            fontSize = 13.sp,
                        )
                    },
                    onClick = {
                        onSelect(index)
                        open = false
                    },
                )
            }
        }
    }
}

@Composable
private fun FirstPickToggle(value: Int, onChange: (Int) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("first pick", color = Muted, fontSize = 11.sp)
        Spacer(Modifier.width(8.dp))
        SidePill("Blue", value == 0, BlueSide) { onChange(0) }
        Spacer(Modifier.width(6.dp))
        SidePill("Red", value == 1, RedSide) { onChange(1) }
    }
}

@Composable
private fun SidePill(label: String, selected: Boolean, accent: Color, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(if (selected) accent.copy(alpha = 0.18f) else PanelSunken)
            .border(1.dp, if (selected) accent else Line, RoundedCornerShape(6.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 5.dp),
    ) {
        Text(
            text = label,
            color = if (selected) accent else Muted,
            fontSize = 12.sp,
        )
    }
}

// -------------------------------------------------------------------- content

@Composable
private fun DraftContent(
    state: UiState,
    viewModel: BoardViewModel,
    onOpenPicker: (PickerTarget) -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 12.dp),
    ) {
        TurnBar(state)
        Spacer(Modifier.height(10.dp))
        SidePanel(0, state, viewModel, onOpenPicker)
        Spacer(Modifier.height(10.dp))
        SidePanel(1, state, viewModel, onOpenPicker)
        Spacer(Modifier.height(16.dp))
        SuggestionsPanel(state, viewModel)
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
            Spacer(Modifier.weight(1f))
            state.modelLabel?.let {
                Text(
                    text = it.substringAfterLast('/'),
                    color = Muted,
                    fontSize = 10.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
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
