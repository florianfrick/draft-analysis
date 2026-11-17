package com.draftassistant.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.draftassistant.data.Champion

/** What the picker is filling: a pick slot, a ban list, or the fearless pool. */
sealed interface PickerTarget {
    data class Slot(val slot: Int) : PickerTarget
    data class Ban(val side: Int) : PickerTarget
    data object Fearless : PickerTarget
}

private fun titleFor(target: PickerTarget): String = when (target) {
    is PickerTarget.Slot -> "Pick champion — ${sideName(target.slot / 5)}"
    is PickerTarget.Ban -> "Ban champion — ${sideName(target.side)}"
    PickerTarget.Fearless -> "Add to the fearless pool"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChampionPickerSheet(
    state: UiState,
    target: PickerTarget,
    takenRiot: (Int) -> Boolean,
    onDismiss: () -> Unit,
    onChoose: (Champion) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var query by remember { mutableStateOf("") }

    val filtered = remember(query, state.champions) {
        val needle = query.trim().lowercase()
        if (needle.isEmpty()) {
            state.champions
        } else {
            state.champions.filter { it.name.lowercase().contains(needle) }
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Panel,
        contentColor = TextPrimary,
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp),
        ) {
            Text(
                text = titleFor(target),
                color = TextPrimary,
                fontSize = 15.sp,
                fontWeight = FontWeight.Medium,
            )

            Spacer(Modifier.height(10.dp))

            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("search champion", color = Muted, fontSize = 13.sp) },
                singleLine = true,
                shape = RoundedCornerShape(8.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Accent,
                    unfocusedBorderColor = Line,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary,
                    cursorColor = Accent,
                ),
                modifier = Modifier.fillMaxWidth(),
            )

            Spacer(Modifier.height(12.dp))

            if (filtered.isEmpty()) {
                Text("no champion matches", color = Muted, fontSize = 13.sp)
                Spacer(Modifier.height(24.dp))
            } else {
                LazyVerticalGrid(
                    columns = GridCells.Adaptive(minSize = 64.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    contentPadding = PaddingValues(bottom = 24.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 380.dp),
                ) {
                    items(filtered, key = { it.riot }) { champion ->
                        ChampionCell(
                            champion = champion,
                            taken = takenRiot(champion.riot),
                            onClick = { onChoose(champion) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ChampionCell(champion: Champion, taken: Boolean, onClick: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .clickable(enabled = !taken, onClick = onClick)
            .alpha(if (taken) 0.32f else 1f)
            .padding(2.dp),
    ) {
        AsyncImage(
            model = champion.img,
            contentDescription = champion.name,
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .size(46.dp)
                .clip(RoundedCornerShape(6.dp))
                .background(PanelRaised),
        )
        Spacer(Modifier.height(3.dp))
        Text(
            text = champion.name,
            color = TextPrimary,
            fontSize = 10.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

/** Shared by the board and the picker. Slots 0-4 are blue, 5-9 red. */
fun sideName(side: Int): String = if (side == 0) "Blue" else "Red"
