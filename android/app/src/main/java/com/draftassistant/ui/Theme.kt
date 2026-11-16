package com.draftassistant.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Palette lifted from frontend/static/style.css so the app reads as the same
// product rather than a lookalike.
val Bg = Color(0xFF14171C)
val Panel = Color(0xFF1D222B)
val PanelRaised = Color(0xFF232A35)
val PanelSunken = Color(0xFF191E26)
val Line = Color(0xFF2C3340)
val TextPrimary = Color(0xFFE8ECF3)
val Muted = Color(0xFF8A93A5)
val Accent = Color(0xFF59B25A)
val BlueSide = Color(0xFF2F7CD4)
val RedSide = Color(0xFFD44A2F)

private val Scheme = darkColorScheme(
    primary = Accent,
    onPrimary = Color(0xFF0B120B),
    secondary = BlueSide,
    onSecondary = Color.White,
    background = Bg,
    onBackground = TextPrimary,
    surface = Panel,
    onSurface = TextPrimary,
    surfaceVariant = PanelRaised,
    onSurfaceVariant = Muted,
    outline = Line,
    error = Color(0xFFE24B4A),
    onError = Color(0xFF1A0A0A),
)

/** The web UI is dark-only, so this theme is too - no system-dark branching. */
@Composable
fun DraftTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
