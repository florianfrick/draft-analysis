package com.draftassistant

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.draftassistant.ui.BoardScreen
import com.draftassistant.ui.DraftTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DraftTheme {
                BoardScreen()
            }
        }
    }
}
