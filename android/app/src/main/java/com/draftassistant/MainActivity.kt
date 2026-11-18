package com.draftassistant

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.draftassistant.data.LocalEngine
import com.draftassistant.ui.BoardScreen
import com.draftassistant.ui.DraftTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // The engine owns two JNI modules and 36 MB of bundled weights. Give
        // it the application context once, before the ViewModel asks for it.
        LocalEngine.initialize(applicationContext)
        // `--es demo blg_geng_worlds2023` starts the scripted walkthrough.
        val demo = intent.getStringExtra("demo")
        setContent {
            DraftTheme {
                BoardScreen(demo = demo)
            }
        }
    }
}
