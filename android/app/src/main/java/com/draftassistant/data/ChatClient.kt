package com.draftassistant.data

import android.content.Context
import android.content.SharedPreferences
import com.draftassistant.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * Client for the optional chat Worker. No key is stored here: the only
 * identifier sent is a random per-install UUID, used by the Worker to apply a
 * server-side quota. It is an abuse throttle, not authentication.
 *
 * Nothing in this class is called by the normal board refresh path. A request
 * happens only when the user taps a question or presses send.
 */
class ChatClient private constructor(context: Context) {

    private val installId: String = installId(context.applicationContext)

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    val configured: Boolean get() = BuildConfig.CHAT_ENDPOINT.isNotBlank()

    suspend fun ask(request: ChatRequest): ChatResponse = withContext(Dispatchers.IO) {
        if (!configured) {
            throw IllegalStateException(
                "Chat endpoint is not configured. Build with -PchatEndpoint=https://<your-worker-url>.")
        }

        val body = ApiJson.encodeToString(request)
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val call = http.newCall(
            Request.Builder()
                .url(BuildConfig.CHAT_ENDPOINT.trimEnd('/'))
                .header("X-Install-Id", installId)
                .header("Content-Type", "application/json")
                .post(body)
                .build(),
        )

        val response = call.execute()
        val text = response.body?.string().orEmpty()
        val decoded = runCatching { ApiJson.decodeFromString(ChatResponse.serializer(), text) }
            .getOrElse {
                throw IllegalStateException(
                    "Chat server returned HTTP ${response.code} with an unreadable response.")
            }

        if (!response.isSuccessful) {
            throw IllegalStateException(decoded.error ?: "Chat server returned HTTP ${response.code}.")
        }
        if (decoded.answer.isBlank()) {
            throw IllegalStateException(decoded.error ?: "Chat server returned an empty answer.")
        }
        decoded
    }

    companion object {
        private const val PREFS = "draft_assistant_chat"
        private const val INSTALL_ID = "install_id"

        @Volatile
        private var instance: ChatClient? = null

        fun initialize(context: Context) {
            if (instance == null) {
                synchronized(this) {
                    if (instance == null) instance = ChatClient(context.applicationContext)
                }
            }
        }

        fun get(): ChatClient = instance
            ?: throw IllegalStateException("ChatClient.initialize() was not called")

        private fun installId(context: Context): String {
            val prefs: SharedPreferences =
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val existing = prefs.getString(INSTALL_ID, null)
            if (!existing.isNullOrBlank()) return existing
            val fresh = UUID.randomUUID().toString().replace("-", "")
            prefs.edit().putString(INSTALL_ID, fresh).apply()
            return fresh
        }
    }
}
