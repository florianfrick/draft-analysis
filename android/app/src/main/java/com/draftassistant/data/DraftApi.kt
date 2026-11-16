package com.draftassistant.data

import com.draftassistant.BuildConfig
import kotlinx.serialization.DeserializationStrategy
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/** A server refusal that already carries the message the backend wrote. */
class ApiException(message: String) : IOException(message)

/**
 * Thin client over the three routes the MVP needs.
 *
 * No Retrofit: the surface is three calls and the error envelope is a single
 * `{"error": "..."}` field, so a converter layer would be more moving parts
 * than it saves.
 *
 * The read timeout is generous because `/api/next-move` and `/api/suggest` run
 * model inference synchronously - a cold first call on a laptop can take
 * several seconds, and a 10s default would surface as a spurious failure.
 */
class DraftApi(private val baseUrl: String = BuildConfig.API_BASE) {

    private val base = baseUrl.trimEnd('/')
    private val json = ApiJson
    private val mediaType = "application/json; charset=utf-8".toMediaType()

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    fun champions(): ChampionsResponse =
        get("/api/champions", ChampionsResponse.serializer())

    fun nextMove(request: BoardRequest): NextMoveResponse =
        post("/api/next-move", request, NextMoveResponse.serializer())

    fun suggest(request: BoardRequest): SuggestResponse =
        post("/api/suggest", request, SuggestResponse.serializer())

    private fun <T> get(path: String, decoder: DeserializationStrategy<T>): T {
        val request = Request.Builder().url(base + path).get().build()
        return client.newCall(request).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw ApiException(envelope(text, response.code))
            json.decodeFromString(decoder, text)
        }
    }

    private fun <T> post(
        path: String,
        payload: BoardRequest,
        decoder: DeserializationStrategy<T>,
    ): T {
        val body = json.encodeToString(BoardRequest.serializer(), payload).toRequestBody(mediaType)
        val request = Request.Builder().url(base + path).post(body).build()
        return client.newCall(request).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw ApiException(envelope(text, response.code))
            json.decodeFromString(decoder, text)
        }
    }

    /** Pulls `{"error": ...}` out of a failure body; falls back to the status code. */
    private fun envelope(text: String, code: Int): String {
        val message = runCatching {
            json.decodeFromString(ErrorResponse.serializer(), text).error
        }.getOrNull()
        return message?.takeIf { it.isNotBlank() } ?: "HTTP $code"
    }
}
