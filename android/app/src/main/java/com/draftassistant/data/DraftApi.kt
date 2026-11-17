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

    /**
     * Reach the same dev backend from an emulator or a physical phone.
     *
     * `10.0.2.2` is an emulator-only alias; on a phone it is just some address
     * on the carrier network. `127.0.0.1` reaches the host from a phone only
     * with `adb reverse tcp:8777 tcp:8777`. Trying both means the same build
     * works on either - important because Android Studio's Run recompiles with
     * BuildConfig.API_BASE every time, undoing a one-off -PapiBase override.
     *
     * The address that answers is cached, so this costs nothing after the
     * first call; a transport failure clears the cache implicitly by trying
     * every candidate again.
     */
    private val bases = listOf(
        baseUrl.trimEnd('/'),
        "http://10.0.2.2:8777",
        "http://127.0.0.1:8777",
    ).distinct()

    @Volatile
    private var workingBase: String? = null

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

    private fun <T> get(path: String, decoder: DeserializationStrategy<T>): T =
        call(decoder) { base -> Request.Builder().url(base + path).get().build() }

    private fun <T> post(
        path: String,
        payload: BoardRequest,
        decoder: DeserializationStrategy<T>,
    ): T {
        val body = json.encodeToString(BoardRequest.serializer(), payload).toRequestBody(mediaType)
        return call(decoder) { base -> Request.Builder().url(base + path).post(body).build() }
    }

    /**
     * A transport failure means "this address is not the backend" - try the
     * next one. An HTTP error means the backend *did* answer; surfacing that
     * is more useful than silently hopping to another host.
     */
    private fun <T> call(
        decoder: DeserializationStrategy<T>,
        requestAt: (String) -> Request,
    ): T {
        val preferred = workingBase
        val order = if (preferred == null) bases else listOf(preferred) + bases.filter { it != preferred }
        var lastFailure: IOException? = null
        for (base in order) {
            try {
                val result = client.newCall(requestAt(base)).execute().use { response ->
                    val text = response.body?.string().orEmpty()
                    if (!response.isSuccessful) throw ApiException(envelope(text, response.code))
                    json.decodeFromString(decoder, text)
                }
                workingBase = base
                return result
            } catch (e: ApiException) {
                throw e
            } catch (e: IOException) {
                lastFailure = e
            }
        }
        throw IOException("No backend at ${order.joinToString(", ")}", lastFailure)
    }

    /** Pulls `{"error": ...}` out of a failure body; falls back to the status code. */
    private fun envelope(text: String, code: Int): String {
        val message = runCatching {
            json.decodeFromString(ErrorResponse.serializer(), text).error
        }.getOrNull()
        return message?.takeIf { it.isNotBlank() } ?: "HTTP $code"
    }
}
