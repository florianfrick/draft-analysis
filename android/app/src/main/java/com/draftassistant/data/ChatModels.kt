package com.draftassistant.data

import kotlinx.serialization.Serializable

/**
 * Wire contract for the optional Cloudflare chat proxy.
 *
 * The app never sends a DeepSeek key. It sends an anonymous install id as a
 * header (added by `ChatClient`) plus evidence already computed on the phone.
 * The Worker owns the prompt and the upstream call; this side owns only facts.
 */
@Serializable
data class ChatMessage(
    val role: String,
    val content: String,
)

@Serializable
data class ChatRequest(
    val question: String,
    val evidence: List<String> = emptyList(),
    val history: List<ChatMessage> = emptyList(),
)

@Serializable
data class ChatUsage(
    @kotlinx.serialization.SerialName("prompt_tokens") val promptTokens: Int? = null,
    @kotlinx.serialization.SerialName("completion_tokens") val completionTokens: Int? = null,
    @kotlinx.serialization.SerialName("total_tokens") val totalTokens: Int? = null,
)

@Serializable
data class ChatResponse(
    val answer: String = "",
    val usage: ChatUsage? = null,
    val remaining: Int? = null,
    val error: String? = null,
)
