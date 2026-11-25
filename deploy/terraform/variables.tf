variable "region" {
  description = "AWS region for Lambda, logs and the Function URL."
  type        = string
  default     = "us-west-2"
}

variable "name" {
  description = "Shared name for the ECR repo, Lambda function and log group."
  type        = string
  default     = "loldraft-web"
}

variable "image_tag" {
  description = "Tag pushed to the public ECR repo and deployed to Lambda."
  type        = string
  default     = "latest"
}

variable "web_origin" {
  description = "Exact browser origin allowed by CORS. No trailing slash."
  type        = string
  default     = "https://florianfrick.github.io"
}

variable "worker_base_url" {
  description = "Cloudflare Worker OpenAI-compatible base URL (note /v1)."
  type        = string
  default     = "https://draft-assistant-chat.floriangfrick.workers.dev/v1"
}

variable "llm_model" {
  description = "Model id the Worker maps to DeepSeek."
  type        = string
  default     = "deepseek-flash"
}

variable "llm_token" {
  description = "Value of the Cloudflare WEBAPP_LLM_TOKEN secret. Server-side only."
  type        = string
  sensitive   = true
}

variable "memory_size" {
  description = "MB. 2048 fits the two default CPU heads with headroom."
  type        = number
  default     = 2048
}

variable "timeout" {
  description = "Lambda timeout in seconds. Covers a ~50s cold start plus one agent turn."
  type        = number
  default     = 300
}

variable "agent_calls" {
  description = "Tool-call budget per agent turn in the hosted deployment."
  type        = number
  default     = 12
}
