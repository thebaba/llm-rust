//! Anthropic API client implementation for chat and completion functionality.
//!
//! This module provides integration with Anthropic's Claude models through their API.

use std::collections::HashMap;

use crate::{
    builder::LLMBackend,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChoice, StreamDelta,
        StreamResponse, Tool, ToolChoice, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for interacting with Anthropic's API.
///
/// Provides methods for chat and completion requests using Anthropic's models.
#[derive(Debug)]
pub struct Anthropic {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub system: String,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: bool,
    pub thinking_budget_tokens: Option<u32>,
    client: Client,
}

/// Anthropic-specific tool format that matches their API structure
#[derive(Serialize, Debug)]
struct AnthropicTool<'a> {
    name: &'a str,
    description: &'a str,
    #[serde(rename = "input_schema")]
    schema: &'a serde_json::Value,
}

/// Configuration for the thinking feature
#[derive(Serialize, Debug)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

/// Request payload for Anthropic's messages API endpoint.
#[derive(Serialize, Debug)]
struct AnthropicCompleteRequest<'a> {
    messages: Vec<AnthropicMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

/// Individual message in an Anthropic chat conversation.
#[derive(Serialize, Debug)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: Vec<MessageContent<'a>>,
}

#[derive(Serialize, Debug)]
struct MessageContent<'a> {
    #[serde(rename = "type")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ImageSource<'a>>,
    // tool use
    #[serde(skip_serializing_if = "Option::is_none", rename = "id")]
    tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "name")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "input")]
    tool_input: Option<Value>,
    // tool result
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_use_id")]
    tool_result_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<String>,
}

#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

#[derive(Serialize, Debug)]
struct ImageSource<'a> {
    #[serde(rename = "type")]
    source_type: &'a str,
    media_type: &'a str,
    data: String,
}

/// Response from Anthropic's messages API endpoint.
#[derive(Deserialize, Debug)]
struct AnthropicCompleteResponse {
    content: Vec<AnthropicContent>,
    usage: Option<AnthropicUsage>,
}

/// Usage information from Anthropic API response.
#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<u32>,
}

/// Content block within an Anthropic API response.
#[derive(Serialize, Deserialize, Debug)]
struct AnthropicContent {
    text: Option<String>,
    #[serde(rename = "type")]
    content_type: Option<String>,
    thinking: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
    id: Option<String>,
}

/// Response from Anthropic's streaming messages API endpoint.
#[derive(Deserialize, Debug)]
struct AnthropicStreamResponse {
    #[serde(rename = "type")]
    response_type: String,
    delta: Option<AnthropicDelta>,
}

/// Delta content within an Anthropic streaming response.
#[derive(Deserialize, Debug)]
struct AnthropicDelta {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
}

// ============================================================================
// Structured Streaming Response Types (for chat_stream_struct)
// ============================================================================

/// Anthropic SSE event for structured streaming.
/// Handles all event types: message_start, content_block_start, content_block_delta,
/// content_block_stop, message_delta, message_stop.
#[derive(Deserialize, Debug)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    /// Index of content block (for content_block_* events)
    index: Option<usize>,
    /// Content block data (for content_block_start)
    content_block: Option<AnthropicContentBlockStart>,
    /// Delta data (for content_block_delta)
    delta: Option<AnthropicStreamEventDelta>,
    /// Message metadata (for message_start)
    message: Option<AnthropicMessageStart>,
    /// Usage data (for message_delta)
    usage: Option<AnthropicUsage>,
}

/// Content block start data for tool_use, text, or thinking blocks.
#[derive(Deserialize, Debug)]
struct AnthropicContentBlockStart {
    #[serde(rename = "type")]
    block_type: String,
    /// Tool call ID (for tool_use blocks)
    id: Option<String>,
    /// Tool function name (for tool_use blocks)
    name: Option<String>,
    /// Initial text content (for text blocks)
    text: Option<String>,
}

/// Delta content for structured streaming.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct AnthropicStreamEventDelta {
    #[serde(rename = "type")]
    delta_type: String,
    /// Text content (for text_delta)
    text: Option<String>,
    /// Partial JSON arguments (for input_json_delta)
    partial_json: Option<String>,
    /// Thinking content (for thinking_delta)
    thinking: Option<String>,
    /// Stop reason (for message_delta)
    stop_reason: Option<String>,
}

/// Message start metadata.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct AnthropicMessageStart {
    id: Option<String>,
    model: Option<String>,
    usage: Option<AnthropicUsage>,
}

impl std::fmt::Display for AnthropicCompleteResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for content in self.content.iter() {
            match content.content_type {
                Some(ref t) if t == "tool_use" => write!(
                    f,
                    "{{\n \"name\": {}, \"input\": {}\n}}",
                    content.name.clone().unwrap_or_default(),
                    content.input.clone().unwrap_or(serde_json::Value::Null)
                )?,
                Some(ref t) if t == "thinking" => {
                    write!(f, "{}", content.thinking.clone().unwrap_or_default())?
                }
                _ => write!(
                    f,
                    "{}",
                    self.content
                        .iter()
                        .map(|c| c.text.clone().unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join("\n")
                )?,
            }
        }
        Ok(())
    }
}

impl ChatResponse for AnthropicCompleteResponse {
    fn text(&self) -> Option<String> {
        Some(
            self.content
                .iter()
                .filter_map(|c| {
                    if c.content_type == Some("text".to_string()) || c.content_type.is_none() {
                        c.text.clone()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    fn thinking(&self) -> Option<String> {
        self.content
            .iter()
            .find(|c| c.content_type == Some("thinking".to_string()))
            .and_then(|c| c.thinking.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        match self
            .content
            .iter()
            .filter_map(|c| {
                if c.content_type == Some("tool_use".to_string()) {
                    Some(ToolCall {
                        id: c.id.clone().unwrap_or_default(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: c.name.clone().unwrap_or_default(),
                            arguments: serde_json::to_string(
                                &c.input.clone().unwrap_or(serde_json::Value::Null),
                            )
                            .unwrap_or_default(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<ToolCall>>()
        {
            v if v.is_empty() => None,
            v => Some(v),
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|anthropic_usage| {
            let cached_tokens = anthropic_usage.cache_creation_input_tokens.unwrap_or(0)
                + anthropic_usage.cache_read_input_tokens.unwrap_or(0);
            Usage {
                prompt_tokens: anthropic_usage.input_tokens,
                completion_tokens: anthropic_usage.output_tokens,
                total_tokens: anthropic_usage.input_tokens + anthropic_usage.output_tokens,
                completion_tokens_details: None,
                prompt_tokens_details: if cached_tokens > 0 {
                    Some(crate::chat::PromptTokensDetails {
                        cached_tokens: Some(cached_tokens),
                        audio_tokens: None,
                    })
                } else {
                    None
                },
            }
        })
    }
}

impl Anthropic {
    /// Creates a new Anthropic client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key for authentication
    /// * `model` - Model identifier (defaults to "claude-3-sonnet-20240229")
    /// * `max_tokens` - Maximum tokens in response (defaults to 300)
    /// * `temperature` - Sampling temperature (defaults to 0.7)
    /// * `timeout_seconds` - Request timeout in seconds (defaults to 30)
    /// * `system` - System prompt (defaults to "You are a helpful assistant.")
    /// *
    /// * `thinking_budget_tokens` - Budget tokens for thinking (optional)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning: Option<bool>,
        thinking_budget_tokens: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| "claude-3-sonnet-20240229".to_string()),
            max_tokens: max_tokens.unwrap_or(300),
            temperature: temperature.unwrap_or(0.7),
            system: system.unwrap_or_else(|| "You are a helpful assistant.".to_string()),
            timeout_seconds: timeout_seconds.unwrap_or(30),
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning: reasoning.unwrap_or(false),
            thinking_budget_tokens,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Anthropic {
    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(raw_bytes) => {
                        vec![MessageContent {
                            message_type: Some("document"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: "application/pdf",
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::ImageURL(ref url) => vec![MessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url }),
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|c| MessageContent {
                            message_type: Some("tool_use"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: Some(c.id.clone()),
                            tool_input: Some(
                                serde_json::from_str(&c.function.arguments)
                                    .unwrap_or(c.function.arguments.clone().into()),
                            ),
                            tool_name: Some(c.function.name.clone()),
                            tool_result_id: None,
                            tool_output: None,
                        })
                        .collect(),
                    MessageType::ToolResult(responses) => responses
                        .iter()
                        .map(|r| MessageContent {
                            message_type: Some("tool_result"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: Some(r.id.clone()),
                            tool_output: Some(r.function.arguments.clone()),
                        })
                        .collect(),
                },
            })
            .collect();

        let maybe_tool_slice: Option<&[Tool]> = tools.or(self.tools.as_deref());
        let anthropic_tools = maybe_tool_slice.map(|slice| {
            slice
                .iter()
                .map(|tool| AnthropicTool {
                    name: &tool.function.name,
                    description: &tool.function.description,
                    schema: &tool.function.parameters,
                })
                .collect::<Vec<_>>()
        });

        let tool_choice = match self.tool_choice {
            Some(ToolChoice::Auto) => {
                Some(HashMap::from([("type".to_string(), "auto".to_string())]))
            }
            Some(ToolChoice::Any) => Some(HashMap::from([("type".to_string(), "any".to_string())])),
            Some(ToolChoice::Tool(ref tool_name)) => Some(HashMap::from([
                ("type".to_string(), "tool".to_string()),
                ("name".to_string(), tool_name.clone()),
            ])),
            Some(ToolChoice::None) => {
                Some(HashMap::from([("type".to_string(), "none".to_string())]))
            }
            None => None,
        };

        let final_tool_choice = if anthropic_tools.is_some() {
            tool_choice.clone()
        } else {
            None
        };

        let thinking = if self.reasoning {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.thinking_budget_tokens.unwrap_or(16000),
            })
        } else {
            None
        };

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: Some(&self.system),
            stream: Some(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: anthropic_tools,
            tool_choice: final_tool_choice,
            thinking,
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Anthropic request payload: {}", json);
            }
        }

        log::debug!("Anthropic request: POST /v1/messages");
        let resp = request.send().await?;
        log::debug!("Anthropic HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        let body = resp.text().await?;
        let json_resp: AnthropicCompleteResponse = serde_json::from_str(&body)
            .map_err(|e| LLMError::HttpError(format!("Failed to parse JSON: {e}")))?;
        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(_) => unimplemented!(),
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    _ => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                },
            })
            .collect();

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: Some(&self.system),
            stream: Some(true),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: None,
            tool_choice: None,
            thinking: None,
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Anthropic API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(crate::chat::create_sse_stream(
            response,
            parse_anthropic_sse_chunk,
        ))
    }

    /// Sends a structured streaming chat request to Anthropic's API.
    ///
    /// Returns a stream of `StreamResponse` objects that include text content,
    /// tool calls, thinking content, and usage metadata.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// A stream of `StreamResponse` objects or an error
    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        // Convert messages to Anthropic format (same as chat_with_tools)
        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(raw_bytes) => {
                        vec![MessageContent {
                            message_type: Some("document"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: "application/pdf",
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::ImageURL(ref url) => vec![MessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url }),
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|c| MessageContent {
                            message_type: Some("tool_use"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: Some(c.id.clone()),
                            tool_input: Some(
                                serde_json::from_str(&c.function.arguments).unwrap_or_default(),
                            ),
                            tool_name: Some(c.function.name.clone()),
                            tool_result_id: None,
                            tool_output: None,
                        })
                        .collect(),
                    MessageType::ToolResult(results) => results
                        .iter()
                        .map(|r| MessageContent {
                            message_type: Some("tool_result"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: Some(r.id.clone()),
                            tool_output: Some(r.function.arguments.clone()),
                        })
                        .collect(),
                },
            })
            .collect();

        // Convert tools to Anthropic format if present
        let anthropic_tools: Option<Vec<AnthropicTool>> = self.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| AnthropicTool {
                    name: &t.function.name,
                    description: &t.function.description,
                    schema: &t.function.parameters,
                })
                .collect()
        });

        // Build tool choice
        let tool_choice: Option<HashMap<String, String>> = match &self.tool_choice {
            Some(ToolChoice::Auto) => {
                Some(HashMap::from([("type".to_string(), "auto".to_string())]))
            }
            Some(ToolChoice::Any) => {
                Some(HashMap::from([("type".to_string(), "any".to_string())]))
            }
            Some(ToolChoice::Tool(ref tool_name)) => Some(HashMap::from([
                ("type".to_string(), "tool".to_string()),
                ("name".to_string(), tool_name.clone()),
            ])),
            Some(ToolChoice::None) => {
                Some(HashMap::from([("type".to_string(), "none".to_string())]))
            }
            None => None,
        };

        let final_tool_choice = if anthropic_tools.is_some() {
            tool_choice
        } else {
            None
        };

        // Build thinking config if reasoning is enabled
        let thinking = if self.reasoning {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.thinking_budget_tokens.unwrap_or(16000),
            })
        } else {
            None
        };

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: Some(&self.system),
            stream: Some(true),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: anthropic_tools,
            tool_choice: final_tool_choice,
            thinking,
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Anthropic stream_struct request payload: {}", json);
            }
        }

        log::debug!("Anthropic request: POST /v1/messages (stream_struct)");
        let response = request.send().await?;
        log::debug!("Anthropic HTTP status (stream_struct): {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Anthropic API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(create_anthropic_sse_struct_stream(response))
    }
}

#[async_trait]
impl CompletionProvider for Anthropic {
    /// Sends a completion request to Anthropic's API.
    ///
    /// Converts the completion request into a chat message format.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        unimplemented!()
    }
}

#[async_trait]
impl EmbeddingProvider for Anthropic {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for Anthropic {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Speech to text not supported".to_string(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Anthropic {}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicModelListResponse {
    data: Vec<AnthropicModelEntry>,
}

impl ModelListResponse for AnthropicModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|m| m.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::Anthropic
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicModelEntry {
    created_at: DateTime<Utc>,
    id: String,
    #[serde(flatten)]
    extra: Value,
}

impl ModelListRawEntry for AnthropicModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[async_trait]
impl ModelsProvider for Anthropic {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let resp = self
            .client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;

        let result: AnthropicModelListResponse = resp.json().await?;

        Ok(Box::new(result))
    }
}

impl crate::LLMProvider for Anthropic {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

// ============================================================================
// Stateful SSE Parser for Structured Streaming
// ============================================================================

/// Stateful parser for Anthropic's SSE streaming with tool call accumulation.
///
/// Handles UTF-8 boundary issues, accumulates tool call arguments across chunks,
/// and emits structured `StreamResponse` objects.
struct AnthropicSSEStreamParser {
    /// Buffer for incomplete SSE events (split across chunks)
    event_buffer: String,
    /// Buffer for incomplete UTF-8 sequences
    utf8_buffer: Vec<u8>,
    /// Currently accumulating tool calls, indexed by content block index
    tool_buffers: HashMap<usize, ToolCall>,
    /// Accumulated usage metadata
    usage: Option<Usage>,
    /// Results queue ready to be emitted
    results: Vec<Result<StreamResponse, LLMError>>,
}

impl AnthropicSSEStreamParser {
    /// Create a new parser instance.
    fn new() -> Self {
        Self {
            event_buffer: String::new(),
            utf8_buffer: Vec::new(),
            tool_buffers: HashMap::new(),
            usage: None,
            results: Vec::new(),
        }
    }

    /// Drain and return all accumulated results.
    fn drain_results(&mut self) -> Vec<Result<StreamResponse, LLMError>> {
        std::mem::take(&mut self.results)
    }

    /// Process incoming bytes from the stream.
    ///
    /// Handles UTF-8 boundary issues and splits on SSE event boundaries (\n\n).
    fn process_bytes(&mut self, bytes: &[u8]) {
        // Append to UTF-8 buffer and try to decode
        self.utf8_buffer.extend_from_slice(bytes);

        match String::from_utf8(std::mem::take(&mut self.utf8_buffer)) {
            Ok(text) => {
                self.event_buffer.push_str(&text);
            }
            Err(e) => {
                // Handle incomplete UTF-8 sequences at chunk boundaries
                let valid_up_to = e.utf8_error().valid_up_to();
                let bytes = e.into_bytes();
                if valid_up_to > 0 {
                    // SAFETY: We know bytes[..valid_up_to] is valid UTF-8
                    let valid = unsafe { std::str::from_utf8_unchecked(&bytes[..valid_up_to]) };
                    self.event_buffer.push_str(valid);
                }
                // Keep incomplete sequence for next chunk
                self.utf8_buffer = bytes[valid_up_to..].to_vec();
            }
        }

        // SSE events are separated by double newlines
        while let Some(pos) = self.event_buffer.find("\n\n") {
            let event = self.event_buffer[..pos].to_string();
            self.event_buffer.drain(..pos + 2);

            self.parse_sse_event(&event);
        }
    }

    /// Parse a single SSE event.
    fn parse_sse_event(&mut self, event: &str) {
        for line in event.lines() {
            let line = line.trim();
            if let Some(data) = line.strip_prefix("data: ") {
                self.parse_data_payload(data);
            }
        }
    }

    /// Parse the JSON data payload from an SSE event.
    fn parse_data_payload(&mut self, data: &str) {
        let event: AnthropicStreamEvent = match serde_json::from_str(data) {
            Ok(e) => e,
            Err(e) => {
                log::debug!("Failed to parse Anthropic stream event: {}", e);
                return;
            }
        };

        match event.event_type.as_str() {
            "message_start" => {
                // Extract initial usage if present
                if let Some(msg) = event.message {
                    if let Some(usage) = msg.usage {
                        self.usage = Some(Usage {
                            prompt_tokens: usage.input_tokens,
                            completion_tokens: usage.output_tokens,
                            total_tokens: usage.input_tokens + usage.output_tokens,
                            completion_tokens_details: None,
                            prompt_tokens_details: usage.cache_read_input_tokens.map(|cached| {
                                crate::chat::PromptTokensDetails {
                                    cached_tokens: Some(cached),
                                    audio_tokens: None,
                                }
                            }),
                        });
                    }
                }
            }

            "content_block_start" => {
                if let (Some(index), Some(block)) = (event.index, event.content_block) {
                    match block.block_type.as_str() {
                        "tool_use" => {
                            // Start accumulating a new tool call
                            self.tool_buffers.insert(
                                index,
                                ToolCall {
                                    id: block.id.unwrap_or_default(),
                                    call_type: "function".to_string(),
                                    function: FunctionCall {
                                        name: block.name.unwrap_or_default(),
                                        arguments: String::new(),
                                    },
                                },
                            );
                        }
                        "text" => {
                            // Emit initial text if present
                            if let Some(text) = block.text {
                                if !text.is_empty() {
                                    self.emit_content(Some(text), None, None);
                                }
                            }
                        }
                        "thinking" => {
                            // Thinking block started, no initial content to emit
                        }
                        _ => {}
                    }
                }
            }

            "content_block_delta" => {
                if let Some(delta) = event.delta {
                    match delta.delta_type.as_str() {
                        "text_delta" => {
                            if let Some(text) = delta.text {
                                self.emit_content(Some(text), None, None);
                            }
                        }
                        "input_json_delta" => {
                            // Accumulate tool call arguments
                            if let (Some(index), Some(partial)) =
                                (event.index, delta.partial_json)
                            {
                                if let Some(tool) = self.tool_buffers.get_mut(&index) {
                                    tool.function.arguments.push_str(&partial);
                                }
                            }
                        }
                        "thinking_delta" => {
                            if let Some(thinking) = delta.thinking {
                                self.emit_content(None, None, Some(thinking));
                            }
                        }
                        _ => {}
                    }
                }
            }

            "content_block_stop" => {
                // Flush completed tool call if present
                if let Some(index) = event.index {
                    if let Some(tool) = self.tool_buffers.remove(&index) {
                        self.emit_content(None, Some(vec![tool]), None);
                    }
                }
            }

            "message_delta" => {
                // Update usage with final token counts
                if let Some(usage) = event.usage {
                    self.usage = Some(Usage {
                        prompt_tokens: self.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                        completion_tokens: usage.output_tokens,
                        total_tokens: self.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0)
                            + usage.output_tokens,
                        completion_tokens_details: None,
                        prompt_tokens_details: self
                            .usage
                            .as_ref()
                            .and_then(|u| u.prompt_tokens_details.clone()),
                    });
                }
            }

            "message_stop" => {
                // Emit final response with usage
                if let Some(usage) = self.usage.take() {
                    self.results.push(Ok(StreamResponse {
                        choices: vec![StreamChoice {
                            delta: StreamDelta {
                                content: None,
                                tool_calls: None,
                                thinking: None,
                            },
                        }],
                        usage: Some(usage),
                    }));
                }
            }

            _ => {
                // Ignore unknown event types (ping, etc.)
            }
        }
    }

    /// Emit a StreamResponse with the given content.
    fn emit_content(
        &mut self,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        thinking: Option<String>,
    ) {
        self.results.push(Ok(StreamResponse {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content,
                    tool_calls,
                    thinking,
                },
            }],
            usage: None,
        }));
    }
}

/// Creates a structured streaming response from an Anthropic HTTP response.
fn create_anthropic_sse_struct_stream(
    response: reqwest::Response,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>> {
    use futures::StreamExt;

    let bytes_stream = response.bytes_stream();

    let stream = bytes_stream
        .scan(AnthropicSSEStreamParser::new(), |parser, chunk| {
            let results = match chunk {
                Ok(bytes) => {
                    parser.process_bytes(&bytes);
                    parser.drain_results()
                }
                Err(e) => vec![Err(LLMError::HttpError(e.to_string()))],
            };
            futures::future::ready(Some(results))
        })
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}

/// Parses a Server-Sent Events (SSE) chunk from Anthropic's streaming API.
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
///
/// # Returns
///
/// * `Ok(Some(String))` - Content token if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
fn parse_anthropic_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<AnthropicStreamResponse>(data) {
                Ok(response) => {
                    if response.response_type == "content_block_delta" {
                        if let Some(delta) = response.delta {
                            if let Some(text) = delta.text {
                                return Ok(Some(text));
                            }
                        }
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }
    Ok(None)
}
