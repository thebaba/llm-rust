//! Ollama API client implementation for chat and completion functionality.
//!
//! This module provides integration with Ollama's local LLM server through its API.

use std::pin::Pin;

use crate::{
    builder::LLMBackend,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
        Tool,
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
use base64::{self, Engine};
use chrono::{DateTime, Utc};
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for interacting with Ollama's API.
///
/// Provides methods for chat and completion requests using Ollama's models.
pub struct Ollama {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    /// Available tools for function calling
    pub tools: Option<Vec<Tool>>,
    client: Client,
}

/// Request payload for Ollama's chat API endpoint.
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: String,
    messages: Vec<OllamaChatMessage<'a>>,
    stream: bool,
    options: Option<OllamaOptions>,
    format: Option<OllamaResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
}

#[derive(Serialize)]
struct OllamaOptions {
    top_p: Option<f32>,
    top_k: Option<u32>,
}

/// Individual message in an Ollama chat conversation.
#[derive(Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

impl<'a> From<&'a ChatMessage> for OllamaChatMessage<'a> {
    fn from(msg: &'a ChatMessage) -> Self {
        Self {
            role: match msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            },
            content: &msg.content,
            images: match &msg.message_type {
                MessageType::Image((_mime, data)) => {
                    Some(vec![base64::engine::general_purpose::STANDARD.encode(data)])
                }
                _ => None,
            },
        }
    }
}

/// Response from Ollama's API endpoints.
#[derive(Deserialize, Debug)]
struct OllamaResponse {
    content: Option<String>,
    response: Option<String>,
    message: Option<OllamaChatResponseMessage>,
}

impl std::fmt::Display for OllamaResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let empty = String::new();
        let text = self
            .content
            .as_ref()
            .or(self.response.as_ref())
            .or(self.message.as_ref().map(|m| &m.content))
            .unwrap_or(&empty);

        // Write tool calls if present
        if let Some(message) = &self.message {
            if let Some(tool_calls) = &message.tool_calls {
                for tc in tool_calls {
                    writeln!(
                        f,
                        "{{\"name\": \"{}\", \"arguments\": {}}}",
                        tc.function.name,
                        serde_json::to_string_pretty(&tc.function.arguments).unwrap_or_default()
                    )?;
                }
            }
        }

        write!(f, "{text}")
    }
}

impl ChatResponse for OllamaResponse {
    fn text(&self) -> Option<String> {
        self.content
            .as_ref()
            .or(self.response.as_ref())
            .or(self.message.as_ref().map(|m| &m.content))
            .map(|s| s.to_string())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.message.as_ref().and_then(|msg| {
            msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: format!("call_{}", tc.function.name),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_default(),
                        },
                    })
                    .collect()
            })
        })
    }
}

/// Message content within an Ollama chat API response.
#[derive(Deserialize, Debug)]
struct OllamaChatResponseMessage {
    content: String,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Request payload for Ollama's generate API endpoint.
#[derive(Serialize)]
struct OllamaGenerateRequest<'a> {
    model: String,
    prompt: &'a str,
    raw: bool,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(untagged)]
enum OllamaResponseType {
    #[serde(rename = "json")]
    Json,
    StructuredOutput(Value),
}

#[derive(Deserialize, Debug, Serialize)]
struct OllamaResponseFormat {
    #[serde(flatten)]
    format: OllamaResponseType,
}

/// Ollama's tool format
#[derive(Serialize, Debug)]
struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,

    pub function: OllamaFunctionTool,
}

#[derive(Serialize, Debug)]
struct OllamaFunctionTool {
    /// Name of the tool
    name: String,
    /// Description of what the tool does
    description: String,
    /// Parameters for the tool
    parameters: OllamaParameters,
}

impl From<&crate::chat::Tool> for OllamaTool {
    fn from(tool: &crate::chat::Tool) -> Self {
        let properties_value = tool
            .function
            .parameters
            .get("properties")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        let required_fields = tool
            .function
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        OllamaTool {
            tool_type: "function".to_owned(),
            function: OllamaFunctionTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: OllamaParameters {
                    schema_type: "object".to_string(),
                    properties: properties_value,
                    required: required_fields,
                },
            },
        }
    }
}

/// Ollama's parameters schema
#[derive(Serialize, Debug)]
struct OllamaParameters {
    /// The type of parameters object (usually "object")
    #[serde(rename = "type")]
    schema_type: String,
    /// Map of parameter names to their properties
    properties: Value,
    /// List of required parameter names
    required: Vec<String>,
}

/// Ollama's tool call response
#[derive(Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Deserialize, Debug)]
struct OllamaFunctionCall {
    /// Name of the tool that was called
    name: String,
    /// Arguments provided to the tool
    arguments: Value,
}

/// Ollama streaming response structure for NDJSON parsing
#[derive(Deserialize, Debug)]
struct OllamaStreamResponse {
    /// Message content and tool calls
    message: Option<OllamaStreamMessage>,
    /// Whether this is the final chunk
    done: bool,
    /// Prompt tokens used (present in final chunk)
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    /// Completion tokens used (present in final chunk)
    #[serde(default)]
    eval_count: Option<u32>,
}

/// Ollama streaming message structure
#[derive(Deserialize, Debug)]
struct OllamaStreamMessage {
    /// Text content
    #[serde(default)]
    content: String,
    /// Tool calls (may be present in streaming responses)
    tool_calls: Option<Vec<OllamaToolCall>>,
}

use crate::chat::{StreamChoice, StreamDelta, StreamResponse, Usage};

/// Stateful parser for Ollama's NDJSON streaming with tool call accumulation
struct OllamaNDJSONStreamParser {
    /// Buffer for incomplete JSON lines split across chunks
    line_buffer: String,
    /// Buffer for incomplete UTF-8 sequences
    utf8_buffer: Vec<u8>,
    /// Accumulated tool call being built
    tool_buffer: ToolCall,
    /// Accumulated usage metadata (if any)
    usage: Option<Usage>,
    /// Results ready to be emitted
    results: Vec<Result<StreamResponse, LLMError>>,
}

impl OllamaNDJSONStreamParser {
    fn new() -> Self {
        Self {
            line_buffer: String::new(),
            utf8_buffer: Vec::new(),
            tool_buffer: ToolCall {
                id: String::new(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: String::new(),
                    arguments: String::new(),
                },
            },
            usage: None,
            results: Vec::new(),
        }
    }

    /// Push the current tool_buffer as a StreamResponse and reset it
    fn push_tool_call(&mut self) {
        if !self.tool_buffer.function.name.is_empty() {
            self.results.push(Ok(StreamResponse {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: None,
                        tool_calls: Some(vec![self.tool_buffer.clone()]),
                        thinking: None,
                    },
                }],
                usage: None,
            }));
        }
        self.tool_buffer = ToolCall {
            id: String::new(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: String::new(),
                arguments: String::new(),
            },
        };
    }

    /// Process raw bytes, handling UTF-8 boundaries and line splitting
    fn process_bytes(&mut self, bytes: &[u8]) {
        // Append to UTF-8 buffer and try to decode
        self.utf8_buffer.extend_from_slice(bytes);

        match String::from_utf8(std::mem::take(&mut self.utf8_buffer)) {
            Ok(text) => {
                self.line_buffer.push_str(&text);
            }
            Err(e) => {
                let valid_up_to = e.utf8_error().valid_up_to();
                let bytes = e.into_bytes();
                if valid_up_to > 0 {
                    // Safe because we know this portion is valid UTF-8
                    let valid = unsafe { std::str::from_utf8_unchecked(&bytes[..valid_up_to]) };
                    self.line_buffer.push_str(valid);
                }
                // Keep the incomplete bytes for next chunk
                self.utf8_buffer = bytes[valid_up_to..].to_vec();
            }
        }

        // Process complete lines (NDJSON uses single newlines)
        while let Some(pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..pos].to_string();
            self.line_buffer.drain(..pos + 1);

            let trimmed = line.trim();
            if !trimmed.is_empty() {
                self.parse_line(trimmed);
            }
        }
    }

    /// Parse a single NDJSON line
    fn parse_line(&mut self, line: &str) {
        let response: OllamaStreamResponse = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                log::debug!("Failed to parse Ollama stream line: {}", e);
                return;
            }
        };

        // Handle final chunk
        if response.done {
            // Flush any accumulated tool call
            self.push_tool_call();

            // Extract usage from final chunk if available
            if let (Some(prompt_tokens), Some(completion_tokens)) =
                (response.prompt_eval_count, response.eval_count)
            {
                self.usage = Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                });
            }

            // Emit usage if present
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
            return;
        }

        // Process message content
        if let Some(message) = response.message {
            let content = if message.content.is_empty() {
                None
            } else {
                Some(message.content)
            };

            // Handle tool calls with accumulation
            if let Some(tool_calls) = message.tool_calls {
                for tc in tool_calls {
                    // If we see a new function name, flush the previous tool call
                    if !tc.function.name.is_empty() {
                        self.push_tool_call();
                        self.tool_buffer.function.name = tc.function.name.clone();
                        self.tool_buffer.id = generate_tool_call_id();
                    }
                    // Accumulate arguments (may span multiple chunks)
                    let args_str = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                    // Remove the outer quotes if it's a string, or keep as-is for objects
                    let args_clean = if args_str.starts_with('"') && args_str.ends_with('"') {
                        // It's a JSON string, unescape it
                        tc.function.arguments.as_str().unwrap_or(&args_str).to_string()
                    } else {
                        args_str
                    };
                    self.tool_buffer.function.arguments.push_str(&args_clean);
                }
            }

            // Emit content if present
            if content.is_some() {
                self.results.push(Ok(StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content,
                            tool_calls: None,
                            thinking: None,
                        },
                    }],
                    usage: None,
                }));
            }
        }
    }
}

/// Generate a simple unique ID for tool calls
fn generate_tool_call_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("call_{:x}{:x}", d.as_secs(), d.subsec_nanos())
}

impl Ollama {
    /// Creates a new Ollama client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the Ollama server
    /// * `api_key` - Optional API key for authentication
    /// * `model` - Model name to use (defaults to "llama3.1")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `json_schema` - JSON schema for structured output
    /// * `tools` - Function tools that the model can use
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            base_url: base_url.into(),
            api_key,
            model: model.unwrap_or("llama3.1".to_string()),
            temperature,
            max_tokens,
            timeout_seconds,
            system,
            top_p,
            top_k,
            json_schema,
            tools,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }

    fn make_chat_request<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        tools: Option<&'a [Tool]>,
        stream: bool,
    ) -> OllamaChatRequest<'a> {
        let mut chat_messages: Vec<OllamaChatMessage> =
            messages.iter().map(OllamaChatMessage::from).collect();

        if let Some(system) = &self.system {
            chat_messages.insert(
                0,
                OllamaChatMessage {
                    role: "system",
                    content: system,
                    images: None,
                },
            );
        }

        // Convert tools to Ollama format if provided
        let ollama_tools = tools.map(|t| t.iter().map(OllamaTool::from).collect());

        // Ollama doesn't require the "name" field in the schema, so we just use the schema itself
        let format = if let Some(schema) = &self.json_schema {
            schema.schema.as_ref().map(|schema| OllamaResponseFormat {
                format: OllamaResponseType::StructuredOutput(schema.clone()),
            })
        } else {
            None
        };

        OllamaChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            stream,
            options: Some(OllamaOptions {
                top_p: self.top_p,
                top_k: self.top_k,
            }),
            format,
            tools: ollama_tools,
        }
    }
}

#[async_trait]
impl ChatProvider for Ollama {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let req_body = self.make_chat_request(messages, tools, false);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Ollama request payload (tools): {}", json);
            }
        }

        let url = format!("{}/api/chat", self.base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Ollama HTTP status (tools): {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp = resp.json::<OllamaResponse>().await?;

        Ok(Box::new(json_resp))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        use futures::StreamExt;

        let struct_stream = self.chat_stream_struct(messages).await?;
        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(resp) => resp
                    .choices
                    .first()
                    .and_then(|c| c.delta.content.clone())
                    .filter(|s| !s.is_empty())
                    .map(Ok),
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::pin(content_stream))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        // Pass tools to the request with stream=true
        let req_body = self.make_chat_request(messages, self.tools.as_deref(), true);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Ollama request payload (stream_struct): {}", json);
            }
        }

        let url = format!("{}/api/chat", self.base_url);
        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;
        log::debug!("Ollama HTTP status (stream_struct): {}", resp.status());

        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Ollama API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(create_ollama_ndjson_struct_stream(resp))
    }
}

#[async_trait]
impl CompletionProvider for Ollama {
    /// Sends a completion request to Ollama's API.
    ///
    /// # Arguments
    ///
    /// * `req` - The completion request containing the prompt
    ///
    /// # Returns
    ///
    /// The completion response containing the generated text or an error
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/api/generate", self.base_url);

        let req_body = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: &req.prompt,
            raw: true,
            stream: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?
            .error_for_status()?;
        let json_resp: OllamaResponse = resp.json().await?;

        if let Some(answer) = json_resp.response.or(json_resp.content) {
            Ok(CompletionResponse { text: answer })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by Ollama".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for Ollama {
    async fn embed(&self, text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/api/embed", self.base_url);

        let body = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: text,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OllamaEmbeddingResponse = resp.json().await?;
        Ok(json_resp.embeddings)
    }
}

#[async_trait]
impl SpeechToTextProvider for Ollama {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Ollama does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OllamaModelEntry {
    pub name: String,
    pub size: Option<u64>,
    pub digest: Option<String>,
    pub details: Option<OllamaModelDetails>,
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OllamaModelDetails {
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

impl ModelListRawEntry for OllamaModelEntry {
    fn get_id(&self) -> String {
        self.name.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        // Ollama doesn't provide creation dates
        DateTime::<Utc>::UNIX_EPOCH
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OllamaModelListResponse {
    pub models: Vec<OllamaModelEntry>,
}

impl ModelListResponse for OllamaModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.models
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::Ollama
    }
}

#[async_trait]
impl ModelsProvider for Ollama {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let url = format!("{}/api/tags", self.base_url);

        let mut request = self.client.get(&url);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;
        let result: OllamaModelListResponse = resp.json().await?;
        Ok(Box::new(result))
    }
}

impl crate::LLMProvider for Ollama {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl TextToSpeechProvider for Ollama {}

/// Creates a structured NDJSON stream that returns `StreamResponse` objects
/// with tool call accumulation support for Ollama's streaming API
fn create_ollama_ndjson_struct_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>> {
    use futures::StreamExt;

    let bytes_stream = response.bytes_stream();

    let stream = bytes_stream
        .scan(OllamaNDJSONStreamParser::new(), |parser, chunk| {
            let results = match chunk {
                Ok(bytes) => {
                    parser.process_bytes(&bytes);
                    parser.results.drain(..).collect::<Vec<_>>()
                }
                Err(e) => vec![Err(LLMError::HttpError(e.to_string()))],
            };
            futures::future::ready(Some(results))
        })
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}
