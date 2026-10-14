// Package llm 提供与 Anthropic Messages API 交互的客户端和类型定义。
//
// 本文件定义了与 Anthropic API 通信所需的全部数据类型，
// 包括请求/响应结构体、消息模型、工具定义和 SSE 流事件。
//
// 参考文档: https://docs.anthropic.com/en/api/messages
package llm

import (
	"encoding/json"
	"fmt"
)

// ============================================================================
// 角色与消息
// ============================================================================

// Role 消息角色，对应 Anthropic API 的 user / assistant 角色。
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message 对话中的一条消息。
// user 消息包含用户的提问或 tool_result 块；
// assistant 消息包含 LLM 的文本回复或 tool_use 请求。
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ============================================================================
// ContentBlock — 消息内容的联合体
// ============================================================================

// ContentBlock 是一条消息的内容块，根据 Type 字段区分不同类型：
//   - "text":         纯文本内容
//   - "tool_use":     LLM 请求调用工具
//   - "tool_result":  工具执行结果（返回给 LLM）
//   - "thinking":     extended thinking 思考过程
//
// 使用 omitempty 确保每种变体只序列化其自身字段。
type ContentBlock struct {
	// --- 通用的类型标识 ---
	Type string `json:"type"`

	// --- text 块 ---
	Text string `json:"text,omitempty"`

	// --- thinking 块 ---
	Thinking string `json:"thinking,omitempty"`

	// --- tool_use 块（由 LLM 生成）---
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`

	// --- tool_result 块（由 agent 生成，返回给 LLM）---
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // 可以是 string 或 []ContentBlock
	IsError   *bool  `json:"is_error,omitempty"`

	// --- 仅内部使用，不参与序列化 ---
	// Partial 表示这个 tool_use 的参数 JSON 没有完整到达（通常因为输出撞上
	// max_tokens）。这样的调用不能被执行，也不应留在对话历史里。
	Partial bool `json:"-"`
}

// ============================================================================
// 工具定义 — 符合 Anthropic tool_use 的 JSON Schema 规范
// ============================================================================

// ToolDef 工具定义，遵循 JSON Schema 规范传给 LLM。
type ToolDef struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema InputSchema `json:"input_schema"`

	// RawSchema 非空时替代 InputSchema 原样下发。
	// 外部来源（如 MCP 服务器）的 JSON Schema 里有 enum、items、$ref 这些
	// InputSchema 表达不了的字段，折进去会把参数说明丢给模型看没了。
	RawSchema json.RawMessage `json:"-"`

	// CacheControl 只在请求前缀的最后一个工具上设置，标记"到这里为止可缓存"。
	// 由 Provider 在发送前填，工具自身不需要关心。
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// MarshalJSON 让 RawSchema 生效：它不参与结构体本身的字段序列化，
// 只有这里能把它替换进 input_schema。
func (t ToolDef) MarshalJSON() ([]byte, error) {
	if len(t.RawSchema) == 0 {
		type plainToolDef ToolDef
		return json.Marshal(plainToolDef(t))
	}
	if !json.Valid(t.RawSchema) {
		return nil, fmt.Errorf("工具 %s 的 input_schema 不是合法 JSON", t.Name)
	}
	return json.Marshal(struct {
		Name         string          `json:"name"`
		Description  string          `json:"description"`
		InputSchema  json.RawMessage `json:"input_schema"`
		CacheControl *CacheControl   `json:"cache_control,omitempty"`
	}{t.Name, t.Description, t.RawSchema, t.CacheControl})
}

// CacheControl 是 Anthropic prompt caching 的断点标记。
// Type 目前只有 "ephemeral"（5 分钟 TTL，命中后自动续期）。
type CacheControl struct {
	Type string `json:"type"`
}

// ephemeralCache 是唯一的缓存类型，构造函数避免各处写裸字符串。
func ephemeralCache() *CacheControl { return &CacheControl{Type: "ephemeral"} }

// InputSchema 工具的 JSON Schema，固定 type 为 "object"。
type InputSchema struct {
	Type       string              `json:"type"` // 固定为 "object"
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required,omitempty"`
}

// Property 工具参数属性的 JSON Schema 定义。
type Property struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// ============================================================================
// 请求 — Anthropic Messages API 请求体
// ============================================================================

// MessageRequest 一次 Messages API 调用的完整请求体。
type MessageRequest struct {
	Model     string        `json:"model"`
	MaxTokens int           `json:"max_tokens"`
	System    []SystemBlock `json:"system,omitempty"` // 块数组：字符串形式带不了 cache_control
	Messages  []Message     `json:"messages"`
	Tools     []ToolDef     `json:"tools,omitempty"`
	Stream    bool          `json:"stream,omitempty"`
}

// SystemBlock 系统提示词的一个内容块。
type SystemBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// ============================================================================
// 响应 — 非流式响应（也用于累积流式结果）
// ============================================================================

// MessageResponse Messages API 的响应体。
type MessageResponse struct {
	ID         string         `json:"id"`
	Model      string         `json:"model"`
	StopReason string         `json:"stop_reason"` // "end_turn" | "tool_use" | "max_tokens"
	Content    []ContentBlock `json:"content"`
	Usage      Usage          `json:"usage"`
}

// Usage token 用量统计。
//
// Anthropic 开启 prompt caching 后，InputTokens 只包含未命中缓存的那部分前缀，
// 缓存读/写分别记账——算总输入和算钱都得把三段加起来，不能只看 InputTokens。
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// Total 返回本次用量的总 token 数。
func (u Usage) Total() int { return u.PromptTokens() + u.OutputTokens }

// PromptTokens 返回本次请求发给模型的输入 token 总数（含缓存段）。
// 这是服务端的权威计数，本地的启发式估算要跟它对齐。
func (u Usage) PromptTokens() int {
	return u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
}

// ============================================================================
// 错误响应
// ============================================================================

// APIError Anthropic API 返回的错误结构。
type APIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// ErrorResponse 完整的错误响应体。
type ErrorResponse struct {
	Error APIError `json:"error"`
}

// ============================================================================
// SSE 流事件 — 用于解析 streaming 响应
// ============================================================================

// StreamEvent 单个 SSE 事件的 JSON 载荷。
//
// 关键事件类型及其结构：
//   - message_start:      {type, message: {id, model, ...}}
//   - content_block_start:{type, index, content_block: {type: "text"|"tool_use", ...}}
//   - content_block_delta:{type, index, delta: {type: "text_delta"|"input_json_delta", ...}}
//   - content_block_stop: {type, index}
//   - message_delta:      {type, delta: {stop_reason, ...}, usage: {output_tokens}}
//   - message_stop:       {type}
//
// 注意: delta 字段在 content_block_delta 和 message_delta 中结构不同，
// 使用 json.RawMessage 延迟解析。
type StreamEvent struct {
	Type         string          `json:"type"`
	Index        *int            `json:"index,omitempty"`
	Message      *StreamMessage  `json:"message,omitempty"`
	ContentBlock *ContentBlock   `json:"content_block,omitempty"`
	Delta        json.RawMessage `json:"delta,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
}

// StreamMessage message_start 事件中的 message 元信息。
type StreamMessage struct {
	ID    string `json:"id"`
	Type  string `json:"type"`
	Role  string `json:"role"`
	Model string `json:"model"`
}

// ============================================================================
// Delta 子类型 — 用于解析 StreamEvent.Delta
// ============================================================================

// ContentBlockDelta content_block_delta 事件中的 delta 载荷。
// 根据 type 字段区分：text_delta / input_json_delta / thinking_delta。
type ContentBlockDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
}

// MessageDelta message_delta 事件中的 delta 载荷。
type MessageDelta struct {
	StopReason   string `json:"stop_reason"`
	StopSequence string `json:"stop_sequence,omitempty"`
}
