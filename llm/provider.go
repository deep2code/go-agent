package llm

import (
	"context"

	"go-agent/log"
)

// ============================================================================
// Provider 接口
// ============================================================================

// Provider 抽象 LLM 后端接口。
// 不同厂商（Anthropic、DeepSeek、OpenAI 等）实现此接口，
// 将内部统一类型转换为各自 API 的 wire format。
type Provider interface {
	// Name 返回 Provider 标识（如 "anthropic"、"deepseek"）。
	Name() string

	// StreamingChat 发送流式聊天请求。
	//
	// 参数:
	//   - ctx: 上下文（取消可中止流式读取）
	//   - req: 通用请求体（与具体 API 格式无关）
	//   - onText: 每收到文本增量时调用的回调
	//
	// 返回完整的 ChatResponse（包含 content blocks、finish_reason、usage）。
	StreamingChat(ctx context.Context, req ChatRequest, onText func(text string)) (*ChatResponse, error)

	// SetHTTPLogger 把请求/响应写入日志文件（可选能力，传 nil 表示关闭）。
	SetHTTPLogger(l *log.HTTPLogger)
}

// ============================================================================
// Provider 可选能力
// ============================================================================

// RetrySetter 由支持配置重试策略的 Provider 实现。
type RetrySetter interface {
	SetRetry(r Retry)
}

// ============================================================================
// 通用请求 / 响应类型
// ============================================================================

// ChatRequest 通用聊天请求体，各 Provider 将其转换为具体 API 格式。
type ChatRequest struct {
	Model       string    // 模型 ID（如 "deepseek-v4-flash"、"claude-opus-4-8"）
	MaxTokens   int       // 最大输出 token 数
	System      string    // 系统提示词
	Messages    []Message // 完整对话历史（使用内部类型）
	Tools       []ToolDef // 可用工具列表
	RoundNumber int       // ReAct 循环轮次（用于日志记录）
}

// ChatResponse 通用聊天响应体，各 Provider 从 API 响应转换为此格式。
type ChatResponse struct {
	ID           string         // 响应的唯一标识
	Model        string         // 实际使用的模型
	FinishReason string         // 停止原因: "stop" | "tool_calls" | "length" | "error"
	Content      []ContentBlock // 内容块（text + tool_use）
	Usage        Usage          // token 用量统计
}

// ============================================================================
// 统一的 FinishReason 常量
// ============================================================================

const (
	FinishStop      = "stop"       // 对话自然结束
	FinishToolCalls = "tool_calls" // 需要执行工具调用
	FinishLength    = "length"     // 输出被 token 上限截断
	FinishError     = "error"      // 发生错误
)
