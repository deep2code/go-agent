package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go-agent/log"
)

// ============================================================================
// DeepSeek API 常量
// ============================================================================

const (
	deepseekBaseURL = "https://api.deepseek.com/v1"
	deepseekTimeout = 10 * time.Minute
)

// ============================================================================
// DeepSeekProvider
// ============================================================================

// DeepSeekProvider 实现 Provider 接口，对接 DeepSeek API（OpenAI 兼容格式）。
type DeepSeekProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	httpLogger *log.HTTPLogger // 可选：HTTP 请求/响应文件日志
	retry      Retry           // 限流/过载/网络抖动的重试策略
}

// NewDeepSeekProvider 创建 DeepSeek Provider。
func NewDeepSeekProvider(apiKey string) *DeepSeekProvider {
	return NewDeepSeekProviderWithBaseURL(deepseekBaseURL, apiKey)
}

// NewDeepSeekProviderWithBaseURL 创建指向自定义 OpenAI 兼容端点的 Provider。
// baseURL 用于自建网关或其他厂商的服务地址。
func NewDeepSeekProviderWithBaseURL(baseURL, apiKey string) *DeepSeekProvider {
	return &DeepSeekProvider{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: deepseekTimeout,
		},
		retry: DefaultRetry(),
	}
}

// SetHTTPLogger 设置 HTTP 日志记录器（可选）。
// 设置后，每次 API 调用的完整请求和响应体将写入日志文件。
func (p *DeepSeekProvider) SetHTTPLogger(l *log.HTTPLogger) {
	p.httpLogger = l
}

// SetRetry 覆盖重试策略。
func (p *DeepSeekProvider) SetRetry(r Retry) {
	p.retry = r
}

// Name 返回 "deepseek"。
func (p *DeepSeekProvider) Name() string {
	return "deepseek"
}

// StreamingChat 发送流式请求到 DeepSeek Chat Completions API。
//
// 限流、过载与网络抖动会按 p.retry 退避重试；一旦已向用户吐出增量就不再重试。
func (p *DeepSeekProvider) StreamingChat(
	ctx context.Context,
	req ChatRequest,
	onText func(text string),
) (*ChatResponse, error) {
	body, err := json.Marshal(p.toOpenAIRequest(req))
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	url := p.baseURL + "/chat/completions"
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		p.httpLogger.LogRequest("deepseek", "POST", url, req.RoundNumber, body)
	}

	var chatResp *ChatResponse
	var streamed bool

	retryErr := p.retry.do(ctx, "deepseek", req.RoundNumber,
		func() bool { return streamed },
		func() error {
			streamed = false
			startTime := time.Now()

			httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
			if err != nil {
				return fmt.Errorf("创建 HTTP 请求失败: %w", err)
			}
			httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("Accept", "text/event-stream")

			resp, err := p.httpClient.Do(httpReq)
			if err != nil {
				p.logError(req.RoundNumber, time.Since(startTime), err.Error(), nil)
				return transportError(err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody, _ := io.ReadAll(resp.Body)
				p.logError(req.RoundNumber, time.Since(startTime),
					fmt.Sprintf("HTTP %d", resp.StatusCode), respBody)
				return p.parseError(resp.StatusCode, respBody, retryAfter(resp.Header))
			}

			parser := newOpenAISSEParser(resp.Body, func(text string) {
				streamed = true
				if onText != nil {
					onText(text)
				}
			})
			out, err := parser.parse(ctx)
			elapsed := time.Since(startTime)
			if err != nil {
				p.logError(req.RoundNumber, elapsed, err.Error(), nil)
				return streamError(err)
			}

			chatResp = out

			if p.httpLogger != nil && p.httpLogger.Enabled() {
				respJSON, _ := json.Marshal(chatResp)
				p.httpLogger.LogResponse("deepseek", req.RoundNumber, elapsed, respJSON)
			}
			return nil
		})
	if retryErr != nil {
		return nil, retryErr
	}
	return chatResp, nil
}

// logError 把失败写入 HTTP 日志（若已启用）。
func (p *DeepSeekProvider) logError(round int, elapsed time.Duration, msg string, body []byte) {
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		p.httpLogger.LogError("deepseek", round, elapsed, msg, body)
	}
}

// ============================================================================
// 内部类型 → OpenAI 格式转换
// ============================================================================

type openAIChatRequest struct {
	Model         string           `json:"model"`
	Messages      []openAIMessage  `json:"messages"`
	Tools         []openAIToolDef  `json:"tools,omitempty"`
	Stream        bool             `json:"stream"`
	StreamOptions *openAIStreamOpt `json:"stream_options,omitempty"`
	MaxTokens     int              `json:"max_tokens,omitempty"`
}

// openAIStreamOpt 让服务端在流的最后一个 chunk 里带上用量统计，
// 否则流式调用拿不到 token 数。
type openAIStreamOpt struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIToolDef struct {
	Type     string            `json:"type"`
	Function openAIFunctionDef `json:"function"`
}

type openAIFunctionDef struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  InputSchema `json:"parameters"`
}

type openAIErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// toOpenAIRequest 将内部 ChatRequest 转换为 OpenAI 格式。
func (p *DeepSeekProvider) toOpenAIRequest(req ChatRequest) openAIChatRequest {
	oaReq := openAIChatRequest{
		Model:         req.Model,
		Stream:        true,
		StreamOptions: &openAIStreamOpt{IncludeUsage: true},
		MaxTokens:     req.MaxTokens,
	}

	if req.System != "" {
		oaReq.Messages = append(oaReq.Messages, openAIMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	for _, msg := range req.Messages {
		oaMsgs := p.convertMessage(msg)
		oaReq.Messages = append(oaReq.Messages, oaMsgs...)
	}

	for _, tool := range req.Tools {
		oaReq.Tools = append(oaReq.Tools, openAIToolDef{
			Type: "function",
			Function: openAIFunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}

	return oaReq
}

func (p *DeepSeekProvider) convertMessage(msg Message) []openAIMessage {
	switch msg.Role {
	case RoleUser:
		return p.convertUserMessage(msg)
	case RoleAssistant:
		return p.convertAssistantMessage(msg)
	default:
		return nil
	}
}

func (p *DeepSeekProvider) convertUserMessage(msg Message) []openAIMessage {
	var result []openAIMessage
	var textParts []string

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_result":
			content := ""
			if s, ok := block.Content.(string); ok {
				content = s
			} else {
				content = fmt.Sprintf("%v", block.Content)
			}
			result = append(result, openAIMessage{
				Role:       "tool",
				ToolCallID: block.ToolUseID,
				Content:    content,
			})
		}
	}

	if len(textParts) > 0 {
		result = append(result, openAIMessage{
			Role:    "user",
			Content: strings.Join(textParts, "\n"),
		})
	}

	return result
}

func (p *DeepSeekProvider) convertAssistantMessage(msg Message) []openAIMessage {
	oaMsg := openAIMessage{Role: "assistant"}

	var textParts []string
	var toolCalls []openAIToolCall

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_use":
			argsJSON, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, openAIToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openAIFunctionCall{
					Name:      block.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	if len(textParts) > 0 {
		oaMsg.Content = strings.Join(textParts, "\n")
	}
	if len(toolCalls) > 0 {
		oaMsg.ToolCalls = toolCalls
	}

	return []openAIMessage{oaMsg}
}

// ============================================================================
// 错误处理
// ============================================================================

// parseError 解析 OpenAI 兼容的错误响应，并标注是否可重试。
func (p *DeepSeekProvider) parseError(statusCode int, body []byte, hint time.Duration) error {
	// 非 JSON 响应体（网关错误页、纯文本）也要走下面的状态码分支
	var errResp openAIErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error.Message == "" {
		errResp.Error.Message = truncateBody(body)
	}

	e := errResp.Error
	msg := e.Message
	if msg == "" {
		msg = e.Type
	}
	switch statusCode {
	case 401:
		return apiError(statusCode, body, fmt.Sprintf("认证失败 (401): %s — 请检查 DEEPSEEK_API_KEY 是否正确", msg), hint)
	case 400:
		return apiError(statusCode, body, fmt.Sprintf("请求参数错误 (400): %s", msg), hint)
	case 402:
		return apiError(statusCode, body, fmt.Sprintf("余额不足 (402): %s", msg), hint)
	case 429:
		return apiError(statusCode, body, fmt.Sprintf("请求频率限制 (429): %s", msg), hint)
	case 500, 502, 503, 504:
		return apiError(statusCode, body, fmt.Sprintf("服务器错误 (HTTP %d): %s", statusCode, msg), hint)
	default:
		return apiError(statusCode, body, fmt.Sprintf("API 错误 (HTTP %d): %s (%s)", statusCode, msg, e.Code), hint)
	}
}
