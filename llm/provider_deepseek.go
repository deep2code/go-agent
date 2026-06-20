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
}

// NewDeepSeekProvider 创建 DeepSeek Provider。
func NewDeepSeekProvider(apiKey string) *DeepSeekProvider {
	return &DeepSeekProvider{
		apiKey:  apiKey,
		baseURL: deepseekBaseURL,
		httpClient: &http.Client{
			Timeout: deepseekTimeout,
		},
	}
}

// SetHTTPLogger 设置 HTTP 日志记录器（可选）。
// 设置后，每次 API 调用的完整请求和响应体将写入日志文件。
func (p *DeepSeekProvider) SetHTTPLogger(l *log.HTTPLogger) {
	p.httpLogger = l
}

// Name 返回 "deepseek"。
func (p *DeepSeekProvider) Name() string {
	return "deepseek"
}

// StreamingChat 发送流式请求到 DeepSeek Chat Completions API。
func (p *DeepSeekProvider) StreamingChat(
	ctx context.Context,
	req ChatRequest,
	onText func(text string),
) (*ChatResponse, error) {
	startTime := time.Now()

	// 1. 将内部类型转换为 OpenAI 格式
	oaReq := p.toOpenAIRequest(req)

	body, err := json.Marshal(oaReq)
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	// ---- 日志：记录请求 ----
	url := p.baseURL + "/chat/completions"
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		p.httpLogger.LogRequest("deepseek", "POST", url, req.RoundNumber, body)
	}

	// 2. 发送 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("创建 HTTP 请求失败: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		elapsed := time.Since(startTime)
		if p.httpLogger != nil && p.httpLogger.Enabled() {
			p.httpLogger.LogError("deepseek", req.RoundNumber, elapsed, err.Error(), nil)
		}
		return nil, fmt.Errorf("HTTP 请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		elapsed := time.Since(startTime)
		if p.httpLogger != nil && p.httpLogger.Enabled() {
			p.httpLogger.LogError("deepseek", req.RoundNumber, elapsed,
				fmt.Sprintf("HTTP %d", resp.StatusCode), respBody)
		}
		return nil, p.parseError(resp.StatusCode, respBody)
	}

	// 3. 使用 OpenAI SSE 解析器处理流式响应
	parser := newOpenAISSEParser(resp.Body, onText)
	chatResp, err := parser.parse(ctx)
	elapsed := time.Since(startTime)

	if err != nil {
		if p.httpLogger != nil && p.httpLogger.Enabled() {
			p.httpLogger.LogError("deepseek", req.RoundNumber, elapsed, err.Error(), nil)
		}
		return nil, err
	}

	// ---- 日志：记录响应 ----
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		respJSON, _ := json.Marshal(chatResp)
		p.httpLogger.LogResponse("deepseek", req.RoundNumber, elapsed, respJSON)
	}

	return chatResp, nil
}

// ============================================================================
// 内部类型 → OpenAI 格式转换
// ============================================================================

type openAIChatRequest struct {
	Model     string          `json:"model"`
	Messages  []openAIMessage `json:"messages"`
	Tools     []openAIToolDef `json:"tools,omitempty"`
	Stream    bool            `json:"stream"`
	MaxTokens int             `json:"max_tokens,omitempty"`
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
		Model:     req.Model,
		Stream:    true,
		MaxTokens: req.MaxTokens,
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

func (p *DeepSeekProvider) parseError(statusCode int, body []byte) error {
	var errResp openAIErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil {
		return fmt.Errorf("API 错误 (HTTP %d): %s", statusCode, string(body))
	}

	e := errResp.Error
	switch statusCode {
	case 401:
		return fmt.Errorf("认证失败 (401): %s — 请检查 DEEPSEEK_API_KEY 是否正确", e.Message)
	case 429:
		return fmt.Errorf("请求频率限制 (429): %s — 请稍后重试", e.Message)
	case 400:
		return fmt.Errorf("请求参数错误 (400): %s", e.Message)
	case 500, 502, 503:
		return fmt.Errorf("服务器错误 (HTTP %d): %s — 请稍后重试", statusCode, e.Message)
	default:
		return fmt.Errorf("API 错误 (HTTP %d): %s (%s)", statusCode, e.Message, e.Code)
	}
}
