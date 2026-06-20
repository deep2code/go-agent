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
// Anthropic API 常量
// ============================================================================

const (
	anthropicAPIURL  = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
	anthropicTimeout = 10 * time.Minute
)

// ============================================================================
// AnthropicProvider
// ============================================================================

// AnthropicProvider 实现 Provider 接口，对接 Anthropic Messages API。
type AnthropicProvider struct {
	apiKey     string
	httpClient *http.Client
	httpLogger *log.HTTPLogger // 可选：HTTP 请求/响应文件日志
}

// NewAnthropicProvider 创建 Anthropic Provider。
func NewAnthropicProvider(apiKey string) *AnthropicProvider {
	return &AnthropicProvider{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: anthropicTimeout,
		},
	}
}

// SetHTTPLogger 设置 HTTP 日志记录器（可选）。
func (p *AnthropicProvider) SetHTTPLogger(l *log.HTTPLogger) {
	p.httpLogger = l
}

// Name 返回 "anthropic"。
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

// StreamingChat 发送流式请求到 Anthropic Messages API。
func (p *AnthropicProvider) StreamingChat(
	ctx context.Context,
	req ChatRequest,
	onText func(text string),
) (*ChatResponse, error) {
	startTime := time.Now()

	msgReq := MessageRequest{
		Model:     req.Model,
		MaxTokens: req.MaxTokens,
		System:    req.System,
		Messages:  req.Messages,
		Tools:     req.Tools,
		Stream:    true,
	}

	body, err := json.Marshal(msgReq)
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	// ---- 日志：记录请求 ----
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		p.httpLogger.LogRequest("anthropic", "POST", anthropicAPIURL, req.RoundNumber, body)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, anthropicAPIURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("创建 HTTP 请求失败: %w", err)
	}

	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	httpReq.Header.Set("content-type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		elapsed := time.Since(startTime)
		if p.httpLogger != nil && p.httpLogger.Enabled() {
			p.httpLogger.LogError("anthropic", req.RoundNumber, elapsed, err.Error(), nil)
		}
		return nil, fmt.Errorf("HTTP 请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		elapsed := time.Since(startTime)
		if p.httpLogger != nil && p.httpLogger.Enabled() {
			p.httpLogger.LogError("anthropic", req.RoundNumber, elapsed,
				fmt.Sprintf("HTTP %d", resp.StatusCode), respBody)
		}
		return nil, p.parseError(resp.StatusCode, respBody)
	}

	// 使用 Anthropic SSE 解析器处理流式响应
	parser := &anthropicSSEParser{
		reader: strings.NewReader(""),
		onText: onText,
	}
	parser.reader = nil
	msgResp, err := parser.parseAnthropicStream(ctx, resp.Body, onText)
	elapsed := time.Since(startTime)

	if err != nil {
		if p.httpLogger != nil && p.httpLogger.Enabled() {
			p.httpLogger.LogError("anthropic", req.RoundNumber, elapsed, err.Error(), nil)
		}
		return nil, err
	}

	chatResp := &ChatResponse{
		ID:           msgResp.ID,
		Model:        msgResp.Model,
		FinishReason: mapAnthropicStopReason(msgResp.StopReason),
		Content:      msgResp.Content,
		Usage:        msgResp.Usage,
	}

	// ---- 日志：记录响应 ----
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		respJSON, _ := json.Marshal(chatResp)
		p.httpLogger.LogResponse("anthropic", req.RoundNumber, elapsed, respJSON)
	}

	return chatResp, nil
}

// ============================================================================
// Anthropic SSE 解析器（内嵌）
// ============================================================================

// anthropicSSEParser 解析 Anthropic SSE 流。
type anthropicSSEParser struct {
	reader *strings.Reader // 仅用于类型占位，实际使用 io.Reader
	onText func(text string)

	// 累积状态
	messageID  string
	model      string
	stopReason string
	usage      Usage

	// 内容块累积
	blocks []*anthropicAccumBlock
}

// anthropicAccumBlock 一个正在累积中的 Anthropic content block。
type anthropicAccumBlock struct {
	Index int
	Type  string // "text" | "tool_use" | "thinking"

	textBuf    strings.Builder // 文本/思考累积
	id         string
	name       string
	inputJSON  strings.Builder // input_json_delta 拼接
}

// parseAnthropicStream 解析 Anthropic SSE 流。
func (p *anthropicSSEParser) parseAnthropicStream(
	ctx context.Context,
	reader io.Reader,
	onText func(text string),
) (*MessageResponse, error) {
	p.onText = onText

	// 使用 bufio 逐行读取
	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 256)
	var eventType string
	var dataLines []string

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// 读取一行
		line, err := readLine(reader, buf, tmp)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("读取 SSE 流出错: %w", err)
		}

		// 空行表示一个事件结束
		if line == "" {
			if len(dataLines) > 0 {
				if err := p.handleAnthropicEvent(eventType, dataLines); err != nil {
					return nil, err
				}
				eventType = ""
				dataLines = dataLines[:0]
			}
			continue
		}

		if after, found := strings.CutPrefix(line, "event: "); found {
			eventType = after
			continue
		}

		if after, found := strings.CutPrefix(line, "data: "); found {
			dataLines = append(dataLines, after)
			continue
		}
	}

	return p.buildAnthropicResponse(), nil
}

// handleAnthropicEvent 处理一个完整的 Anthropic SSE 事件。
func (p *anthropicSSEParser) handleAnthropicEvent(eventType string, dataLines []string) error {
	jsonData := strings.Join(dataLines, "")

	var event StreamEvent
	if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
		return fmt.Errorf("解析 SSE 事件失败: %w\nraw: %.200s", err, jsonData)
	}

	switch event.Type {
	case "message_start":
		if event.Message != nil {
			p.messageID = event.Message.ID
			p.model = event.Message.Model
		}
		if event.Usage != nil {
			p.usage.InputTokens = event.Usage.InputTokens
		}

	case "content_block_start":
		if event.ContentBlock == nil {
			break
		}
		idx := 0
		if event.Index != nil {
			idx = *event.Index
		}

		block := &anthropicAccumBlock{
			Index: idx,
			Type:  event.ContentBlock.Type,
		}
		if block.Type == "tool_use" {
			block.id = event.ContentBlock.ID
			block.name = event.ContentBlock.Name
		}

		for len(p.blocks) <= idx {
			p.blocks = append(p.blocks, nil)
		}
		p.blocks[idx] = block

	case "content_block_delta":
		if event.Delta == nil {
			break
		}
		idx := 0
		if event.Index != nil {
			idx = *event.Index
		}

		var delta ContentBlockDelta
		if err := json.Unmarshal(event.Delta, &delta); err != nil {
			return nil
		}

		if idx >= len(p.blocks) || p.blocks[idx] == nil {
			break
		}
		block := p.blocks[idx]

		switch delta.Type {
		case "text_delta":
			block.textBuf.WriteString(delta.Text)
			if p.onText != nil {
				p.onText(delta.Text)
			}
		case "thinking_delta":
			block.textBuf.WriteString(delta.Thinking)
		case "input_json_delta":
			block.inputJSON.WriteString(delta.PartialJSON)
		}

	case "content_block_stop":
		// 不需要额外处理

	case "message_delta":
		if event.Delta != nil {
			var msgDelta MessageDelta
			if err := json.Unmarshal(event.Delta, &msgDelta); err == nil {
				p.stopReason = msgDelta.StopReason
			}
		}
		if event.Usage != nil {
			p.usage.OutputTokens = event.Usage.OutputTokens
		}

	case "message_stop":
		// 流结束
	}
	return nil
}

// buildAnthropicResponse 从累积状态构建 MessageResponse。
func (p *anthropicSSEParser) buildAnthropicResponse() *MessageResponse {
	resp := &MessageResponse{
		ID:         p.messageID,
		Model:      p.model,
		StopReason: p.stopReason,
		Usage:      p.usage,
	}

	for _, block := range p.blocks {
		if block == nil {
			continue
		}

		switch block.Type {
		case "text":
			resp.Content = append(resp.Content, ContentBlock{
				Type: "text",
				Text: block.textBuf.String(),
			})
		case "thinking":
			resp.Content = append(resp.Content, ContentBlock{
				Type:     "thinking",
				Thinking: block.textBuf.String(),
			})
		case "tool_use":
			inputStr := block.inputJSON.String()
			var input map[string]any
			if inputStr != "" {
				json.Unmarshal([]byte(inputStr), &input)
			}
			if input == nil {
				input = make(map[string]any)
			}
			resp.Content = append(resp.Content, ContentBlock{
				Type:  "tool_use",
				ID:    block.id,
				Name:  block.name,
				Input: input,
			})
		}
	}

	return resp
}

// ============================================================================
// 辅助函数
// ============================================================================

// readLine 从 io.Reader 读取一行（处理 \r\n 和 \n）。
func readLine(r io.Reader, buf []byte, tmp []byte) (string, error) {
	buf = buf[:0]
	for {
		n, err := r.Read(tmp)
		for i := 0; i < n; i++ {
			b := tmp[i]
			if b == '\n' {
				return string(buf), nil
			}
			if b != '\r' {
				buf = append(buf, b)
			}
		}
		if err != nil {
			if err == io.EOF && len(buf) > 0 {
				return string(buf), io.EOF
			}
			return "", err
		}
	}
}

// mapAnthropicStopReason 映射 Anthropic stop_reason → 内部 FinishReason。
func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return FinishStop
	case "tool_use":
		return FinishToolCalls
	case "max_tokens":
		return FinishLength
	default:
		return FinishError
	}
}

// parseError 解析 Anthropic API 错误响应。
func (p *AnthropicProvider) parseError(statusCode int, body []byte) error {
	var errResp ErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil {
		return fmt.Errorf("API 错误 (HTTP %d): %s", statusCode, string(body))
	}

	apiErr := errResp.Error
	switch statusCode {
	case 401:
		return fmt.Errorf("认证失败 (401): %s — 请检查 ANTHROPIC_API_KEY 是否正确", apiErr.Message)
	case 429:
		return fmt.Errorf("请求频率限制 (429): %s — 请稍后重试", apiErr.Message)
	case 400:
		return fmt.Errorf("请求参数错误 (400): %s", apiErr.Message)
	case 500, 502, 503:
		return fmt.Errorf("服务器错误 (HTTP %d): %s — 请稍后重试", statusCode, apiErr.Message)
	default:
		return fmt.Errorf("API 错误 (HTTP %d): %s (%s)", statusCode, apiErr.Message, apiErr.Type)
	}
}
