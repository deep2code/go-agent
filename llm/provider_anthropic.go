package llm

import (
	"bufio"
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
	endpoint   string // Messages API 地址；自建网关可覆盖
	httpClient *http.Client
	httpLogger *log.HTTPLogger // 可选：HTTP 请求/响应文件日志
	retry      Retry           // 限流/过载/网络抖动的重试策略
}

// NewAnthropicProvider 创建 Anthropic Provider。
func NewAnthropicProvider(apiKey string) *AnthropicProvider {
	return NewAnthropicProviderWithBaseURL(anthropicAPIURL, apiKey)
}

// NewAnthropicProviderWithBaseURL 创建指向自定义 Messages 兼容端点的 Provider。
func NewAnthropicProviderWithBaseURL(endpoint, apiKey string) *AnthropicProvider {
	return &AnthropicProvider{
		apiKey:   apiKey,
		endpoint: endpoint,
		httpClient: &http.Client{
			Timeout: anthropicTimeout,
		},
		retry: DefaultRetry(),
	}
}

// SetHTTPLogger 设置 HTTP 日志记录器（可选）。
func (p *AnthropicProvider) SetHTTPLogger(l *log.HTTPLogger) {
	p.httpLogger = l
}

// SetRetry 覆盖重试策略。
func (p *AnthropicProvider) SetRetry(r Retry) {
	p.retry = r
}

// Name 返回 "anthropic"。
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

// StreamingChat 发送流式请求到 Anthropic Messages API。
//
// 限流、过载与网络抖动会按 p.retry 退避重试；一旦已经向 onText 吐出增量，
// 就不再重试，避免终端上出现重复内容。
func (p *AnthropicProvider) StreamingChat(
	ctx context.Context,
	req ChatRequest,
	onText func(text string),
) (*ChatResponse, error) {
	body, err := json.Marshal(buildAnthropicRequest(req))
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	if p.httpLogger != nil && p.httpLogger.Enabled() {
		p.httpLogger.LogRequest("anthropic", "POST", p.endpoint, req.RoundNumber, body)
	}

	var chatResp *ChatResponse
	var streamed bool

	retryErr := p.retry.do(ctx, "anthropic", req.RoundNumber,
		func() bool { return streamed },
		func() error {
			streamed = false
			startTime := time.Now()

			httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, bytes.NewReader(body))
			if err != nil {
				return fmt.Errorf("创建 HTTP 请求失败: %w", err)
			}
			httpReq.Header.Set("x-api-key", p.apiKey)
			httpReq.Header.Set("anthropic-version", anthropicVersion)
			httpReq.Header.Set("content-type", "application/json")

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

			msgResp, err := newAnthropicSSEParser().parseAnthropicStream(ctx, resp.Body, func(text string) {
				streamed = true
				if onText != nil {
					onText(text)
				}
			})
			elapsed := time.Since(startTime)
			if err != nil {
				p.logError(req.RoundNumber, elapsed, err.Error(), nil)
				return streamError(err)
			}

			chatResp = &ChatResponse{
				ID:           msgResp.ID,
				Model:        msgResp.Model,
				FinishReason: mapAnthropicStopReason(msgResp.StopReason),
				Content:      msgResp.Content,
				Usage:        msgResp.Usage,
			}

			if p.httpLogger != nil && p.httpLogger.Enabled() {
				respJSON, _ := json.Marshal(chatResp)
				p.httpLogger.LogResponse("anthropic", req.RoundNumber, elapsed, respJSON)
			}
			return nil
		})
	if retryErr != nil {
		return nil, retryErr
	}
	return chatResp, nil
}

// logError 把失败写入 HTTP 日志（若已启用）。
func (p *AnthropicProvider) logError(round int, elapsed time.Duration, msg string, body []byte) {
	if p.httpLogger != nil && p.httpLogger.Enabled() {
		p.httpLogger.LogError("anthropic", round, elapsed, msg, body)
	}
}

// minCachePrefixChars 是值得一打缓存断点的最短请求前缀。
//
// 服务端只缓存达到长度门槛的前缀（Sonnet/Opus 约 1024 token 起，按英文量级
// 折算成字符），而缓存写入按 1.25 倍计价。前缀不够长时断点换不来任何命中，
// 等于白付溢价——摘要那种一次性短请求就是这种情形。
const minCachePrefixChars = 4000

// buildAnthropicRequest 把通用请求转成 Messages API 的 wire 格式。
func buildAnthropicRequest(req ChatRequest) MessageRequest {
	wire := MessageRequest{
		Model:     req.Model,
		MaxTokens: req.MaxTokens,
		Messages:  req.Messages,
		Tools:     req.Tools,
		Stream:    true,
	}
	if req.System != "" {
		wire.System = []SystemBlock{{Type: "text", Text: req.System}}
	}
	if prefixChars(wire) < minCachePrefixChars {
		return wire
	}

	// 断点只打一个，放在前缀的最后一项上：它把 system + tools 整段一起缓存。
	// 打两个断点会让 system 被写两遍，多付一次 1.25 倍。
	if n := len(wire.Tools); n > 0 {
		// 复制一份再改：调用方传进来的 Tools 还被上层用于估算和后续轮次，
		// 就地写入会让 cache_control 泄漏到不该带的请求里。
		tools := make([]ToolDef, n)
		copy(tools, req.Tools)
		tools[n-1].CacheControl = ephemeralCache()
		wire.Tools = tools
	} else if n := len(wire.System); n > 0 {
		wire.System[n-1].CacheControl = ephemeralCache()
	}
	return wire
}

// prefixChars 估算请求前缀（system + tools）的规模。
func prefixChars(req MessageRequest) int {
	total := 0
	for _, b := range req.System {
		total += len(b.Text)
	}
	tools, err := json.Marshal(req.Tools)
	if err == nil {
		total += len(tools)
	}
	return total
}

// ============================================================================
// Anthropic SSE 解析器（内嵌）
// ============================================================================

// anthropicSSEParser 解析 Anthropic SSE 流。
type anthropicSSEParser struct {
	onText func(text string)

	// 累积状态
	messageID  string
	model      string
	stopReason string
	usage      Usage

	// 内容块累积
	blocks []*anthropicAccumBlock

	// sawMessageStop 记录流是否正常收尾；未到 message_stop 即为提前截断。
	sawMessageStop bool
}

// anthropicAccumBlock 一个正在累积中的 Anthropic content block。
type anthropicAccumBlock struct {
	Index int
	Type  string // "text" | "tool_use" | "thinking"

	textBuf   strings.Builder // 文本/思考累积
	id        string
	name      string
	inputJSON strings.Builder // input_json_delta 拼接
	partial   bool            // 参数 JSON 不完整
}

// newAnthropicSSEParser 创建 Anthropic SSE 解析器。
func newAnthropicSSEParser() *anthropicSSEParser {
	return &anthropicSSEParser{}
}

// parseAnthropicStream 解析 Anthropic SSE 流。
func (p *anthropicSSEParser) parseAnthropicStream(
	ctx context.Context,
	reader io.Reader,
	onText func(text string),
) (*MessageResponse, error) {
	p.onText = onText

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	var eventType string
	var dataLines []string

	flush := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		err := p.handleAnthropicEvent(eventType, dataLines)
		eventType = ""
		dataLines = dataLines[:0]
		return err
	}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		line := scanner.Text()

		// 空行表示一个事件结束
		if line == "" {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}

		if after, found := strings.CutPrefix(line, "event:"); found {
			eventType = strings.TrimSpace(after)
			continue
		}

		if after, found := strings.CutPrefix(line, "data:"); found {
			dataLines = append(dataLines, strings.TrimPrefix(after, " "))
			continue
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("读取 SSE 流出错: %w", err)
	}

	// 流末尾可能没有空行，最后一个事件在此收尾
	if err := flush(); err != nil {
		return nil, err
	}

	if !p.sawMessageStop {
		return nil, fmt.Errorf("响应流在 message_stop 之前结束")
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
			// 缓存段（cache_creation / cache_read）只出现在 message_start，
			// 逐字段拷贝会把它们丢掉，剩下的 InputTokens 就成了偏小的假数。
			p.usage = *event.Usage
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
		if event.Index != nil && *event.Index < len(p.blocks) {
			if b := p.blocks[*event.Index]; b != nil && b.Type == "tool_use" {
				// 空参数视为合法（无参工具）；非空但解析失败说明 JSON 被截断。
				raw := b.inputJSON.String()
				var probe map[string]any
				if raw != "" && json.Unmarshal([]byte(raw), &probe) != nil {
					b.partial = true
				}
			}
		}

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
		p.sawMessageStop = true
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
				Type:    "tool_use",
				ID:      block.id,
				Name:    block.name,
				Input:   input,
				Partial: block.partial,
			})
		}
	}

	return resp
}

// ============================================================================
// 辅助函数
// ============================================================================

// mapAnthropicStopReason 映射 Anthropic stop_reason → 内部 FinishReason。
func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn", "refusal":
		return FinishStop
	case "tool_use":
		return FinishToolCalls
	case "max_tokens":
		return FinishLength
	case "":
		return FinishStop
	default:
		return FinishError
	}
}

// parseError 解析 Anthropic API 错误响应，并标注是否可重试。
func (p *AnthropicProvider) parseError(statusCode int, body []byte, hint time.Duration) error {
	// 响应体不是 JSON 时（网关错误页、纯文本）也要走下面的状态码分支，
	// 否则用户只看到一句 "API 错误"，分不清是限流还是服务挂了。
	var errResp ErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error.Message == "" {
		errResp.Error.Message = truncateBody(body)
	}

	msg := errResp.Error.Message
	switch statusCode {
	case 401:
		return apiError(statusCode, body, fmt.Sprintf("认证失败 (401): %s — 请检查 ANTHROPIC_API_KEY 是否正确", msg), hint)
	case 400:
		return apiError(statusCode, body, fmt.Sprintf("请求参数错误 (400): %s", msg), hint)
	case 429:
		return apiError(statusCode, body, fmt.Sprintf("请求频率限制 (429): %s", msg), hint)
	case statusOverloaded:
		return apiError(statusCode, body, fmt.Sprintf("服务过载 (529): %s", msg), hint)
	case 500, 502, 503, 504:
		return apiError(statusCode, body, fmt.Sprintf("服务器错误 (HTTP %d): %s", statusCode, msg), hint)
	default:
		return apiError(statusCode, body, fmt.Sprintf("API 错误 (HTTP %d): %s (%s)", statusCode, msg, errResp.Error.Type), hint)
	}
}

// truncateBody 用于把错误响应体放进日志/错误文案。
func truncateBody(body []byte) string {
	const max = 500
	if len(body) <= max {
		return string(body)
	}
	return string(body[:max]) + "..."
}
