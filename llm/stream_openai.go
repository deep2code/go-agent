package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

// ============================================================================
// OpenAI 格式 SSE 解析器
// ============================================================================

// openaiSSEParser 解析 OpenAI 兼容格式的 SSE 流式响应。
//
// SSE 事件格式:
//
//	data: {"id":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
//	data: {"id":"...","choices":[{"index":0,"delta":{"tool_calls":[...]},"finish_reason":null}]}
//	data: [DONE]
//
// 工具调用参数被拆成多个 JSON 片段流式传输，需要在客户端按 index 分组拼接。
type openaiSSEParser struct {
	reader *bufio.Reader
	onText func(text string)

	// 累积状态
	messageID    string
	model        string
	finishReason string
	sawDone      bool
	usage        Usage

	// 内容累积
	textBuf strings.Builder

	// 工具调用累积: index → toolCallAccum
	toolCalls map[int]*toolCallAccum
}

// toolCallAccum 一个正在累积中的工具调用。
type toolCallAccum struct {
	ID        string
	Type      string
	Name      string
	Arguments strings.Builder
}

// newOpenAISSEParser 创建 OpenAI SSE 解析器。
func newOpenAISSEParser(reader io.Reader, onText func(text string)) *openaiSSEParser {
	return &openaiSSEParser{
		reader:    bufio.NewReader(reader),
		onText:    onText,
		toolCalls: make(map[int]*toolCallAccum),
	}
}

// parse 解析完整的 SSE 流，返回累积的 ChatResponse。
func (p *openaiSSEParser) parse(ctx context.Context) (*ChatResponse, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		line, err := p.reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("读取 SSE 流出错: %w", err)
		}
		if line != "" {
			if err := p.handleLine(strings.TrimRight(line, "\r\n")); err != nil {
				return nil, err
			}
		}
		if err == io.EOF {
			break
		}
	}

	// 没收到 [DONE] 也没收到 finish_reason，说明连接被中途掐断，
	// 此时累积到的内容可能是半条工具调用，必须让上层重试而不是继续处理。
	if !p.sawDone && p.finishReason == "" {
		return nil, fmt.Errorf("响应流提前结束（未收到 [DONE]）")
	}

	return p.buildResponse(), nil
}

// handleLine 处理一行 SSE 文本。
func (p *openaiSSEParser) handleLine(line string) error {
	if line == "" {
		return nil
	}
	// 兼容 "data: {...}" 与 "data:{...}" 两种写法
	after, found := strings.CutPrefix(line, "data:")
	if !found {
		return nil
	}
	data := strings.TrimLeft(after, " ")
	if data == "[DONE]" {
		p.sawDone = true
		return nil
	}
	return p.processChunk(data)
}

// processChunk 处理一个 data chunk。
func (p *openaiSSEParser) processChunk(jsonData string) error {
	var chunk openAIStreamChunk
	if err := json.Unmarshal([]byte(jsonData), &chunk); err != nil {
		return fmt.Errorf("解析 SSE chunk 失败: %w\nraw: %.200s", err, jsonData)
	}

	// 捕获消息 ID 和模型
	if chunk.ID != "" {
		p.messageID = chunk.ID
	}
	if chunk.Model != "" {
		p.model = chunk.Model
	}

	// 捕获用量：带 stream_options.include_usage 时，最后一个 chunk 才给出
	if chunk.Usage != nil {
		p.usage.InputTokens = chunk.Usage.PromptTokens
		p.usage.OutputTokens = chunk.Usage.CompletionTokens
	}

	// 处理 choices
	if len(chunk.Choices) == 0 {
		return nil
	}
	choice := chunk.Choices[0] // 通常只有一个 choice

	delta := choice.Delta

	// 累积文本
	if delta.Content != "" {
		p.textBuf.WriteString(delta.Content)
		if p.onText != nil {
			p.onText(delta.Content)
		}
	}

	// 累积工具调用
	for _, tc := range delta.ToolCalls {
		idx := 0
		if tc.Index != nil {
			idx = *tc.Index
		}

		acc, ok := p.toolCalls[idx]
		if !ok {
			acc = &toolCallAccum{}
			p.toolCalls[idx] = acc
		}

		if tc.ID != "" {
			acc.ID = tc.ID
		}
		if tc.Type != "" {
			acc.Type = tc.Type
		}
		if tc.Function.Name != "" {
			acc.Name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			acc.Arguments.WriteString(tc.Function.Arguments)
		}
	}

	// 捕获 finish_reason（在最后一个 chunk 中非空）
	if choice.FinishReason != "" {
		p.finishReason = choice.FinishReason
	}

	return nil
}

// buildResponse 从累积状态构建 ChatResponse。
func (p *openaiSSEParser) buildResponse() *ChatResponse {
	resp := &ChatResponse{
		ID:           p.messageID,
		Model:        p.model,
		FinishReason: mapFinishReason(p.finishReason),
		Usage:        p.usage,
	}

	// 文本内容
	if text := p.textBuf.String(); text != "" {
		resp.Content = append(resp.Content, ContentBlock{
			Type: "text",
			Text: text,
		})
	}

	// 工具调用内容（按 index 升序，index 可能不连续）
	indexes := make([]int, 0, len(p.toolCalls))
	for idx := range p.toolCalls {
		indexes = append(indexes, idx)
	}
	sort.Ints(indexes)

	for _, idx := range indexes {
		acc := p.toolCalls[idx]

		// 将累积的 arguments JSON 字符串解析为 map
		argsStr := acc.Arguments.String()
		var input map[string]any
		if argsStr != "" {
			json.Unmarshal([]byte(argsStr), &input)
		}
		if input == nil {
			input = make(map[string]any)
		}

		// 非空但解析失败的参数，是被 max_tokens 截断的半条 JSON
		partial := isPartialJSON(argsStr)

		resp.Content = append(resp.Content, ContentBlock{
			Type:    "tool_use",
			ID:      acc.ID,
			Name:    acc.Name,
			Input:   input,
			Partial: partial,
		})
	}

	return resp
}

// isPartialJSON 判断字符串是否像一段被截断的 JSON 对象。
// 空串与完整对象都返回 false。
func isPartialJSON(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	var probe map[string]any
	return json.Unmarshal([]byte(s), &probe) != nil
}

// ============================================================================
// OpenAI Stream Chunk 类型
// ============================================================================

// openAIStreamChunk 单个 SSE data chunk 的 JSON 结构。
type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Model   string               `json:"model"`
	Created int64                `json:"created"`
	Usage   *openAIStreamUsage   `json:"usage"`
	Choices []openAIStreamChoice `json:"choices"`
}

// openAIStreamUsage 流式响应最后一个 chunk 里的用量统计。
type openAIStreamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// openAIStreamChoice chunk 中的 choice。
type openAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason string            `json:"finish_reason"`
}

// openAIStreamDelta choice 中的 delta。
type openAIStreamDelta struct {
	Role      string                 `json:"role,omitempty"`
	Content   string                 `json:"content,omitempty"`
	ToolCalls []openAIStreamToolCall `json:"tool_calls,omitempty"`
}

// openAIStreamToolCall delta 中的 tool_call。
type openAIStreamToolCall struct {
	Index    *int              `json:"index,omitempty"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function openAIFunctionRef `json:"function"`
}

// openAIFunctionRef tool_call 中的函数引用。
type openAIFunctionRef struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ============================================================================
// Finish Reason 映射
// ============================================================================

// mapFinishReason 将 OpenAI finish_reason 映射为内部统一的 FinishReason。
func mapFinishReason(reason string) string {
	switch reason {
	case "stop":
		return FinishStop
	case "tool_calls":
		return FinishToolCalls
	case "length":
		return FinishLength
	case "":
		return FinishStop
	default:
		return FinishError
	}
}
