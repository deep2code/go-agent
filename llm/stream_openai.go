package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// ============================================================================
// OpenAI 格式 SSE 解析器
// ============================================================================

// openaiSSEParser 解析 OpenAI 兼容格式的 SSE 流式响应。
//
// SSE 事件格式:
//   data: {"id":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
//   data: {"id":"...","choices":[{"index":0,"delta":{"tool_calls":[...]},"finish_reason":null}]}
//   data: [DONE]
//
// 工具调用参数被拆成多个 JSON 片段流式传输，需要在客户端按 index 分组拼接。
type openaiSSEParser struct {
	reader *bufio.Reader
	onText func(text string)

	// 累积状态
	messageID    string
	model        string
	finishReason string

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
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("读取 SSE 流出错: %w", err)
		}

		line = strings.TrimRight(line, "\r\n")

		// 跳过空行和非 data 行
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// 流结束标记
		if data == "[DONE]" {
			break
		}

		if err := p.processChunk(data); err != nil {
			return nil, err
		}
	}

	return p.buildResponse(), nil
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
	}

	// 文本内容
	if text := p.textBuf.String(); text != "" {
		resp.Content = append(resp.Content, ContentBlock{
			Type: "text",
			Text: text,
		})
	}

	// 工具调用内容（按 index 排序）
	for i := 0; i < len(p.toolCalls); i++ {
		acc, ok := p.toolCalls[i]
		if !ok {
			continue
		}

		// 将累积的 arguments JSON 字符串解析为 map
		argsStr := acc.Arguments.String()
		var input map[string]any
		if argsStr != "" {
			json.Unmarshal([]byte(argsStr), &input)
		}
		if input == nil {
			input = make(map[string]any)
		}

		resp.Content = append(resp.Content, ContentBlock{
			Type:  "tool_use",
			ID:    acc.ID,
			Name:  acc.Name,
			Input: input,
		})
	}

	return resp
}

// ============================================================================
// OpenAI Stream Chunk 类型
// ============================================================================

// openAIStreamChunk 单个 SSE data chunk 的 JSON 结构。
type openAIStreamChunk struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Model   string                  `json:"model"`
	Created int64                   `json:"created"`
	Choices []openAIStreamChoice    `json:"choices"`
}

// openAIStreamChoice chunk 中的 choice。
type openAIStreamChoice struct {
	Index        int              `json:"index"`
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason string           `json:"finish_reason"`
}

// openAIStreamDelta choice 中的 delta。
type openAIStreamDelta struct {
	Role      string               `json:"role,omitempty"`
	Content   string               `json:"content,omitempty"`
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
		return "" // 流进行中
	default:
		return FinishError
	}
}
