package agent

import (
	"context"
	"fmt"
	"strings"

	"go-agent/llm"
)

// summaryMaxTokens 给摘要调用预留的输出上限。摘要本该短，
// 但也不能小到把结论截断。
const summaryMaxTokens = 600

// newSummarizer 返回一个绑定当前 provider 的 Summarizer，供 CompactHistory 使用。
//
// 摘要是一次独立的短调用：不带工具、不带系统提示词，只把待折叠的往返压成要点。
// max_tokens 用小额度，避免摘要本身把省下来的预算吃回去。
func (a *Agent) newSummarizer() Summarizer {
	return func(ctx context.Context, messages []llm.Message) (string, error) {
		transcript := renderTranscript(messages)
		if transcript == "" {
			return "", nil
		}

		req := llm.ChatRequest{
			Model:     a.config.Model,
			MaxTokens: summaryMaxTokens,
			System:    summarizerPrompt,
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []llm.ContentBlock{{Type: "text", Text: transcript}},
			}},
		}

		resp, err := a.provider.StreamingChat(ctx, req, nil)
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(joinText(resp.Content)), nil
	}
}

const summarizerPrompt = `你在为编程智能体压缩对话历史。下面的内容即将从上下文中移除。
请把要点浓缩成简短中文列表，只保留后续继续工作必需的信息：
- 已经查明的事实（文件位置、函数名、关键结论）
- 已经改动的文件和改动意图
- 已经执行过的命令及其结果要点
- 仍未完成的下一步

不要编造原文里没有的内容，不要复述工具调用本身。控制在 12 行以内。`

// renderTranscript 把一段历史摊平成纯文本，供摘要调用阅读。
// 工具结果只取文本内容；单块过长时截断，避免摘要请求本身超限。
func renderTranscript(messages []llm.Message) string {
	const maxBlockChars = 1200
	var sb strings.Builder
	for _, m := range messages {
		who := "用户"
		if m.Role == llm.RoleAssistant {
			who = "助手"
		}
		for _, b := range m.Content {
			switch b.Type {
			case "text":
				if t := strings.TrimSpace(b.Text); t != "" {
					fmt.Fprintf(&sb, "%s: %s\n", who, clip(t, maxBlockChars))
				}
			case "thinking":
				// 思考过程不进气泡：留了它反而稀释摘要
			case "tool_use":
				fmt.Fprintf(&sb, "%s调用工具 %s，参数 %s\n", who, b.Name, clipArgs(b.Input, maxArgChars))
			case "tool_result":
				if t := strings.TrimSpace(toolResultText(b)); t != "" {
					fmt.Fprintf(&sb, "工具结果: %s\n", clip(t, maxBlockChars))
				}
			}
		}
	}
	return sb.String()
}

// maxArgChars 是摘要里单个参数值的上限。
//
// 摘要请求本意是"把即将丢掉的东西压成几行"，如果把 write_file 的整份 content 原样
// 摊进去，摘要调用自己就成了被折叠内容的大小——省下的预算又还回去了。
const maxArgChars = 200

// clipArgs 把工具参数摊成一行，逐个值截断；键按字母序，保证同样的调用给出同样的文本。
func clipArgs(input map[string]any, max int) string {
	if len(input) == 0 {
		return "{}"
	}
	keys := make([]string, 0, len(input))
	for k := range input {
		keys = append(keys, k)
	}
	sortStrings(keys)

	var sb strings.Builder
	sb.WriteString("{")
	for i, k := range keys {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "%s=%q", k, clip(fmt.Sprintf("%v", input[k]), max))
	}
	sb.WriteString("}")
	return sb.String()
}

// clip 按 rune 截断，避免把中文切成半个字。
func clip(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
