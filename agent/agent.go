// Package agent 实现 ReAct（Reasoning + Acting）智能体核心循环。
//
// 智能体接收用户任务后，循环调用 LLM 获取响应：
//   - 如果 LLM 返回文本回复：任务完成，返回给用户
//   - 如果 LLM 返回 tool_calls 请求：执行工具，将结果反馈给 LLM，继续循环
//   - 如果达到最大迭代次数：强制终止，返回错误
//
// 每个循环轮次中，完整的对话历史都会发送给 LLM，使其具备
// 上下文记忆和多步推理能力。
//
// 完整的执行流程和日志输出清晰地展示了代码在哪里执行。
package agent

import (
	"context"
	"fmt"
	"strings"

	"go-agent/llm"
	"go-agent/log"
	"go-agent/tools"
)

// ============================================================================
// 配置
// ============================================================================

// Config 智能体配置。
type Config struct {
	// Model 使用的 LLM 模型 ID，例如 "deepseek-pro"、"claude-opus-4-8"。
	Model string

	// MaxTokens 每次 API 调用的最大输出 token 数。
	MaxTokens int

	// MaxIterations ReAct 循环的最大迭代次数，防止无限循环。
	MaxIterations int

	// SystemPrompt 系统提示词，定义智能体的行为和规范。
	SystemPrompt string

	// WorkingDir 工作目录，工具操作相对于此目录执行。
	WorkingDir string
}

// DefaultConfig 返回推荐的默认配置。
func DefaultConfig(workingDir string, model string) Config {
	if model == "" {
		model = "deepseek-v4-flash"
	}
	return Config{
		Model:         model,
		MaxTokens:     32000,
		MaxIterations: 20,
		SystemPrompt:  SystemPrompt(workingDir),
		WorkingDir:    workingDir,
	}
}

// ============================================================================
// Agent — 智能体核心
// ============================================================================

// Agent 代码智能体，封装了 LLM Provider、工具注册表和对话历史。
type Agent struct {
	config   Config
	provider llm.Provider // LLM 后端抽象（支持 Anthropic / DeepSeek 等）
	registry *tools.Registry
	history  []llm.Message // 完整对话历史（system prompt 不在其中）
}

// New 创建一个新的智能体实例。
func New(config Config, provider llm.Provider, registry *tools.Registry) *Agent {
	return &Agent{
		config:   config,
		provider: provider,
		registry: registry,
		history:  make([]llm.Message, 0),
	}
}

// ============================================================================
// Run — ReAct 主循环（带详细执行日志）
// ============================================================================

// Run 执行用户任务，返回 LLM 的最终文本回复。
//
// 工作流程:
//  1. 将用户任务追加到对话历史
//  2. 进入 ReAct 循环:
//     a. 构造请求（含完整历史 + 系统提示词 + 工具定义）
//     b. 流式调用 LLM，实时打印文本增量
//     c. 将 LLM 回复追加到历史
//     d. 根据 finish_reason 决定下一步:
//     - "stop"       → 退出循环，返回最终文本
//     - "tool_calls" → 执行所有工具，将结果追加到历史，继续循环
//     - "length"     → 追加部分消息，继续循环让 LLM 续写
//  3. 返回最终回复文本和累计 token 用量
func (a *Agent) Run(ctx context.Context, userTask string) (reply string, usage llm.Usage, err error) {
	log.Info("========== Agent.Run() 开始 ==========")
	log.Debug("[agent/agent.go:98] 用户任务: %s", truncate(userTask, 200))
	log.Debug("[agent/agent.go:99] Provider: %s, Model: %s, MaxTokens: %d, MaxIterations: %d",
		a.provider.Name(), a.config.Model, a.config.MaxTokens, a.config.MaxIterations)

	// 步骤 1：将用户任务加入历史
	a.history = append(a.history, llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{{Type: "text", Text: userTask}},
	})
	log.Debug("[agent/agent.go:105] 用户消息已追加到历史，当前历史长度: %d 条消息", len(a.history))

	totalUsage := llm.Usage{}

	// 步骤 2：ReAct 循环
	for iteration := 0; iteration < a.config.MaxIterations; iteration++ {
		// 检查上下文取消
		select {
		case <-ctx.Done():
			log.Warn("[agent/agent.go:114] 上下文已取消，终止循环")
			return "", totalUsage, ctx.Err()
		default:
		}

		log.Info("[agent/agent.go:119] --- ReAct 循环 第 %d/%d 轮 ---", iteration+1, a.config.MaxIterations)
		log.Debug("[agent/agent.go:120] 构造 ChatRequest: model=%s, maxTokens=%d, tools=%d, historyLen=%d",
			a.config.Model, a.config.MaxTokens, len(a.registry.Definitions()), len(a.history))

		// 步骤 2a：构造请求
		req := llm.ChatRequest{
			Model:       a.config.Model,
			MaxTokens:   a.config.MaxTokens,
			System:      a.config.SystemPrompt,
			Messages:    a.history,
			Tools:       a.registry.Definitions(),
			RoundNumber: iteration + 1, // 1-based，用于日志
		}
		log.Debug("[agent/agent.go:132] 请求已构造，发起 LLM 调用 (provider=%s)...", a.provider.Name())

		// 步骤 2b：流式调用 LLM
		fmt.Print("\n🤖 ")
		var textBuilder strings.Builder
		resp, err := a.provider.StreamingChat(ctx, req, func(text string) {
			textBuilder.WriteString(text)
			fmt.Print(text) // 实时打印文本增量到终端
		})
		fmt.Println() // 文本回复后的换行

		if err != nil {
			log.Error("[agent/agent.go:143] LLM 调用失败: %v", err)
			return "", totalUsage, fmt.Errorf("LLM 调用失败: %w", err)
		}

		log.Debug("[agent/agent.go:147] LLM 调用返回: id=%s, model=%s, finishReason=%s, contentBlocks=%d",
			resp.ID, resp.Model, resp.FinishReason, len(resp.Content))

		// 累积 token 用量
		totalUsage.InputTokens += resp.Usage.InputTokens
		totalUsage.OutputTokens += resp.Usage.OutputTokens
		log.Debug("[agent/agent.go:153] Token 用量: 本轮 in=%d out=%d, 累计 in=%d out=%d",
			resp.Usage.InputTokens, resp.Usage.OutputTokens,
			totalUsage.InputTokens, totalUsage.OutputTokens)

		// 步骤 2c：将 LLM 回复加入历史
		a.history = append(a.history, llm.Message{
			Role:    llm.RoleAssistant,
			Content: resp.Content,
		})
		log.Debug("[agent/agent.go:161] Assistant 消息已追加到历史，当前历史长度: %d", len(a.history))

		// 步骤 2d：根据 finish_reason 分支
		switch resp.FinishReason {
		case llm.FinishStop:
			// 任务完成，返回最终文本
			log.Info("[agent/agent.go:167] finish_reason=stop，任务完成")
			log.Info("[agent/agent.go:168] ========== Agent.Run() 结束 ==========")
			log.Info("[agent/agent.go:169] 总 Token: 输入=%d 输出=%d 总计=%d, 共 %d 轮",
				totalUsage.InputTokens, totalUsage.OutputTokens,
				totalUsage.InputTokens+totalUsage.OutputTokens,
				iteration+1)
			return textBuilder.String(), totalUsage, nil

		case llm.FinishToolCalls:
			toolCount := countToolCalls(resp.Content)
			log.Info("[agent/agent.go:176] finish_reason=tool_calls，检测到 %d 个工具调用", toolCount)

			// 收集所有 tool_use 块并执行
			toolResults, err := a.executeToolCalls(ctx, resp.Content)
			if err != nil {
				log.Error("[agent/agent.go:181] 工具执行系统级错误: %v", err)
				return "", totalUsage, fmt.Errorf("工具执行失败: %w", err)
			}

			// 将工具执行结果作为 user 消息追加到历史
			// （Provider 负责将内部的 tool_result 转换为对应的 API 格式）
			a.history = append(a.history, llm.Message{
				Role:    llm.RoleUser,
				Content: toolResults,
			})
			log.Debug("[agent/agent.go:192] 工具结果已追加到历史（%d 个结果），当前历史长度: %d",
				len(toolResults), len(a.history))
			// 继续循环，让 LLM 处理工具结果

		case llm.FinishLength:
			// 输出被截断，将继续循环让 LLM 续写
			log.Warn("[agent/agent.go:198] finish_reason=length，输出达到 token 上限，自动续写...")
			fmt.Println("  ⚠️  输出达到 token 上限，自动续写...")
			// LLM 的回复已加入历史，下一轮自动续写

		case llm.FinishError:
			log.Error("[agent/agent.go:203] finish_reason=error，LLM 返回错误")
			return "", totalUsage, fmt.Errorf("LLM 返回错误 finish_reason")

		default:
			log.Error("[agent/agent.go:207] 未知的 finish_reason: %s", resp.FinishReason)
			return "", totalUsage, fmt.Errorf("未知的 finish_reason: %s", resp.FinishReason)
		}
	}

	// 达到最大迭代次数
	log.Error("[agent/agent.go:213] 达到最大迭代次数 (%d)，任务可能过于复杂", a.config.MaxIterations)
	return "", totalUsage, fmt.Errorf("达到最大迭代次数 (%d)，任务可能过于复杂，请简化后重试", a.config.MaxIterations)
}

// ============================================================================
// 工具执行
// ============================================================================

// executeToolCalls 执行 assistant 消息中的所有 tool_use 块。
// 返回对应的 tool_result ContentBlock 列表。
func (a *Agent) executeToolCalls(ctx context.Context, content []llm.ContentBlock) ([]llm.ContentBlock, error) {
	var results []llm.ContentBlock

	for _, block := range content {
		if block.Type != "tool_use" {
			continue
		}

		log.Info("[agent/agent.go:231] 🔧 执行工具: %s, ID=%s", block.Name, block.ID)
		for k, v := range block.Input {
			valStr := fmt.Sprintf("%v", v)
			if len(valStr) > 100 {
				valStr = valStr[:100] + "..."
			}
			log.Debug("[agent/agent.go:236]   参数: %s = %s", k, valStr)
		}

		// 查找工具
		tool, ok := a.registry.Get(block.Name)
		if !ok {
			log.Warn("[agent/agent.go:241] 未知工具: %s", block.Name)
			results = append(results, llm.ContentBlock{
				Type:      "tool_result",
				ToolUseID: block.ID,
				Content:   fmt.Sprintf("未知工具: %s", block.Name),
				IsError:   new(bool),
			})
			*results[len(results)-1].IsError = true
			continue
		}

		// 执行工具
		log.Debug("[agent/agent.go:252] 调用 %s.Execute()...", block.Name)
		result, execErr := tool.Execute(ctx, block.Input)

		if execErr != nil {
			// 工具执行错误 → is_error: true
			errMsg := execErr.Error()
			log.Warn("[agent/agent.go:258] ❌ 工具 %s 执行失败: %s", block.Name, errMsg)
			fmt.Printf("  ❌ 工具执行失败: %s\n", errMsg)
			isErr := true
			results = append(results, llm.ContentBlock{
				Type:      "tool_result",
				ToolUseID: block.ID,
				Content:   errMsg,
				IsError:   &isErr,
			})
		} else {
			// 工具执行成功
			preview := result
			if len(preview) > 200 {
				preview = preview[:200] + "..."
			}
			log.Debug("[agent/agent.go:273] ✅ 工具 %s 执行成功，结果长度: %d 字节", block.Name, len(result))
			fmt.Printf("  ✅ 结果: %s\n", preview)
			results = append(results, llm.ContentBlock{
				Type:      "tool_result",
				ToolUseID: block.ID,
				Content:   result,
			})
		}
	}

	log.Debug("[agent/agent.go:283] executeToolCalls 完成，返回 %d 个结果", len(results))
	return results, nil
}

// ============================================================================
// History 管理
// ============================================================================

// ClearHistory 清空对话历史。
func (a *Agent) ClearHistory() {
	log.Info("[agent/agent.go:293] 清空对话历史（之前共 %d 条消息）", len(a.history))
	a.history = a.history[:0]
}

// HistoryLen 返回当前对话历史的消息数。
func (a *Agent) HistoryLen() int {
	return len(a.history)
}

// ============================================================================
// 辅助函数
// ============================================================================

// countToolCalls 统计 content 中 tool_use 块的数量。
func countToolCalls(content []llm.ContentBlock) int {
	count := 0
	for _, block := range content {
		if block.Type == "tool_use" {
			count++
		}
	}
	return count
}

// truncate 截断字符串到指定长度。
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
