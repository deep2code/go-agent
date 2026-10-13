// Package agent 实现 ReAct（Reasoning + Acting）智能体核心循环。
//
// 智能体接收用户任务后，循环调用 LLM 获取响应：
//   - 如果 LLM 返回文本回复：任务完成，返回给用户
//   - 如果 LLM 返回 tool_calls 请求：执行工具，将结果反馈给 LLM，继续循环
//   - 如果达到最大迭代次数：强制终止
//
// 每轮把完整历史发给 LLM 以保留记忆，代价是对话越长越容易撑爆上下文窗口，
// 因此发送前有两道处理：RepairHistory 修好中断留下的残缺消息，
// CompactHistory 在接近预算时折叠旧的工具输出、丢弃最早的往返。
package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"go-agent/llm"
	"go-agent/log"
	"go-agent/tools"
)

// ============================================================================
// 配置
// ============================================================================

// Config 智能体配置。
type Config struct {
	// Model 使用的 LLM 模型 ID。
	Model string

	// MaxTokens 单次 API 调用的最大输出 token 数。
	MaxTokens int

	// ContextWindow 模型的上下文窗口大小（输入 + 输出）。
	ContextWindow int

	// MaxIterations ReAct 循环的最大迭代次数，防止无限循环。
	MaxIterations int

	// SystemPrompt 系统提示词。
	SystemPrompt string

	// WorkingDir 工作目录，工具操作相对于此目录执行。
	WorkingDir string

	// Retry 限流与网络抖动的重试策略。
	Retry llm.Retry

	// Writer 过程输出目标；nil 表示 os.Stdout。
	Writer io.Writer

	// Quiet 为 true 时不输出任何过程信息，只返回结果（非交互模式用）。
	Quiet bool
}

// DefaultConfig 返回推荐的默认配置。
func DefaultConfig(workingDir string, model string) Config {
	if model == "" {
		model = DefaultModel
	}
	return Config{
		Model:         model,
		MaxTokens:     32000,
		ContextWindow: ContextWindowFor(model),
		MaxIterations: 30,
		SystemPrompt:  SystemPrompt(workingDir),
		WorkingDir:    workingDir,
		Retry:         llm.DefaultRetry(),
	}
}

// DefaultModel 未指定模型时使用的 ID。
const DefaultModel = "deepseek-chat"

// contextReserve 为响应与系统提示词预留的 token 数。
const contextReserve = 4000

// HistoryBudget 返回允许历史占用的 token 数。
func (c Config) HistoryBudget() int {
	budget := c.ContextWindow - c.MaxTokens - contextReserve
	if budget < 4000 {
		budget = 4000
	}
	return budget
}

// ContextWindowFor 按模型名推断上下文窗口大小。
func ContextWindowFor(model string) int {
	switch {
	case strings.Contains(model, "deepseek"):
		return 128000
	case strings.Contains(model, "claude"):
		return 200000
	case strings.Contains(model, "gpt-4o"), strings.Contains(model, "gpt-4.1"):
		return 128000
	case strings.Contains(model, "qwen"):
		return 128000
	case strings.Contains(model, "glm"):
		return 128000
	default:
		return 128000
	}
}

// Out 返回过程输出应写入的位置。
func (c Config) Out() io.Writer {
	if c.Quiet {
		return io.Discard
	}
	if c.Writer != nil {
		return c.Writer
	}
	return os.Stdout
}

// ============================================================================
// 结果与统计
// ============================================================================

// 任务未能正常完成时返回的错误。
var (
	// ErrInterrupted 表示用户中断了任务；历史已保留，可直接继续。
	ErrInterrupted = errors.New("任务被中断")
	// ErrMaxIterations 表示达到最大迭代次数仍未收敛。
	ErrMaxIterations = errors.New("达到最大迭代次数")
)

// Result 一次任务的结果。
type Result struct {
	Reply       string    // 最终文本回复
	Usage       llm.Usage // 本次累计用量
	Iterations  int       // ReAct 循环轮数
	Compactions int       // 因上下文压力触发的折叠次数
	Elapsed     time.Duration
	StopReason  string // completed / interrupted / max_iterations / error
	CostUSD     float64
}

// ============================================================================
// Agent — 智能体核心
// ============================================================================

// Agent 代码智能体，封装了 LLM Provider、工具注册表和对话历史。
type Agent struct {
	config   Config
	provider llm.Provider
	registry *tools.Registry
	history  []llm.Message
	usage    llm.Usage
}

// New 创建一个新的智能体实例。
func New(config Config, provider llm.Provider, registry *tools.Registry) *Agent {
	if p, ok := provider.(llm.RetrySetter); ok {
		p.SetRetry(config.Retry)
	}
	return &Agent{
		config:   config,
		provider: provider,
		registry: registry,
		history:  make([]llm.Message, 0),
	}
}

// Config 返回当前配置的副本。
func (a *Agent) Config() Config { return a.config }

// SetModel 切换模型，同时按新模型调整上下文窗口。
func (a *Agent) SetModel(model string) {
	log.Info("切换模型: %s -> %s（上下文窗口 %d）", a.config.Model, model, ContextWindowFor(model))
	a.config.Model = model
	a.config.ContextWindow = ContextWindowFor(model)
}

// HistoryLen 返回当前对话历史的消息数。
func (a *Agent) HistoryLen() int { return len(a.history) }

// History 返回对话历史的副本，用于落盘或恢复。
func (a *Agent) History() []llm.Message { return cloneMessages(a.history) }

// Usage 返回累计 token 用量。
func (a *Agent) Usage() llm.Usage { return a.usage }

// LoadHistory 用已有历史替换当前历史（恢复会话）。
func (a *Agent) LoadHistory(messages []llm.Message) {
	a.history = RepairHistory(messages)
	log.Info("已恢复会话历史：%d 条消息", len(a.history))
}

// ClearHistory 清空对话历史。
func (a *Agent) ClearHistory() {
	log.Info("清空对话历史（之前共 %d 条消息）", len(a.history))
	a.history = a.history[:0]
	a.usage = llm.Usage{}
}

// ============================================================================
// Run — ReAct 主循环
// ============================================================================

// Run 执行用户任务，返回 LLM 的最终文本回复。
//
// 出错时历史不会被丢弃：残缺的 tool_use 会在下一次调用前由 RepairHistory 补好，
// 所以被 Ctrl+C 打断的任务可以直接接着说"继续"，不必 /clear 从头再来。
func (a *Agent) Run(ctx context.Context, userTask string) (*Result, error) {
	start := time.Now()
	out := a.config.Out()

	a.history = append(a.history, llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{{Type: "text", Text: userTask}},
	})

	res := &Result{}
	var reply strings.Builder

	for iteration := 1; iteration <= a.config.MaxIterations; iteration++ {
		if err := ctx.Err(); err != nil {
			res.StopReason = "interrupted"
			return res, fmt.Errorf("%w: %v", ErrInterrupted, err)
		}

		roundUsage, err := a.prepareForSend()
		if err != nil {
			log.Error("上下文准备失败: %v", err)
			res.StopReason = "error"
			return res, err
		}
		if roundUsage > 0 {
			res.Compactions++
		}

		log.Info("ReAct 第 %d/%d 轮：历史 %d 条消息，估算 %d token",
			iteration, a.config.MaxIterations, len(a.history), roundUsage)

		fmt.Fprintf(out, "\n🤖 ")
		resp, err := a.call(ctx, iteration)
		fmt.Fprintln(out)

		if err != nil {
			res.StopReason = "error"
			if errors.Is(err, context.Canceled) {
				res.StopReason = "interrupted"
				return res, fmt.Errorf("%w: %v", ErrInterrupted, err)
			}
			return res, fmt.Errorf("LLM 调用失败: %w", err)
		}

		a.usage.InputTokens += resp.Usage.InputTokens
		a.usage.OutputTokens += resp.Usage.OutputTokens
		res.Usage = a.usage
		res.Iterations = iteration

		// 记录助手回复
		content := resp.Content
		if len(content) == 0 {
			content = []llm.ContentBlock{{Type: "text", Text: ""}}
		}
		a.history = append(a.history, llm.Message{Role: llm.RoleAssistant, Content: content})

		switch resp.FinishReason {
		case llm.FinishStop:
			reply.WriteString(joinText(content))
			res.Reply = strings.TrimSpace(reply.String())
			res.StopReason = "completed"
			res.Elapsed = time.Since(start)
			res.CostUSD = EstimateCost(a.config.Model, a.usage)
			log.Info("任务完成：共 %d 轮，输入 %d / 输出 %d token",
				iteration, a.usage.InputTokens, a.usage.OutputTokens)
			return res, nil

		case llm.FinishToolCalls:
			blocks, err := a.executeToolCalls(ctx, content, out)
			if err != nil {
				res.StopReason = "interrupted"
				return res, err
			}
			a.history = append(a.history, llm.Message{Role: llm.RoleUser, Content: blocks})
			reply.Reset() // 中间步骤的文本不是最终答案

		case llm.FinishLength:
			partial := HasPartialToolUse(content)
			log.Warn("输出达到 token 上限（第 %d 轮），%s", iteration,
				map[bool]string{true: "且工具调用参数被截断，已丢弃", false: "请求续写"}[partial])
			fmt.Fprintf(out, "  ⚠️  输出撞上 max_tokens 上限%s\n",
				map[bool]string{true: "，被截断的工具调用已丢弃", false: ""}[partial])
			a.history = append(a.history, llm.Message{
				Role:    llm.RoleUser,
				Content: []llm.ContentBlock{{Type: "text", Text: continuationPrompt(partial)}},
			})
			reply.Reset()

		default:
			res.StopReason = "error"
			return res, fmt.Errorf("LLM 返回未知结束原因: %s", resp.FinishReason)
		}
	}

	res.StopReason = "max_iterations"
	res.Elapsed = time.Since(start)
	res.CostUSD = EstimateCost(a.config.Model, a.usage)
	log.Error("达到最大迭代次数（%d）", a.config.MaxIterations)
	return res, fmt.Errorf("%w（%d 轮），请把任务拆小后重试", ErrMaxIterations, a.config.MaxIterations)
}

// call 发起一次流式调用，遇到上下文超限则强制折叠后重试一次。
func (a *Agent) call(ctx context.Context, round int) (*llm.ChatResponse, error) {
	req := a.request(round)
	out := a.config.Out()

	resp, err := a.provider.StreamingChat(ctx, req, func(text string) {
		fmt.Fprint(out, text)
	})
	if err == nil || !llm.IsOverContext(err) {
		return resp, err
	}

	// 估算偏乐观时仍可能撞墙：这里按服务端反馈再压一次
	log.Warn("服务端判定上下文超限，折叠历史后重试一次")
	budget := a.config.HistoryBudget() / 2
	compacted, _ := CompactHistory(a.history, budget)
	a.history = compacted

	retry, err2 := a.provider.StreamingChat(ctx, a.request(round), func(text string) {
		fmt.Fprint(out, text)
	})
	if err2 != nil {
		return nil, err // 首错信息更贴近用户看到的失败原因
	}
	return retry, nil
}

// request 组装本轮请求。
func (a *Agent) request(round int) llm.ChatRequest {
	return llm.ChatRequest{
		Model:       a.config.Model,
		MaxTokens:   a.config.MaxTokens,
		System:      a.config.SystemPrompt,
		Messages:    a.history,
		Tools:       a.registry.Definitions(),
		RoundNumber: round,
	}
}

// prepareForSend 修复历史并压到预算内，返回折叠省下的 token 数（0 表示未折叠）。
func (a *Agent) prepareForSend() (int, error) {
	repaired := RepairHistory(a.history)
	if len(repaired) == 0 {
		a.history = repaired
		return 0, fmt.Errorf("对话历史为空，无法发起请求")
	}
	a.history = repaired

	overhead := EstimateTools(a.registry.Definitions()) + estimateTextTokens(a.config.SystemPrompt)
	budget := a.config.HistoryBudget() - overhead
	if used := EstimateTokens(a.history); used <= budget {
		return 0, nil
	}
	compacted, freed := CompactHistory(a.history, budget)
	a.history = compacted
	log.Warn("上下文接近预算（预算 %d），已折叠历史省下约 %d token", budget, freed)
	return freed, nil
}

// ============================================================================
// 工具执行
// ============================================================================

// executeToolCalls 执行 assistant 消息中的所有 tool_use 块，
// 返回与之配对的 tool_result 块。
//
// 单个工具失败不作为整体失败：错误信息带 is_error 回给模型，由它决定换思路。
// 只有用户中断才会提前返回，此时未执行到的调用交给 RepairHistory 补占位结果。
func (a *Agent) executeToolCalls(ctx context.Context, content []llm.ContentBlock, out io.Writer) ([]llm.ContentBlock, error) {
	results := make([]llm.ContentBlock, 0, len(content))

	for _, block := range content {
		if block.Type != "tool_use" {
			continue
		}
		if block.Partial {
			continue // 参数被截断，不能执行
		}

		select {
		case <-ctx.Done():
			return results, fmt.Errorf("%w: %v", ErrInterrupted, ctx.Err())
		default:
		}

		previewArgs := summarizeArgs(block.Input)
		log.Info("执行工具 %s(%s)", block.Name, previewArgs)
		fmt.Fprintf(out, "  🔧 %s %s\n", block.Name, previewArgs)

		tool, ok := a.registry.Get(block.Name)
		if !ok {
			log.Warn("未知工具: %s", block.Name)
			results = append(results, errorResult(block.ID,
				fmt.Sprintf("未知工具: %s。可用工具: %s", block.Name, strings.Join(a.registry.Names(), ", "))))
			continue
		}

		result, execErr := tool.Execute(ctx, block.Input)
		if execErr != nil {
			log.Warn("工具 %s 失败: %v", block.Name, execErr)
			fmt.Fprintf(out, "  ❌ %v\n", execErr)
			results = append(results, errorResult(block.ID, execErr.Error()))
			continue
		}

		log.Info("工具 %s 完成，返回 %d 字节", block.Name, len(result))
		fmt.Fprint(out, indent(result))
		results = append(results, llm.ContentBlock{
			Type:      "tool_result",
			ToolUseID: block.ID,
			Content:   result,
		})
	}

	return results, nil
}

// errorResult 构造一条 is_error 的工具结果。
func errorResult(id, msg string) llm.ContentBlock {
	isErr := true
	return llm.ContentBlock{
		Type:      "tool_result",
		ToolUseID: id,
		Content:   msg,
		IsError:   &isErr,
	}
}

// summarizeArgs 把工具参数压成一行展示。
func summarizeArgs(input map[string]any) string {
	if len(input) == 0 {
		return "{}"
	}
	keys := make([]string, 0, len(input))
	for k := range input {
		keys = append(keys, k)
	}
	sortStrings(keys)

	var sb strings.Builder
	sb.WriteString("(")
	for i, k := range keys {
		if i > 0 {
			sb.WriteString(", ")
		}
		val := fmt.Sprintf("%v", input[k])
		val = strings.ReplaceAll(val, "\n", "\\n")
		if len(val) > 60 {
			val = truncateBytes(val, 60)
		}
		sb.WriteString(fmt.Sprintf("%s=%s", k, val))
	}
	sb.WriteString(")")
	return sb.String()
}

// indent 把工具输出缩进两格后交给终端。
func indent(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	var sb strings.Builder
	const maxLines = 30
	for i, line := range lines {
		if i == maxLines {
			sb.WriteString(fmt.Sprintf("  …（还有 %d 行未显示）\n", len(lines)-maxLines))
			break
		}
		if len(line) > 200 {
			line = truncateBytes(line, 200)
		}
		sb.WriteString("  │ " + line + "\n")
	}
	return sb.String()
}

// ============================================================================
// 辅助
// ============================================================================

// continuationPrompt 是输出被截断后补发给模型的一条提示。
func continuationPrompt(droppedToolCall bool) string {
	if droppedToolCall {
		return "（上一步的输出达到 max_tokens 上限，未完成的工具调用已作废。请把要写的文件拆成更小的块，或缩短单次调用的参数后重新发起。）"
	}
	return "（上一步的输出达到 max_tokens 上限被截断。请从中断处继续，不要重复已经输出的内容；若内容过长请拆成多步完成。）"
}

func joinText(blocks []llm.ContentBlock) string {
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "")
}

func truncateBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
