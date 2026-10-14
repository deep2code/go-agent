// Package agent 实现 ReAct（Reasoning + Acting）智能体核心循环。
//
// 智能体接收用户任务后，循环调用 LLM 获取响应：
//   - 如果 LLM 返回文本回复：任务完成，返回给用户
//   - 如果 LLM 返回 tool_calls 请求：执行工具，将结果反馈给 LLM，继续循环
//   - 如果达到最大迭代次数：强制终止
//
// 每轮把完整历史发给 LLM 以保留记忆，代价是对话越长越容易撑爆上下文窗口。
// 治理分两处落：工具输出**进历史之前**过一次 L1（超限就切成头 + 指针 + 尾，全文落盘，见 toolcache.go）；
// 发送前则先 RepairHistory 修好中断留下的残缺消息，再把历史里早期的大输出换成同一形式的指针（L2），
// 到这一步还是超预算才动用最后一档——CompactHistory 折叠旧的工具输出、丢弃最早的往返。
package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

	// MaxCostUSD 是单次会话的累计花费上限（按 EstimateCost 的口径），
	// 达到后 ReAct 循环以 cost_limit 停止。0 表示不设限。
	// 模型不在价目表里时无从换算金额，此时上限不生效并会告警一次。
	MaxCostUSD float64

	// Cache 是大输出的外置缓存（L1/L2 两级治理共用的那一份）。
	// 为 nil 时 New 按 WorkingDir 就地建一份；父子 agent 传同一个实例，
	// 否则主对话里生成的指针在子 agent 那边读不到。
	Cache *ToolCache
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
	// ErrCostLimit 表示会话累计花费达到 -max-cost 设定的上限。
	ErrCostLimit = errors.New("达到花费上限")
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

	// estScale 是"服务端真实输入 token / 本地估算"的滑动比例，0 表示还没校准过。
	// 预算判断必须走同一口径，否则启发式估算偏乐观时会在服务端那里撞墙。
	estScale float64

	// sentEstimate 是上一次真正发出去的载荷估算值，供服务端返回后校准。
	sentEstimate int

	// cache 是 L1/L2 共用的大输出外置缓存（由 New 填好，一定非 nil）。
	cache *ToolCache

	// warnedNoPrice 防止无价目表时每轮重复告警。
	warnedNoPrice bool
}

// New 创建一个新的智能体实例。
func New(config Config, provider llm.Provider, registry *tools.Registry) *Agent {
	if p, ok := provider.(llm.RetrySetter); ok {
		p.SetRetry(config.Retry)
	}
	if config.Cache == nil {
		// 缺省落在工作目录下的 .go-agent/：那里已经是智能体自己的地盘，
		// 也被 .gitignore 和搜索的跳过规则覆盖（不会把回灌过的输出当代码搜回来）。
		config.Cache = NewToolCache(
			filepath.Join(config.WorkingDir, ".go-agent", "toolcache"), ".go-agent/toolcache")
	}
	return &Agent{
		config:   config,
		provider: provider,
		registry: registry,
		history:  make([]llm.Message, 0),
		cache:    config.Cache,
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

// ToolNames 返回该 agent 可用的工具名。
// 用于在上层校验子 agent 的工具面（子 agent 必须只读）。
func (a *Agent) ToolNames() []string { return a.registry.Names() }

// SystemPrompt 返回当前生效的系统提示词。
// 用于校验子 agent 拿到的是只读提示词，而不是主 agent 那份。
func (a *Agent) SystemPrompt() string { return a.config.SystemPrompt }

// History 返回对话历史的副本，用于落盘或恢复。
func (a *Agent) History() []llm.Message { return cloneMessages(a.history) }

// Usage 返回累计 token 用量。
func (a *Agent) Usage() llm.Usage { return a.usage }

// AddUsage 把另一次运行（如子 agent）的用量累加进来，
// 这样主对话的花费统计能覆盖子任务烧掉的 token。
func (a *Agent) AddUsage(u llm.Usage) {
	a.usage.InputTokens += u.InputTokens
	a.usage.OutputTokens += u.OutputTokens
	a.usage.CacheCreationInputTokens += u.CacheCreationInputTokens
	a.usage.CacheReadInputTokens += u.CacheReadInputTokens
}

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

		roundUsage, err := a.prepareForSend(ctx)
		if err != nil {
			log.Error("上下文准备失败: %v", err)
			res.StopReason = "error"
			return res, err
		}
		if roundUsage > 0 {
			res.Compactions++
		}

		if err := a.checkCostLimit(); err != nil {
			res.Usage = a.usage
			res.Iterations = iteration - 1
			res.CostUSD = EstimateCost(a.config.Model, a.usage)
			res.Elapsed = time.Since(start)
			res.StopReason = "cost_limit"
			log.Warn("%v", err)
			return res, err
		}

		log.Info("ReAct 第 %d/%d 轮：历史 %d 条消息，估算 %d token（校准 %s）",
			iteration, a.config.MaxIterations, len(a.history), a.sentEstimate, a.calibrationLabel())

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
		a.usage.CacheCreationInputTokens += resp.Usage.CacheCreationInputTokens
		a.usage.CacheReadInputTokens += resp.Usage.CacheReadInputTokens
		a.calibrate(resp.Usage.PromptTokens())
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
	a.compact(ctx, a.config.HistoryBudget()/2)

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
//
// 预算是服务端的 token 口径，而 EstimateTokens 是本地启发式估算，两者有系统性偏差
// （尤其中英混排）。所以拿校准比例换算过再比较，不能直接拿估算值顶。
func (a *Agent) prepareForSend(ctx context.Context) (int, error) {
	repaired := RepairHistory(a.history)
	if len(repaired) == 0 {
		a.history = repaired
		return 0, fmt.Errorf("对话历史为空，无法发起请求")
	}
	a.history = repaired

	// L2 在估算之前：先给早期的大输出换成指针，再决定要不要动用折叠。
	// 顺序很要紧——放在估算之后等于这一级永远轮不到发言，
	// 因为超预算的判断已经按换指针之前的体积做完了。
	if freed := a.Pointerize(a.history); freed > 0 {
		log.Info("早期工具输出已换成指针，省下 %d 字节（完整内容仍在 .go-agent/toolcache 里可回读）", freed)
	}

	// 写入类参数跟着一起瘦身，但门更严：只砍已经拿到成功结果、且工具自己声明过"这是大正文"的。
	if freed := a.SlimBodyParams(a.history); freed > 0 {
		log.Info("早期写入参数正文已移除，省下 %d 字节（全文在各自文件里）", freed)
	}

	overhead := EstimateTools(a.registry.Definitions()) + estimateTextTokens(a.config.SystemPrompt)
	budget := a.config.HistoryBudget() - overhead
	scale := a.tokenScale()

	// 把本地估算换算到服务端口径后再判断是否超预算。
	used := int(float64(EstimateTokens(a.history)) * scale)
	freed := 0
	if used > budget {
		// CompactHistory 按本地估算工作，所以预算要折算回本地口径。
		localBudget := int(float64(budget) / scale)
		freed = a.compact(ctx, localBudget)
		log.Warn("上下文接近预算（真实口径 %d/%d，估算已按 %.2f 校准），已折叠历史省下约 %d token",
			used, budget, scale, freed)
	}

	a.sentEstimate = EstimateTokens(a.history) + overhead
	return freed, nil
}

// tokenScale 返回当前"服务端真实 token / 本地估算"的换算比例，未校准时为 1。
func (a *Agent) tokenScale() float64 {
	if a.estScale == 0 {
		return 1
	}
	return a.estScale
}

// calibrate 用服务端返回的真实输入 token 校正本地估算。
//
// 比例限幅在 [0.4, 3]：单轮的异常值（比如折叠后历史骤减）不该把系数带到离谱的位置；
// 取中点平滑一次，让系数随对话推进逐步收敛。
func (a *Agent) calibrate(actual int) {
	if actual <= 0 || a.sentEstimate <= 0 {
		return
	}
	ratio := float64(actual) / float64(a.sentEstimate)
	if ratio < 0.4 {
		ratio = 0.4
	}
	if ratio > 3 {
		ratio = 3
	}
	before := a.estScale
	if a.estScale == 0 {
		a.estScale = ratio
	} else {
		a.estScale = (a.estScale + ratio) / 2
	}
	log.Debug("token 估算校准：服务端 %d / 本地估算 %d → 系数 %.2f（原 %.2f）",
		actual, a.sentEstimate, a.estScale, before)
}

// calibrationLabel 给日志用的一句校准状态。
func (a *Agent) calibrationLabel() string {
	if a.estScale == 0 {
		return "未校准"
	}
	return fmt.Sprintf("%.2f", a.estScale)
}

// checkCostLimit 在发起下一次调用之前核对累计花费。
//
// 上限拦不住已经烧掉的那一次，但能在下一轮停下来——一轮的代价远小于放任整个循环。
func (a *Agent) checkCostLimit() error {
	limit := a.config.MaxCostUSD
	if limit <= 0 {
		return nil
	}
	spent := EstimateCost(a.config.Model, a.usage)
	if spent == 0 {
		if a.usage.Total() > 0 && !a.warnedNoPrice {
			a.warnedNoPrice = true
			log.Warn("模型 %s 不在内置价目表里，无法换算花费：-max-cost 上限不会生效", a.config.Model)
		}
		return nil
	}
	if spent < limit {
		return nil
	}
	return fmt.Errorf("%w：本次会话已花约 $%.4f，上限 $%.4f；可用 /cost 核对，或提高上限后重试",
		ErrCostLimit, spent, limit)
}

// compact 折叠历史到 budget 以内，并重新修复一遍。
//
// 折叠会改变消息序列（丢弃往返、插入摘要），结果可能不再满足角色交替；
// 之前直接发出去会撞上服务端的格式校验。折叠后必须再过一次 RepairHistory。
func (a *Agent) compact(ctx context.Context, budget int) int {
	compacted, freed := CompactHistory(ctx, a.history, budget, a.newSummarizer())
	a.history = RepairHistory(compacted)
	return freed
}

// ============================================================================
// 工具执行
// ============================================================================

// maxParallelTools 一组只读调用最多同时跑几个。
//
// 不放开是因为大仓库里一趟可能十几个 read_file，全开会把磁盘和 CPU 打满；
// 4 个已经把"排队等 IO"的时间吃掉大半。
const maxParallelTools = 4

// executeToolCalls 执行 assistant 消息中的所有 tool_use 块，
// 返回与之配对的 tool_result 块。
//
// 单个工具失败不作为整体失败：错误信息带 is_error 回给模型，由它决定换思路。
// 只有用户中断才会提前返回，此时未执行到的调用交给 RepairHistory 补占位结果。
//
// 彼此独立的只读调用会并发跑（见 tools.Parallelizable），但结果**严格按模型给出的顺序回填**：
// tool_result 的先后是历史的一部分，乱序会让下一轮请求里配对关系看起来对不上。
func (a *Agent) executeToolCalls(ctx context.Context, content []llm.ContentBlock, out io.Writer) ([]llm.ContentBlock, error) {
	// 槽位与 content 下标一一对应；nil 表示这一格没执行（非 tool_use、被截断、或中断未跑）
	slots := make([]*llm.ContentBlock, len(content))
	err := a.runBlocks(ctx, content, slots, out)
	return flattenSlots(slots), err
}

// runBlocks 按原序遍历调用，能并发的连续段攒成一组并发跑，其余逐个串行。
// slots 按 content 下标回填。返回 ErrInterrupted 时，未跑到的格子留空。
func (a *Agent) runBlocks(ctx context.Context, content []llm.ContentBlock, slots []*llm.ContentBlock, out io.Writer) error {
	i := 0
	for i < len(content) {
		block := content[i]
		if block.Type != "tool_use" || block.Partial {
			i++
			continue
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("%w: %v", ErrInterrupted, ctx.Err())
		default:
		}

		tool, ok := a.registry.Get(block.Name)
		if !ok {
			log.Warn("未知工具: %s", block.Name)
			slots[i] = ptrTo(errorResult(block.ID,
				fmt.Sprintf("未知工具: %s。可用工具: %s", block.Name, strings.Join(a.registry.Names(), ", "))))
			i++
			continue
		}

		if !tools.Parallelizable(tool) {
			a.preview(block, out)
			slots[i] = ptrTo(a.runOne(ctx, block, tool, out))
			i++
			continue
		}

		// 攒出连续的、都可并行的这一段
		end := i + 1
		for end < len(content) && a.parallelCandidate(content[end]) {
			end++
		}
		// 回显按原序打完再并发跑：谁先完成是不确定的，终端上的顺序必须确定
		for k := i; k < end; k++ {
			a.preview(content[k], out)
		}
		a.runParallel(ctx, content[i:end], slots[i:end], out)
		// 组里被中断的格子留空，交给外层循环在下一次 ctx 检查时收尾
		i = end
	}

	select {
	case <-ctx.Done():
		return fmt.Errorf("%w: %v", ErrInterrupted, ctx.Err())
	default:
	}
	return nil
}

// parallelCandidate 判断这一格能不能进并发组（与 runBlocks 里的分组条件保持一致）。
func (a *Agent) parallelCandidate(block llm.ContentBlock) bool {
	if block.Type != "tool_use" || block.Partial {
		return false
	}
	tool, ok := a.registry.Get(block.Name)
	return ok && tools.Parallelizable(tool)
}

// preview 打一行"要调用什么"，在真正执行之前。
func (a *Agent) preview(block llm.ContentBlock, out io.Writer) {
	previewArgs := summarizeArgs(block.Input)
	log.Info("执行工具 %s(%s)", block.Name, previewArgs)
	fmt.Fprintf(out, "  🔧 %s %s\n", block.Name, previewArgs)
}

// runParallel 并发执行一组只读调用，结果按传入顺序写回 slots。
func (a *Agent) runParallel(ctx context.Context, blocks []llm.ContentBlock, slots []*llm.ContentBlock, out io.Writer) {
	sem := make(chan struct{}, maxParallelTools)
	var wg sync.WaitGroup
	for k := range blocks {
		select {
		case <-ctx.Done():
			// 剩下的不启动：留空格子由 RepairHistory 补占位结果
			continue
		default:
		}
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			tool, _ := a.registry.Get(blocks[k].Name)
			slots[k] = ptrTo(a.runOneQuiet(ctx, blocks[k], tool))
		}(k)
	}
	wg.Wait()

	// 结果按原序落终端，而不是按谁先跑完
	for k, block := range blocks {
		if slots[k] == nil {
			continue
		}
		a.echoResult(block, *slots[k], out)
	}
}

// runOne 执行一个调用并把过程回显出来（串行路径用）。
func (a *Agent) runOne(ctx context.Context, block llm.ContentBlock, tool tools.Tool, out io.Writer) llm.ContentBlock {
	res := a.runOneQuiet(ctx, block, tool)
	a.echoResult(block, res, out)
	return res
}

// runOneQuiet 只执行、不打印：并发时两个 goroutine 同时往终端写会互相插字。
//
// 工具返回后立刻过 L1 治理（见 agent/toolcache.go）：这里正是"外部文本进历史"的那道门，
// 守卫已经在工具内部跑过，所以落盘的全文是隐去密钥之后的版本。
func (a *Agent) runOneQuiet(ctx context.Context, block llm.ContentBlock, tool tools.Tool) llm.ContentBlock {
	result, execErr := tool.Execute(ctx, block.Input)
	if execErr != nil {
		log.Warn("工具 %s 失败: %v", block.Name, execErr)
		return errorResult(block.ID, a.cache.Govern(block.ID, execErr.Error()))
	}
	// 重复读取先省掉：这一份和历史里那一份逐字相同，就没必要再发一遍（也不会落盘）。
	if notice := a.duplicateNotice(block.Name, block.Input, result); notice != "" {
		log.Info("工具 %s 的结果与历史里的一份相同，省略 %d 字节", block.Name, len(result))
		return llm.ContentBlock{Type: "tool_result", ToolUseID: block.ID, Content: notice}
	}
	log.Info("工具 %s 完成，返回 %d 字节", block.Name, len(result))
	return llm.ContentBlock{
		Type:      "tool_result",
		ToolUseID: block.ID,
		Content:   a.cache.Govern(block.ID, result),
	}
}

// echoResult 把一次执行的结果打到终端，顺序由调用方决定。
func (a *Agent) echoResult(block llm.ContentBlock, res llm.ContentBlock, out io.Writer) {
	text, _ := res.Content.(string)
	if res.IsError != nil && *res.IsError {
		fmt.Fprintf(out, "  ❌ %s\n", text)
		return
	}
	fmt.Fprint(out, indent(text))
}

// flattenSlots 按原顺序收集已执行的结果，跳过没执行到的格子。
func flattenSlots(slots []*llm.ContentBlock) []llm.ContentBlock {
	out := make([]llm.ContentBlock, 0, len(slots))
	for _, s := range slots {
		if s != nil {
			out = append(out, *s)
		}
	}
	return out
}

func ptrTo(b llm.ContentBlock) *llm.ContentBlock { return &b }

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
