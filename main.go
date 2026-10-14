// 代码智能体 (Code Agent) — 基于 LLM 的交互式代码助手。
//
// 两种用法：
//
//	交互：            go-agent                     进入 REPL，写操作逐个确认
//	非交互：          go-agent -p "任务" --yes      跑完输出结果，适合脚本与 CI
//
// 外部工具：在 .go-agent/mcp.json 里声明 MCP 服务器，启动时拉起并导入它的工具。
// 这些工具和本地工具走同一套审批门与内容守卫，子智能体拿不到它们。
//
// 环境变量:
//
//	LLM_PROVIDER       deepseek(默认) / anthropic / openai 兼容端点
//	DEEPSEEK_API_KEY   DeepSeek 密钥
//	ANTHROPIC_API_KEY  Anthropic 密钥
//	LLM_MODEL          模型 ID，也可用 -model 覆盖
//	LLM_BASE_URL       OpenAI 兼容端点地址（配合 LLM_PROVIDER=openai）
//	LOG_LEVEL          debug / info(默认) / warn / error
//	LLM_LOG_DIR        HTTP 日志目录，默认 ./logs
//	GO_AGENT_MCP_CONFIG 外部工具配置，默认 .go-agent/mcp.json
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"go-agent/agent"
	"go-agent/llm"
	"go-agent/log"
	"go-agent/mcp"
	"go-agent/tools"
)

// ============================================================================
// 运行期上下文
// ============================================================================

type app struct {
	cfg      agent.Config
	provider llm.Provider
	registry *tools.Registry
	gate     *tools.Gate
	agent    *agent.Agent
	plan     *tools.PlanTool

	// 会话级文件元信息缓存：主 agent 和子 agent 共用一份。
	// 各拿一份的话，子 agent 搜过的文件主 agent 还得重新判定一遍。
	index *tools.RepoIndex

	out  io.Writer
	in   *bufio.Reader
	json bool // 只把结果以 JSON 打到 stdout

	// 已拉起的 MCP 服务器子进程。退出时必须关掉，否则每次用完 CLI 都留下几个孤儿。
	// 收尾可能来自主流程 defer 和信号处理协程两边，用独立的锁保护，
	// 不要拿 a.mu：关进程要等几秒，别把 Ctrl+C 卡住。
	mcpMu      sync.Mutex
	mcpClients []*mcp.Client

	mu        sync.Mutex
	running   bool
	cancelRun context.CancelFunc
	interrupt int
	sessionID string
	turns     int
}

func main() {
	a, prompt, err := newApp(os.Args[1:])
	if err != nil {
		var e errExit
		switch {
		case errors.As(err, &e):
			os.Exit(e.code)
		case errors.Is(err, flag.ErrHelp):
			os.Exit(0)
		}
		fmt.Fprintf(os.Stderr, "❌ %v\n", err)
		os.Exit(2)
	}

	if prompt != "" {
		os.Exit(a.runOnce(prompt))
	}
	os.Exit(a.repl())
}

// ============================================================================
// 启动装配
// ============================================================================

func newApp(args []string) (*app, string, error) {
	fs := flag.NewFlagSet("go-agent", flag.ContinueOnError)
	fs.Usage = usage(fs)

	var (
		prompt    = fs.String("p", "", "非交互模式：执行一条任务后退出")
		model     = fs.String("model", "", "模型 ID，覆盖 LLM_MODEL")
		provider  = fs.String("provider", "", "deepseek / anthropic / openai")
		yes       = fs.Bool("yes", false, "预先放行所有写操作与命令（禁止清单仍然生效）")
		asJSON    = fs.Bool("json", false, "只输出 JSON 结果（隐含 -quiet）")
		quiet     = fs.Bool("quiet", false, "不打印执行过程")
		maxTurns  = fs.Int("max-turns", 30, "单次任务允许的最大轮数")
		maxCost   = fs.Float64("max-cost", 0, "会话累计花费上限（美元），0 表示不设限")
		cmdTO     = fs.Duration("cmd-timeout", 2*time.Minute, "单条命令默认超时")
		resume    = fs.String("resume", "", "恢复会话：会话 ID 或 last")
		noSession = fs.Bool("no-session", false, "不把对话写入 .go-agent/sessions")
		logLevel  = fs.String("log-level", os.Getenv("LOG_LEVEL"), "日志级别 debug/info/warn/error")
		mcpConfig = fs.String("mcp-config", "", "MCP 配置文件，默认 .go-agent/mcp.json（可用 GO_AGENT_MCP_CONFIG 指定）")
		printVers = fs.Bool("version", false, "打印版本后退出")
	)

	if err := fs.Parse(args); err != nil {
		return nil, "", err
	}
	if *printVers {
		fmt.Println("go-agent v0.2")
		return nil, "", errExit{0}
	}

	if *logLevel != "" {
		log.SetLevel(log.ParseLevel(*logLevel))
	}

	promptText := strings.TrimSpace(*prompt)
	if promptText == "" {
		promptText = strings.TrimSpace(strings.Join(fs.Args(), " "))
	}
	nonInteractive := promptText != ""

	workingDir, err := os.Getwd()
	if err != nil {
		return nil, "", fmt.Errorf("无法获取当前目录: %w", err)
	}

	httpLogDir := envOr("LLM_LOG_DIR", "logs")
	httpLogger, err := log.NewHTTPLogger(httpLogDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  HTTP 日志不可用: %v（继续运行）\n", err)
		httpLogger = nil
	}

	prov, provName, err := selectProvider(*provider, httpLogger)
	if err != nil {
		return nil, "", err
	}

	modelID := firstNonEmpty(*model, os.Getenv("LLM_MODEL"), "")
	if modelID == "" {
		modelID = defaultModel(provName)
	}

	a := &app{
		out:       os.Stdout,
		in:        bufio.NewReader(os.Stdin),
		json:      *asJSON,
		sessionID: agent.NewSessionID(),
	}
	if *noSession {
		a.sessionID = ""
	}

	gateMode := tools.ModeAsk
	switch {
	case *yes:
		gateMode = tools.ModeAuto
	case nonInteractive:
		gateMode = tools.ModeReadOnly
	}
	a.gate = tools.NewGate(gateMode)
	a.gate.Out = a.ui()

	// 注册工具
	a.registry = tools.NewRegistry()
	state := tools.NewFileState()
	idx := tools.NewRepoIndex()
	a.index = idx
	a.plan = tools.NewPlanTool(a.ui())
	a.registry.Register(tools.NewReadFileTool(workingDir, state))
	a.registry.Register(tools.NewSearchCodeTool(workingDir).WithIndex(idx))
	a.registry.Register(tools.NewListDirectoryTool(workingDir))
	a.registry.Register(tools.NewRepoOverviewTool(workingDir))
	a.registry.Register(tools.NewWriteFileTool(workingDir, a.gate, state).WithIndex(idx))
	a.registry.Register(tools.NewEditFileTool(workingDir, a.gate, state).WithIndex(idx))
	a.registry.Register(tools.NewMoveFileTool(workingDir, a.gate, state).WithIndex(idx))
	a.registry.Register(tools.NewDeleteFileTool(workingDir, a.gate, state).WithIndex(idx))
	a.registry.Register(tools.NewRunCommandTool(workingDir, a.gate, *cmdTO, a.ui()))
	a.registry.Register(a.plan)
	a.registry.Register(tools.NewTaskTool(&subAgentRunner{app: a}, a.gate, a.ui()))

	// 外部工具：配置不存在就是没装，不会因此报错
	if err := a.connectMCPServers(mcp.ConfigPath(
		firstNonEmpty(*mcpConfig, os.Getenv("GO_AGENT_MCP_CONFIG"), mcp.DefaultConfigPath), workingDir)); err != nil {
		return nil, "", err
	}

	cfg := agent.DefaultConfig(workingDir, modelID)
	cfg.MaxIterations = *maxTurns
	cfg.MaxCostUSD = *maxCost
	cfg.Quiet = *asJSON || *quiet
	cfg.Writer = os.Stdout
	// 主 agent 和子 agent 共用同一份大输出缓存（cfg 会被复制进 newSubAgent，指针跟着走）。
	// 各建一份的话，两边在同一个目录里写同名文件，容量上限也变成各算各的。
	cfg.Cache = agent.NewToolCache(
		filepath.Join(workingDir, ".go-agent", "toolcache"), ".go-agent/toolcache")
	a.cfg = cfg
	a.provider = prov
	a.agent = agent.New(cfg, prov, a.registry)

	// 交互模式下由人逐个确认；非交互模式没有可问的人。
	if gateMode == tools.ModeAsk {
		a.gate.Ask = a.confirm
	}

	if *resume != "" {
		if err := a.resumeSession(*resume); err != nil {
			a.closeMCPServers() // 已经拉起来的子进程要带走，别留给调用方
			return nil, "", err
		}
	}

	a.installSignalHandler()
	return a, promptText, nil
}

// connectMCPServers 拉起配置里的 MCP 服务器，把它们的工具注册进同一个 registry。
//
// 单台起不来只警告：配置里通常有好几台，一台坏了不该让 CLI 用不了。
// 但坏的那台一定要说清楚——静默跳过的话，人以为工具在，模型却看不到。
func (a *app) connectMCPServers(path string) error {
	servers, err := mcp.LoadConfig(path)
	if err != nil {
		return err
	}
	for _, cfg := range servers {
		if cfg.Disabled {
			log.Debug("MCP 服务器 %s 配置为 disabled，跳过", cfg.Name)
			continue
		}
		if missing := uniqueStrings(cfg.MissingEnv); len(missing) > 0 {
			// 空 token 起来的服务器会一路 401，现场看不出来；配置引用了没设置的变量就直接说
			fmt.Fprintf(a.ui(), "⚠️  MCP 服务器 %s 的配置引用了未设置的环境变量：%s（已展开成空值）\n",
				cfg.Name, strings.Join(missing, "、"))
		}

		ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
		client, err := mcp.StartServer(ctx, cfg)
		cancel()
		if err != nil {
			fmt.Fprintf(a.ui(), "⚠️  MCP 服务器 %s 启动失败：%v\n", cfg.Name, err)
			continue
		}

		names, skipped, err := tools.RegisterMCPServer(a.registry, client, a.gate, cfg.Timeout)
		if err != nil {
			client.Close()
			fmt.Fprintf(a.ui(), "⚠️  MCP 服务器 %s 没有可用工具：%v\n", cfg.Name, err)
			continue
		}
		a.mcpClients = append(a.mcpClients, client)

		if len(names) > 0 {
			fmt.Fprintf(a.ui(), "🔌 MCP %s：导入 %d 个工具（%s）\n",
				cfg.Name, len(names), strings.Join(names, ", "))
		} else {
			fmt.Fprintf(a.ui(), "🔌 MCP %s：已连接，但服务器没提供工具\n", cfg.Name)
		}
		for _, s := range skipped {
			fmt.Fprintf(a.ui(), "   跳过 %s\n", s)
		}
	}
	return nil
}

// closeMCPServers 收摊：这些子进程是我们拉起来的，退出时全部带走。
func (a *app) closeMCPServers() {
	a.mcpMu.Lock()
	clients := a.mcpClients
	a.mcpClients = nil
	a.mcpMu.Unlock()

	for _, c := range clients {
		if err := c.Close(); err != nil {
			// 进程是被信号终止的，Wait 返回退出码属正常，不值得吵人
			log.Debug("关闭 MCP 服务器 %s: %v", c.Server(), err)
		}
	}
}

// ui 返回过程输出的目标：JSON 模式下丢弃，避免污染 stdout。
func (a *app) ui() io.Writer {
	if a.json {
		return io.Discard
	}
	return a.out
}

// ============================================================================
// 子智能体
// ============================================================================

// subAgentRunner 把 tools.SubAgentRunner 接到一个全新的子 agent 上。
type subAgentRunner struct{ app *app }

// RunSubAgent 在全新上下文里跑一个子任务，只回传结论。
//
// 子 agent 默认只读（读文件/搜索/列目录/仓库骨架）：它翻很多文件的过程
// 留在自己的历史里，不污染主对话，也不能改动文件系统或执行命令。
// writable 由 task 工具在派发前问过用户，这里只负责据此换一套工具面和提示词。
func (r *subAgentRunner) RunSubAgent(ctx context.Context, task string, writable bool) (string, error) {
	child := r.app.newSubAgent(writable)
	res, err := child.Run(ctx, task)

	// 子任务烧掉的 token 要计入总花费，否则 /cost 会低估。
	// 即便子 agent 中途报错，它已经发生的调用也真实花了钱，所以无条件累加。
	r.app.agent.AddUsage(child.Usage())

	if err != nil {
		return "", err
	}
	return res.Reply, nil
}

// newSubAgent 造一个独立历史的子 agent。
//
// 两副面孔：只读面只有读文件/搜索/列目录/仓库骨架；可写面在此之上加
// write_file 与 edit_file——**只加这两个**。执行命令的破坏半径无法用 diff 表达，
// 删除和移动会让"改哪个文件"变成"哪个文件没了"，而 task 会让子 agent 再派孙 agent，
// 审批就散落到没人看的那一层去了。
func (a *app) newSubAgent(writable bool) *agent.Agent {
	cfg := a.cfg
	cfg.Quiet = true       // 子过程不刷屏，主 agent 只拿最终结论
	cfg.MaxIterations = 12 // 子任务范围更小，更快收敛
	if writable {
		cfg.SystemPrompt = agent.SubAgentWritePrompt(cfg.WorkingDir)
	} else {
		cfg.SystemPrompt = agent.SubAgentPrompt(cfg.WorkingDir)
	}
	return agent.New(cfg, a.provider, a.subRegistry(writable))
}

// subRegistry 装配子 agent 的工具面。
//
// 单独拆出来是为了让"这道门到底共用不共用"可测：可写面拿的是 a.gate 这一个实例，
// 父子共用一道门意味着子 agent 每次写入都要单独问一次，而不是继承一次"批准派发"。
// FileState 则**各自一份**——共享的话，父级读过的文件会替子级开门，
// 第 3 条的"先读后改"就只剩父级那一次读了。
// RepoIndex 反过来必须共用：它只缓存元信息，各拿一份等于白白重新判定每个文件。
func (a *app) subRegistry(writable bool) *tools.Registry {
	reg := tools.NewRegistry()
	state := tools.NewFileState()
	reg.Register(tools.NewReadFileTool(a.cfg.WorkingDir, state))
	reg.Register(tools.NewSearchCodeTool(a.cfg.WorkingDir).WithIndex(a.index))
	reg.Register(tools.NewListDirectoryTool(a.cfg.WorkingDir))
	reg.Register(tools.NewRepoOverviewTool(a.cfg.WorkingDir))

	if writable {
		reg.Register(tools.NewWriteFileTool(a.cfg.WorkingDir, a.gate, state).WithIndex(a.index))
		reg.Register(tools.NewEditFileTool(a.cfg.WorkingDir, a.gate, state).WithIndex(a.index))
	}
	return reg
}

// ============================================================================
// 执行一条任务
// ============================================================================

func (a *app) runOnce(prompt string) int {
	defer a.closeMCPServers()

	ctx, cancel := context.WithCancel(context.Background())
	a.setRunning(cancel)
	defer a.setRunning(nil)

	res, err := a.agent.Run(ctx, prompt)
	a.persistSession(res)

	if a.json {
		payload := map[string]any{
			"reply":       "",
			"stop_reason": "error",
			"error":       "",
		}
		if res != nil {
			payload["reply"] = res.Reply
			payload["stop_reason"] = res.StopReason
			payload["iterations"] = res.Iterations
			payload["usage"] = res.Usage
			payload["cost_usd"] = res.CostUSD
		}
		if err != nil {
			payload["error"] = err.Error()
		}
		data, _ := json.MarshalIndent(payload, "", "  ")
		fmt.Println(string(data))
		if err != nil {
			return 1
		}
		return 0
	}

	if err != nil {
		fmt.Fprintf(a.out, "\n❌ %v\n", err)
		if res != nil && res.Reply != "" {
			fmt.Fprintf(a.out, "已完成的回复内容：\n%s\n", res.Reply)
		}
		return 1
	}

	fmt.Fprintf(a.out, "\n%s\n", report(res))
	return 0
}

// report 把一次任务的结果写成人能读的一段话。
func report(res *agent.Result) string {
	var sb strings.Builder
	sb.WriteString(strings.Repeat("─", 56) + "\n")
	sb.WriteString(fmt.Sprintf("✅ 完成 · %d 轮 · 耗时 %s\n",
		res.Iterations, res.Elapsed.Round(time.Millisecond)))
	sb.WriteString(fmt.Sprintf("   token 输入 %d / 输出 %d，约 $%.4f\n",
		res.Usage.InputTokens, res.Usage.OutputTokens, res.CostUSD))
	if res.Compactions > 0 {
		sb.WriteString(fmt.Sprintf("   期间折叠过 %d 次历史以腾出上下文\n", res.Compactions))
	}
	return sb.String()
}

// repl 交互式循环。
func (a *app) repl() int {
	defer a.closeMCPServers()

	a.printWelcome()

	for {
		input, eof := a.readInput()
		if eof {
			fmt.Println()
			return 0
		}
		if input == "" {
			continue
		}

		if handled, quit := a.handleCommand(input); handled {
			if quit {
				return 0
			}
			continue
		}

		a.runTask(input)
	}
}

func (a *app) runTask(input string) {
	ctx, cancel := context.WithCancel(context.Background())
	a.setRunning(cancel)

	res, err := a.agent.Run(ctx, input)
	a.setRunning(nil)
	a.turns++
	a.persistSession(res)

	switch {
	case errors.Is(err, agent.ErrInterrupted):
		fmt.Fprintf(a.out, "\n⚠️  已中断。对话还在，输入「继续」可以从这里接着做。\n")
		return
	case err != nil:
		fmt.Fprintf(a.out, "\n❌ %v\n", err)
		fmt.Fprintf(a.out, "对话已保留；需要重来可以输入 /clear 清空。\n")
		return
	}

	fmt.Fprintf(a.out, "\n%s\n", report(res))
}

// ============================================================================
// 会话持久化
// ============================================================================

func (a *app) persistSession(res *agent.Result) {
	if a.sessionID == "" {
		return
	}
	usage := a.agent.Usage()
	if res != nil {
		usage = res.Usage
	}
	s := agent.Session{
		ID:         a.sessionID,
		WorkingDir: a.cfg.WorkingDir,
		Provider:   a.provider.Name(),
		Model:      a.cfg.Model,
		Messages:   a.agent.History(),
		Usage:      usage,
		Turns:      a.turns,
	}
	if err := agent.Save(a.cfg.WorkingDir, s); err != nil {
		log.Warn("会话保存失败: %v", err)
	}
}

func (a *app) resumeSession(id string) error {
	s, err := agent.Load(a.cfg.WorkingDir, id)
	if err != nil {
		return err
	}
	a.agent.LoadHistory(s.Messages)
	a.sessionID = s.ID
	fmt.Fprintf(a.out, "✅ 已恢复会话 %s（%d 条消息，模型 %s）\n", s.ID, len(s.Messages), s.Model)
	return nil
}

// ============================================================================
// 中断处理
// ============================================================================

// installSignalHandler 让 Ctrl+C 在执行中只取消当前任务，空闲时才退出。
// 旧行为是执行中按 Ctrl+C 直接退进程，把整段对话一起丢掉。
func (a *app) installSignalHandler() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		for sig := range ch {
			a.mu.Lock()
			running, cancel, count := a.running, a.cancelRun, a.interrupt
			if running {
				a.interrupt++
			}
			a.mu.Unlock()

			if running && cancel != nil && count == 0 {
				fmt.Fprintf(a.out, "\n\n⏸  正在取消当前任务…（再按一次立即退出）\n")
				cancel()
				continue
			}
			fmt.Fprintf(a.out, "\n\n👋 收到 %v，退出。\n", sig)
			// 立即退出也要收摊，否则我们拉起的 MCP 子进程全变成孤儿
			a.closeMCPServers()
			os.Exit(130)
		}
	}()
}

func (a *app) setRunning(cancel context.CancelFunc) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.running = cancel != nil
	a.cancelRun = cancel
	a.interrupt = 0
}

// ============================================================================
// 输入与确认
// ============================================================================

// confirm 向用户确认一次有副作用的操作。
// detail 可能是多行（写入类操作带着 diff）：每一行都缩进对齐，
// 让改动内容和"允许执行？"这个问题在视觉上分开。
func (a *app) confirm(title, detail string) bool {
	fmt.Fprintf(a.out, "\n  🔶 %s\n", title)
	for _, line := range strings.Split(strings.TrimRight(detail, "\n"), "\n") {
		fmt.Fprintf(a.out, "     %s\n", line)
	}
	fmt.Fprint(a.out, "  允许执行？[y/N] ")
	line, err := a.in.ReadString('\n')
	if err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes", "是":
		// 同一轮里用户可能只想放行这一次，不改变全局模式
		return true
	case "a", "all":
		a.gate.Mode = tools.ModeAuto
		fmt.Fprintf(a.out, "  本轮剩余操作自动放行（/reset-approval 可恢复逐个确认）\n")
		return true
	default:
		return false
	}
}

// readInput 读取一条任务。行尾 \ 表示续行，空行结束。
func (a *app) readInput() (string, bool) {
	var lines []string
	for {
		fmt.Print("> ")
		line, err := a.in.ReadString('\n')
		if err != nil {
			return "", true
		}
		line = strings.TrimRight(line, "\r\n")

		if strings.TrimSpace(line) == "" {
			if len(lines) == 0 {
				continue
			}
			break
		}
		if strings.HasSuffix(line, "\\") {
			lines = append(lines, strings.TrimSuffix(line, "\\"))
			continue
		}
		lines = append(lines, line)
		break
	}
	return strings.Join(lines, "\n"), false
}

// ============================================================================
// 斜杠命令
// ============================================================================

func (a *app) handleCommand(input string) (handled, quit bool) {
	name, arg, _ := strings.Cut(input, " ")
	arg = strings.TrimSpace(arg)

	switch name {
	case "/exit", "/quit":
		return true, true
	case "/help":
		printHelp()
		return true, false
	case "/clear":
		a.agent.ClearHistory()
		a.sessionID = agent.NewSessionID()
		fmt.Println("✅ 对话历史已清空，新会话开始")
		return true, false
	case "/model":
		if arg == "" {
			fmt.Printf("📋 当前模型: %s（上下文 %d）\nProvider: %s\n",
				a.cfg.Model, a.cfg.ContextWindow, a.provider.Name())
			return true, false
		}
		a.cfg.Model = arg
		a.agent.SetModel(arg)
		fmt.Printf("✅ 已切换到 %s\n\n", arg)
		return true, false
	case "/log":
		if arg == "" {
			fmt.Printf("📋 当前日志级别: %v\n\n", log.GetLevel())
			return true, false
		}
		log.SetLevel(log.ParseLevel(arg))
		fmt.Printf("✅ 日志级别: %v\n\n", log.GetLevel())
		return true, false
	case "/cost":
		u := a.agent.Usage()
		cost := agent.EstimateCost(a.cfg.Model, u)
		fmt.Printf("📊 累计 token：输入 %d / 输出 %d，约 $%.4f（按内置价目表估算）\n",
			u.InputTokens, u.OutputTokens, cost)
		if u.CacheReadInputTokens > 0 || u.CacheCreationInputTokens > 0 {
			fmt.Printf("   缓存：命中读 %d、写入 %d（这两段按 0.1x / 1.25x 计价，已计入上面的金额）\n",
				u.CacheReadInputTokens, u.CacheCreationInputTokens)
		}
		if a.cfg.MaxCostUSD > 0 {
			fmt.Printf("   本次运行上限 $%.2f，已用 %.0f%%\n",
				a.cfg.MaxCostUSD, cost/a.cfg.MaxCostUSD*100)
		}
		return true, false
	case "/tools":
		fmt.Printf("🔧 已注册工具: %s\n\n", strings.Join(a.registry.Names(), ", "))
		return true, false
	case "/plan":
		steps := a.plan.Steps()
		if len(steps) == 0 {
			fmt.Println("（当前没有清单）")
		}
		for _, s := range steps {
			fmt.Println("  " + s)
		}
		fmt.Println()
		return true, false
	case "/sessions":
		list, err := agent.List(a.cfg.WorkingDir)
		if err != nil {
			fmt.Printf("❌ %v\n\n", err)
			return true, false
		}
		if len(list) == 0 {
			fmt.Println("（还没有保存的会话）")
			fmt.Println()
			return true, false
		}
		for i, s := range list {
			if i >= 15 {
				fmt.Printf("  …（共 %d 个）\n", len(list))
				break
			}
			fmt.Printf("  %s  %s  %d 轮  %s\n", s.UpdatedAt.Format("01-02 15:04"), s.ID, s.Turns, s.Preview)
		}
		fmt.Printf("\n用 /resume <ID> 恢复，/resume last 恢复最近一次\n\n")
		return true, false
	case "/resume":
		if err := a.resumeSession(arg); err != nil {
			fmt.Printf("❌ %v\n\n", err)
		}
		fmt.Println()
		return true, false
	case "/reset-approval":
		a.gate.Mode = tools.ModeAsk
		fmt.Println("✅ 恢复逐个确认")
		return true, false
	default:
		return false, false
	}
}

// ============================================================================
// Provider 选择
// ============================================================================

// selectProvider 按选项和密钥选择 Provider。
// 返回的 name 是"实际生效的厂商"，可能与 p.Name() 不同：
// OpenAI 兼容端点复用 DeepSeekProvider（其 Name() 恒为 deepseek），
// 而默认模型要按厂商区分，所以这里把选择分支的名字带回去。
func selectProvider(fromFlag string, httpLogger *log.HTTPLogger) (llm.Provider, string, error) {
	name := strings.ToLower(firstNonEmpty(fromFlag, os.Getenv("LLM_PROVIDER"), "deepseek"))
	deepseekKey := os.Getenv("DEEPSEEK_API_KEY")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	openaiKey := os.Getenv("OPENAI_API_KEY")

	var p llm.Provider
	var effName string
	switch name {
	case "anthropic":
		if anthropicKey == "" {
			return nil, "", fmt.Errorf("provider=anthropic 但 ANTHROPIC_API_KEY 未设置")
		}
		p = llm.NewAnthropicProvider(anthropicKey)
		effName = "anthropic"
	case "deepseek":
		switch {
		case deepseekKey != "":
			p = llm.NewDeepSeekProvider(deepseekKey)
			effName = "deepseek"
		case anthropicKey != "":
			fmt.Fprintln(os.Stderr, "未找到 DEEPSEEK_API_KEY，改用 Anthropic")
			p = llm.NewAnthropicProvider(anthropicKey)
			effName = "anthropic"
		default:
			return nil, "", fmt.Errorf("缺少密钥：设置 DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY（见 -h）")
		}
	case "openai":
		if openaiKey == "" {
			return nil, "", fmt.Errorf("provider=openai 但 OPENAI_API_KEY 未设置")
		}
		p = llm.NewDeepSeekProviderWithBaseURL(envOr("LLM_BASE_URL", "https://api.openai.com/v1"), openaiKey)
		effName = "openai"
	default:
		return nil, "", fmt.Errorf("未知 provider: %s（支持 deepseek / anthropic / openai）", name)
	}

	if httpLogger != nil {
		p.SetHTTPLogger(httpLogger)
	}
	log.Info("Provider: %s", effName)
	return p, effName, nil
}

// defaultModel 返回该厂商未指定模型时使用的默认 ID。
func defaultModel(providerName string) string {
	switch providerName {
	case "anthropic":
		return "claude-sonnet-4-5"
	case "openai":
		return "gpt-4o-mini"
	default:
		return agent.DefaultModel
	}
}

// ============================================================================
// 帮助与欢迎
// ============================================================================

func (a *app) printWelcome() {
	out := a.out
	fmt.Fprintln(out, strings.Repeat("━", 60))
	fmt.Fprintln(out, "🤖 Go 代码智能体 v0.2")
	fmt.Fprintln(out, strings.Repeat("━", 60))
	fmt.Fprintf(out, "📁 工作目录: %s\n", a.cfg.WorkingDir)
	fmt.Fprintf(out, "🔌 Provider:  %s\n", a.provider.Name())
	fmt.Fprintf(out, "🧠 模型:     %s（上下文 %d）\n", a.cfg.Model, a.cfg.ContextWindow)
	fmt.Fprintf(out, "🔧 工具:     %s\n", strings.Join(a.registry.Names(), ", "))
	fmt.Fprintf(out, "🔐 审批:     %s\n", string(a.gate.Mode))
	fmt.Fprintf(out, "🔄 最大轮次: %d\n", a.cfg.MaxIterations)
	if a.sessionID != "" {
		fmt.Fprintf(out, "💾 会话:     %s（/resume last 可恢复）\n", a.sessionID)
	}
	fmt.Fprintln(out, strings.Repeat("━", 60))
	fmt.Fprintln(out, "输入任务开始（/help 看命令，/exit 退出）")
}

func printHelp() {
	fmt.Println(`
命令：
  /help                 本帮助
  /clear                清空对话，开新会话
  /sessions             列出保存过的会话
  /resume <ID|last>     恢复某次对话
  /model [名]           查看或切换模型
  /cost                 累计 token 与估算花费
  /tools                列出可用工具
  /plan                 查看当前任务清单
  /log [级别]           查看或切换日志级别
  /reset-approval       恢复逐个确认
  /exit                 退出

任务写法：多行输入在行尾加 \，空行结束。
中断：执行中按 Ctrl+C 取消当前任务，对话保留，可直接说"继续"。`)
}

func usage(fs *flag.FlagSet) func() {
	return func() {
		fmt.Println(`用法：
  go-agent [选项]                     进入交互模式
  go-agent -p "任务" --yes            非交互执行（默认只读，写操作需 --yes）

选项：`)
		fs.PrintDefaults()
		fmt.Println(`
密钥：DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY(+LLM_BASE_URL)

外部工具：默认读 .go-agent/mcp.json（-mcp-config 可换路径）。
里面的工具一样要过审批门：改动类的远端调用会逐个向你确认，返回内容按外部数据处理。`)
	}
}

// ============================================================================
// 辅助
// ============================================================================

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// uniqueStrings 去掉重复项，保持首次出现的顺序。
func uniqueStrings(vals []string) []string {
	var out []string
	for _, v := range vals {
		seen := false
		for _, o := range out {
			if o == v {
				seen = true
				break
			}
		}
		if !seen {
			out = append(out, v)
		}
	}
	return out
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// errExit 表示"已完成，按 code 退出"，不是错误。
type errExit struct{ code int }

func (e errExit) Error() string { return "" }
