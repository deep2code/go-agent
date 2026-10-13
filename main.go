// 代码智能体 (Code Agent) — 基于 LLM 的交互式代码助手。
//
// 两种用法：
//
//	交互：            go-agent                     进入 REPL，写操作逐个确认
//	非交互：          go-agent -p "任务" --yes      跑完输出结果，适合脚本与 CI
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
	"strings"
	"sync"
	"syscall"
	"time"

	"go-agent/agent"
	"go-agent/llm"
	"go-agent/log"
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

	out  io.Writer
	in   *bufio.Reader
	json bool // 只把结果以 JSON 打到 stdout

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
		cmdTO     = fs.Duration("cmd-timeout", 2*time.Minute, "单条命令默认超时")
		resume    = fs.String("resume", "", "恢复会话：会话 ID 或 last")
		noSession = fs.Bool("no-session", false, "不把对话写入 .go-agent/sessions")
		logLevel  = fs.String("log-level", os.Getenv("LOG_LEVEL"), "日志级别 debug/info/warn/error")
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
	a.plan = tools.NewPlanTool(a.ui())
	a.registry.Register(tools.NewReadFileTool(workingDir, state))
	a.registry.Register(tools.NewSearchCodeTool(workingDir))
	a.registry.Register(tools.NewListDirectoryTool(workingDir))
	a.registry.Register(tools.NewWriteFileTool(workingDir, a.gate, state))
	a.registry.Register(tools.NewEditFileTool(workingDir, a.gate, state))
	a.registry.Register(tools.NewMoveFileTool(workingDir, a.gate, state))
	a.registry.Register(tools.NewDeleteFileTool(workingDir, a.gate, state))
	a.registry.Register(tools.NewRunCommandTool(workingDir, a.gate, *cmdTO, a.ui()))
	a.registry.Register(a.plan)

	cfg := agent.DefaultConfig(workingDir, modelID)
	cfg.MaxIterations = *maxTurns
	cfg.Quiet = *asJSON || *quiet
	cfg.Writer = os.Stdout
	a.cfg = cfg
	a.provider = prov
	a.agent = agent.New(cfg, prov, a.registry)

	// 交互模式下由人逐个确认；非交互模式没有可问的人。
	if gateMode == tools.ModeAsk {
		a.gate.Ask = a.confirm
	}

	if *resume != "" {
		if err := a.resumeSession(*resume); err != nil {
			return nil, "", err
		}
	}

	a.installSignalHandler()
	return a, promptText, nil
}

// ui 返回过程输出的目标：JSON 模式下丢弃，避免污染 stdout。
func (a *app) ui() io.Writer {
	if a.json {
		return io.Discard
	}
	return a.out
}

// ============================================================================
// 执行一条任务
// ============================================================================

func (a *app) runOnce(prompt string) int {
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
func (a *app) confirm(title, detail string) bool {
	fmt.Fprintf(a.out, "\n  🔶 %s\n     %s\n  允许执行？[y/N] ", title, detail)
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
		fmt.Printf("📊 累计 token：输入 %d / 输出 %d，约 $%.4f（按内置价目表估算）\n\n",
			u.InputTokens, u.OutputTokens, agent.EstimateCost(a.cfg.Model, u))
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
密钥：DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY(+LLM_BASE_URL)`)
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
