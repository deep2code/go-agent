// 代码智能体 (Code Agent) — 基于 LLM 的交互式代码助手
//
// 支持多个 LLM 后端 (Provider):
//   - DeepSeek (默认): 设置 DEEPSEEK_API_KEY
//   - Anthropic: 设置 ANTHROPIC_API_KEY + LLM_PROVIDER=anthropic
//
// 使用方法:
//
//	export DEEPSEEK_API_KEY="sk-..."
//	go run .
//
// 环境变量:
//
//	LLM_PROVIDER      后端 Provider: "deepseek"(默认) 或 "anthropic"
//	DEEPSEEK_API_KEY  DeepSeek API 密钥
//	ANTHROPIC_API_KEY Anthropic API 密钥
//	LLM_MODEL         模型 ID (默认: deepseek-v4-flash / claude-opus-4-8)
//	LOG_LEVEL         日志级别: "debug" / "info"(默认) / "warn" / "error"
//	LLM_LOG_DIR        HTTP 日志目录 (默认: ./logs)
//
// 特殊命令:
//
//	/help   — 显示帮助
//	/clear  — 清空对话历史
//	/exit   — 退出程序
//	/model  — 显示当前模型
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"go-agent/agent"
	"go-agent/llm"
	"go-agent/log"
	"go-agent/tools"
)

func main() {
	// ─────────────────────────────────────────────
	// 1. 日志级别
	// ─────────────────────────────────────────────
	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel != "" {
		log.SetLevel(log.ParseLevel(logLevel))
	}

	log.Info("[main.go:48] ========== Go Code Agent 启动 ==========")
	log.Debug("[main.go:49] LOG_LEVEL=%v", log.GetLevel())

	// ─────────────────────────────────────────────
	// 1.5. HTTP 请求/响应日志（写入文件）
	// ─────────────────────────────────────────────
	httpLogDir := os.Getenv("LLM_LOG_DIR")
	if httpLogDir == "" {
		httpLogDir = "logs"
	}
	httpLogger, err := log.NewHTTPLogger(httpLogDir)
	if err != nil {
		fmt.Printf("⚠️  无法创建 HTTP 日志文件: %v (继续运行)\n\n", err)
	} else {
		defer httpLogger.Close()
	}

	// ─────────────────────────────────────────────
	// 2. 选择 Provider 和 API Key
	// ─────────────────────────────────────────────
	provider := selectProvider(httpLogger)
	log.Info("[main.go:55] Provider: %s", provider.Name())

	// ─────────────────────────────────────────────
	// 3. 获取工作目录
	// ─────────────────────────────────────────────
	workingDir, err := os.Getwd()
	if err != nil {
		log.Error("[main.go:61] 无法获取当前目录: %v", err)
		fmt.Printf("❌ 无法获取当前目录: %v\n", err)
		os.Exit(1)
	}
	log.Debug("[main.go:65] 工作目录: %s", workingDir)

	// ─────────────────────────────────────────────
	// 4. 初始化组件
	// ─────────────────────────────────────────────
	log.Debug("[main.go:70] 初始化工具注册表...")

	// 工具注册表
	registry := tools.NewRegistry()
	registry.Register(tools.NewReadFileTool(workingDir))
	registry.Register(tools.NewWriteFileTool(workingDir))
	registry.Register(tools.NewListDirectoryTool(workingDir))
	registry.Register(tools.NewSearchCodeTool(workingDir))
	registry.Register(tools.NewRunCommandTool(workingDir))
	log.Debug("[main.go:78] 已注册 5 个工具: read_file, write_file, list_directory, search_code, run_command")

	// 模型选择
	model := os.Getenv("LLM_MODEL")
	if model == "" {
		model = defaultModel(provider)
	}

	// 智能体
	config := agent.DefaultConfig(workingDir, model)
	agt := agent.New(config, provider, registry)
	log.Debug("[main.go:89] Agent 已创建: model=%s, maxTokens=%d, maxIterations=%d",
		config.Model, config.MaxTokens, config.MaxIterations)

	// ─────────────────────────────────────────────
	// 5. 信号处理（Ctrl+C 优雅退出）
	// ─────────────────────────────────────────────
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Info("[main.go:102] 收到退出信号: %v", sig)
		fmt.Println("\n\n👋 收到退出信号，正在退出...")
		cancel()
	}()

	// ─────────────────────────────────────────────
	// 6. 欢迎信息
	// ─────────────────────────────────────────────
	printWelcome(config, provider)

	// ─────────────────────────────────────────────
	// 7. REPL 循环
	// ─────────────────────────────────────────────
	fmt.Println("输入你的任务（输入 /help 查看帮助，/exit 退出）")

	for {
		// 读取用户输入
		input := readInput()
		if input == "" {
			continue
		}

		// 处理特殊命令
		switch {
		case input == "/exit" || input == "/quit":
			fmt.Println("👋 再见!")
			log.Info("[main.go:122] 用户退出")
			return

		case input == "/help":
			printHelp()
			continue

		case input == "/clear":
			agt.ClearHistory()
			fmt.Println("✅ 对话历史已清空")
			continue

		case input == "/model":
			fmt.Printf("📋 当前模型: %s\n", config.Model)
			fmt.Printf("🔌 Provider: %s\n\n", provider.Name())
			continue

		case strings.HasPrefix(input, "/model "):
			newModel := strings.TrimPrefix(input, "/model ")
			config.Model = newModel
			log.Info("[main.go:140] 切换模型: %s -> %s", model, newModel)
			model = newModel
			fmt.Printf("✅ 已切换模型为: %s\n\n", newModel)
			continue

		case input == "/log":
			lvl := log.GetLevel()
			fmt.Printf("📋 当前日志级别: %v\n\n", lvl)
			continue

		case strings.HasPrefix(input, "/log "):
			newLevel := strings.TrimPrefix(input, "/log ")
			lvl := log.ParseLevel(newLevel)
			log.SetLevel(lvl)
			log.Info("[main.go:153] 切换日志级别: %v", lvl)
			fmt.Printf("✅ 已切换日志级别为: %v\n\n", lvl)
			continue
		}

		// 执行任务
		log.Info("[main.go:159] ── 开始执行用户任务 ──")
		reply, usage, err := agt.Run(ctx, input)

		if err != nil {
			// 区分用户取消 vs 其他错误
			if ctx.Err() != nil {
				log.Warn("[main.go:164] 任务被用户取消")
				fmt.Println("\n⚠️  任务被取消")
				break
			}
			log.Error("[main.go:168] 任务执行失败: %v", err)
			fmt.Printf("\n❌ 错误: %v\n\n", err)
			continue
		}

		// 显示 token 用量
		totalTokens := usage.InputTokens + usage.OutputTokens
		fmt.Printf("\n📊 Token 用量: 输入 %d + 输出 %d = 总计 %d\n\n",
			usage.InputTokens,
			usage.OutputTokens,
			totalTokens,
		)
		log.Info("[main.go:180] 任务完成: inTokens=%d outTokens=%d total=%d",
			usage.InputTokens, usage.OutputTokens, totalTokens)

		_ = reply // reply 已通过流式回调实时打印
	}
}

// ============================================================================
// Provider 选择
// ============================================================================

// selectProvider 根据环境变量选择并创建 LLM Provider。
func selectProvider(httpLogger *log.HTTPLogger) llm.Provider {
	providerEnv := strings.ToLower(os.Getenv("LLM_PROVIDER"))
	deepseekKey := os.Getenv("DEEPSEEK_API_KEY")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")

	switch {
	case providerEnv == "anthropic":
		if anthropicKey == "" {
			fmt.Println("❌ 设置了 LLM_PROVIDER=anthropic 但未设置 ANTHROPIC_API_KEY")
			os.Exit(1)
		}
		log.Debug("[main.go:220] 使用 Anthropic Provider")
		p := llm.NewAnthropicProvider(anthropicKey)
		p.SetHTTPLogger(httpLogger)
		return p

	case providerEnv == "deepseek" || providerEnv == "":
		if deepseekKey != "" {
			log.Debug("[main.go:227] 使用 DeepSeek Provider")
			p := llm.NewDeepSeekProvider(deepseekKey)
			p.SetHTTPLogger(httpLogger)
			return p
		}
		if anthropicKey != "" {
			log.Debug("[main.go:233] 未设置 DEEPSEEK_API_KEY，自动使用 Anthropic")
			p := llm.NewAnthropicProvider(anthropicKey)
			p.SetHTTPLogger(httpLogger)
			return p
		}
		fmt.Println("❌ 未设置 DEEPSEEK_API_KEY 环境变量")
		fmt.Println()
		fmt.Println("请设置后重试：")
		fmt.Println("  export DEEPSEEK_API_KEY=\"sk-...\"")
		fmt.Println()
		fmt.Println("或使用 Anthropic:")
		fmt.Println("  export ANTHROPIC_API_KEY=\"sk-ant-...\"")
		fmt.Println("  export LLM_PROVIDER=anthropic")
		os.Exit(1)

	default:
		fmt.Printf("❌ 未知的 LLM_PROVIDER: %s (支持: deepseek, anthropic)\n", providerEnv)
		os.Exit(1)
	}

	return nil // unreachable
}

// defaultModel 返回 Provider 的默认模型。
func defaultModel(provider llm.Provider) string {
	switch provider.Name() {
	case "deepseek":
		return "deepseek-v4-flash"
	case "anthropic":
		return "claude-opus-4-8"
	default:
		return "deepseek-v4-flash"
	}
}

// ============================================================================
// 辅助函数
// ============================================================================

// printWelcome 打印欢迎信息。
func printWelcome(config agent.Config, provider llm.Provider) {
	fmt.Println(strings.Repeat("━", 60))
	fmt.Println("🤖 Go 代码智能体 (Code Agent) v2")
	fmt.Println(strings.Repeat("━", 60))
	fmt.Printf("📁 工作目录: %s\n", config.WorkingDir)
	fmt.Printf("🔌 Provider:  %s\n", provider.Name())
	fmt.Printf("🧠 模型:     %s\n", config.Model)
	fmt.Printf("🔧 可用工具: read_file, write_file, list_directory, search_code, run_command\n")
	fmt.Printf("🔄 最大轮次: %d\n", config.MaxIterations)
	fmt.Printf("📝 日志级别: %v\n", log.GetLevel())
	fmt.Println(strings.Repeat("━", 60))
}

// printHelp 打印帮助信息。
func printHelp() {
	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  💡 命令帮助")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()
	fmt.Println("  输入任务描述即可让智能体开始工作，例如：")
	fmt.Println("    • 「列出当前目录所有 Go 文件」")
	fmt.Println("    • 「读取 main.go 并解释其功能」")
	fmt.Println("    • 「搜索所有包含 TODO 的文件」")
	fmt.Println("    • 「创建一个 hello.go 文件，输出 Hello World」")
	fmt.Println("    • 「运行 go build 检查是否有编译错误」")
	fmt.Println()
	fmt.Println("  特殊命令:")
	fmt.Println("    /help       显示本帮助")
	fmt.Println("    /clear      清空对话历史（重置上下文）")
	fmt.Println("    /model      查看当前模型")
	fmt.Println("    /model <名> 切换模型（如 /model deepseek-pro、deepseek-flash）")
	fmt.Println("    /log        查看当前日志级别")
	fmt.Println("    /log <级别> 切换日志级别（debug / info / warn / error）")
	fmt.Println("    /exit       退出程序")
	fmt.Println()
	fmt.Println("  多行输入: 行尾加 \\ 续行，空行结束输入")
	fmt.Println("  Ctrl+C:    取消当前任务或退出程序")
	fmt.Println()
}

// readInput 从标准输入读取用户输入。
// 单行直接输入；行尾 \\ 表示续行，空行结束多行输入。
func readInput() string {
	reader := bufio.NewReader(os.Stdin)
	var lines []string

	for {
		fmt.Print("> ")

		line, err := reader.ReadString('\n')
		if err != nil {
			// EOF
			fmt.Println()
			break
		}

		line = strings.TrimRight(line, "\r\n")

		// 空行：如果没有内容，继续等待；如果有内容，结束多行输入
		if strings.TrimSpace(line) == "" {
			if len(lines) == 0 {
				continue // 忽略纯空行
			}
			break
		}

		// 以 \ 结尾表示继续输入
		if strings.HasSuffix(line, "\\") {
			lines = append(lines, strings.TrimSuffix(line, "\\"))
			continue
		}

		// 搜索特殊命令的单行输入 /exit, /help 等
		lines = append(lines, line)
		break
	}

	if len(lines) == 0 {
		return ""
	}
	return strings.Join(lines, "\n")
}
