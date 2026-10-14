package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go-agent/agent"
	"go-agent/llm"
	"go-agent/tools"
)

// isolateEnv 让每个用例都在干净的环境变量下运行：
// 密钥、provider、模型都不许从开发机上残留的 shell 环境里漏进来。
func isolateEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{"DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"} {
		t.Setenv(k, "")
	}
	t.Setenv("LOG_LEVEL", "error")
	// 日志目录必须重定向，否则测试会在仓库里写出 logs/。
	t.Setenv("LLM_LOG_DIR", t.TempDir())
	// 外部工具配置也必须隔离：开发机上真实存在的 .go-agent/mcp.json 会让每个用例
	// 都去拉起子进程，工具数量断言也会莫名其妙地红。
	t.Setenv("GO_AGENT_MCP_CONFIG", filepath.Join(t.TempDir(), "no-mcp.json"))
}

// mustApp 装配一次 CLI，失败即用例失败。
func mustApp(t *testing.T, args ...string) (*app, string) {
	t.Helper()
	isolateEnv(t)
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	a, prompt, err := newApp(args)
	if err != nil {
		t.Fatalf("newApp(%v) 失败: %v", args, err)
	}
	return a, prompt
}

func TestGateModeFollowsFlags(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		mode    tools.GateMode
		wantAsk bool
	}{
		{"交互默认逐个确认", nil, tools.ModeAsk, true},
		{"交互加 yes", []string{"--yes"}, tools.ModeAuto, false},
		{"非交互无授权只能只读", []string{"-p", "看看代码"}, tools.ModeReadOnly, false},
		{"非交互加授权", []string{"-p", "改一下", "--yes"}, tools.ModeAuto, false},
		{"位置参数同样算非交互", []string{"看看代码"}, tools.ModeReadOnly, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := mustApp(t, tc.args...)
			if a.gate.Mode != tc.mode {
				t.Errorf("审批模式 = %q，期望 %q", a.gate.Mode, tc.mode)
			}
			if got := a.gate.Ask != nil; got != tc.wantAsk {
				t.Errorf("Ask 回调设置 = %v，期望 %v（只有 ask 模式该问人）", got, tc.wantAsk)
			}
		})
	}
}

func TestPromptResolution(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"-p 给出任务", []string{"-p", "  带空格的任务  "}, "带空格的任务"},
		{"位置参数给出任务", []string{"列出", "文件"}, "列出 文件"},
		{"-p 优先于位置参数", []string{"-p", "来自 p", "来自位置"}, "来自 p"},
		{"空白 -p 回落到位置参数", []string{"-p", "   ", "位置任务"}, "位置任务"},
		{"都没有则进交互", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, prompt := mustApp(t, tc.args...)
			if prompt != tc.want {
				t.Errorf("prompt = %q，期望 %q", prompt, tc.want)
			}
			if (prompt != "") != (a.gate.Mode == tools.ModeReadOnly || a.gate.Mode == tools.ModeAuto) {
				t.Errorf("任务文本与审批模式不自洽: prompt=%q mode=%q", prompt, a.gate.Mode)
			}
		})
	}
}

func TestQuietAndJSON(t *testing.T) {
	t.Run("json 隐含 quiet 且过程输出丢弃", func(t *testing.T) {
		a, _ := mustApp(t, "-p", "x", "--json")
		if !a.cfg.Quiet {
			t.Error("--json 应隐含 Quiet")
		}
		if a.ui() != io.Discard {
			t.Error("--json 下过程输出必须走 io.Discard，否则会污染 stdout 的 JSON")
		}
	})
	t.Run("quiet 只压过程不出 JSON", func(t *testing.T) {
		a, _ := mustApp(t, "-p", "x", "--quiet")
		if !a.cfg.Quiet || a.json {
			t.Errorf("Quiet = %v, json = %v，期望 true/false", a.cfg.Quiet, a.json)
		}
		if a.ui() == io.Discard {
			t.Error("--quiet 不应把审批提示丢进 Discard，人还得看确认框")
		}
	})
	t.Run("默认有过程输出", func(t *testing.T) {
		a, _ := mustApp(t)
		if a.cfg.Quiet || a.json || a.ui() == io.Discard {
			t.Error("交互默认应打印过程")
		}
	})
}

func TestConfigPlumbing(t *testing.T) {
	a, _ := mustApp(t, "-p", "x", "-max-turns", "5", "-model", "deepseek-reasoner")
	if a.cfg.MaxIterations != 5 {
		t.Errorf("MaxIterations = %d，期望 5", a.cfg.MaxIterations)
	}
	if a.cfg.Model != "deepseek-reasoner" {
		t.Errorf("Model = %q，期望被 -model 覆盖", a.cfg.Model)
	}
	if a.cfg.ContextWindow != agent.ContextWindowFor("deepseek-reasoner") {
		t.Errorf("ContextWindow = %d，未按模型推断", a.cfg.ContextWindow)
	}

	b, _ := mustApp(t)
	if b.cfg.MaxIterations != 30 {
		t.Errorf("默认 MaxIterations = %d，期望 30", b.cfg.MaxIterations)
	}
	if b.cfg.Model != agent.DefaultModel {
		t.Errorf("默认 Model = %q，期望 %q", b.cfg.Model, agent.DefaultModel)
	}
	if b.cfg.MaxCostUSD != 0 {
		t.Errorf("默认不该有花费上限，得到 %v", b.cfg.MaxCostUSD)
	}

	c, _ := mustApp(t, "-max-cost", "2.5")
	if c.cfg.MaxCostUSD != 2.5 {
		t.Errorf("MaxCostUSD = %v，期望 2.5", c.cfg.MaxCostUSD)
	}
	if b.cfg.WorkingDir == "" {
		t.Error("WorkingDir 未设置")
	}
}

func TestSessionFlag(t *testing.T) {
	a, _ := mustApp(t, "-p", "x")
	if a.sessionID == "" {
		t.Error("默认应有会话 ID")
	}
	b, _ := mustApp(t, "-p", "x", "--no-session")
	if b.sessionID != "" {
		t.Errorf("--no-session 后 sessionID = %q，应为空", b.sessionID)
	}
	if b.provider.Name() != "deepseek" {
		t.Errorf("provider = %q，期望 deepseek", b.provider.Name())
	}
}

// TestConfirmRendersDiff 管的是确认框的最后一公里：工具算出了 diff，
// 还要逐行缩进打到终端，用户才看得到自己要放行什么。
func TestConfirmRendersDiff(t *testing.T) {
	a, _ := mustApp(t)
	var buf bytes.Buffer
	a.out = &buf
	a.in = bufio.NewReader(strings.NewReader("y\n"))

	if !a.confirm("write_file", "覆盖文件 foo.go（12 字节）\n@@ -1,2 +1,2 @@\n-old\n+new") {
		t.Fatal("回答 y 应放行")
	}
	out := buf.String()
	for _, want := range []string{"🔶 write_file", "     -old", "     +new", "     覆盖文件", "允许执行？"} {
		if !strings.Contains(out, want) {
			t.Errorf("确认输出缺少 %q；完整输出:\n%s", want, out)
		}
	}
}

// TestEveryToolIsDocumented 是提示词与工具表的同步门：
// 往 registry 加新工具却忘了写进系统提示词，模型就永远调不到它。
func TestEveryToolIsDocumented(t *testing.T) {
	a, _ := mustApp(t)

	want := []string{"delete_file", "edit_file", "list_directory", "move_file", "read_file",
		"repo_overview", "run_command", "search_code", "task", "update_plan", "write_file"}
	got := a.registry.Names()
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("注册的工具 = %v，期望 %v", got, want)
	}

	prompt := agent.SystemPrompt("/work")
	for _, name := range got {
		if !strings.Contains(prompt, name) {
			t.Errorf("工具 %q 已注册但系统提示词里没提到，模型不会调用它", name)
		}
	}
	for _, def := range a.registry.Definitions() {
		if !strings.Contains(prompt, def.Name) {
			t.Errorf("工具定义 %q 未出现在提示词中", def.Name)
		}
	}
	if !strings.Contains(prompt, "/work") {
		t.Error("系统提示词应包含工作目录")
	}
}

// TestSubAgentIsReadOnly 是安全门，两副面孔都要钉：
// 只读面一个改动类工具都不能有；可写面只能多 write_file 与 edit_file——
// 多了 run_command 就是给审批门开侧路（命令的破坏半径没法用 diff 表达），
// 多了 delete/move 让"改了哪个文件"变成"哪个文件没了"，多了 task 就能无限派孙 agent。
func TestSubAgentIsReadOnly(t *testing.T) {
	a, _ := mustApp(t)

	// 任何一副面孔都不许出现的东西
	neverAllowed := []string{"delete_file", "move_file", "run_command", "task", "update_plan"}

	for _, tc := range []struct {
		name     string
		writable bool
		want     string
		prompt   string
	}{
		{"只读面", false, "list_directory,read_file,repo_overview,search_code", agent.SubAgentPrompt(a.cfg.WorkingDir)},
		{"可写面", true, "edit_file,list_directory,read_file,repo_overview,search_code,write_file", agent.SubAgentWritePrompt(a.cfg.WorkingDir)},
	} {
		child := a.newSubAgent(tc.writable)
		got := strings.Join(child.ToolNames(), ",")
		if got != tc.want {
			t.Errorf("%s 工具 = %q，期望 %q", tc.name, got, tc.want)
		}
		for _, name := range child.ToolNames() {
			for _, forbidden := range neverAllowed {
				if name == forbidden {
					t.Errorf("%s 拿到了 %q：子 agent 能改动不该它改的东西，或再派孙 agent 递归下去", tc.name, name)
				}
			}
		}

		// 提示词必须和它真拿到的工具一致：少了会让它猜，多了会让它调根本没有的工具。
		if got, want := child.SystemPrompt(), tc.prompt; got != want {
			t.Errorf("%s 用错了系统提示词", tc.name)
		}
		childPrompt := child.SystemPrompt()
		for _, name := range child.ToolNames() {
			if !strings.Contains(childPrompt, name) {
				t.Errorf("%s 有工具 %q 但提示词没提到", tc.name, name)
			}
		}
		for _, forbidden := range neverAllowed {
			if strings.Contains(childPrompt, forbidden) {
				t.Errorf("%s 提示词里出现了 %q：它没有这个工具，照着说会误导模型", tc.name, forbidden)
			}
		}
		if strings.Contains(childPrompt, "Code Agent") {
			t.Errorf("%s 提示词与主 agent 共用同一份", tc.name)
		}
	}

	// 主 agent 必须能派子 agent
	if _, ok := a.registry.Get("task"); !ok {
		t.Error("主 agent 没有注册 task 工具")
	}

	// 两副面孔必须真的是两份提示词：复制粘贴出一份、只靠工具表区分，
	// 意味着改一处忘另一处时没人拦得住。
	if agent.SubAgentPrompt(a.cfg.WorkingDir) == agent.SubAgentWritePrompt(a.cfg.WorkingDir) {
		t.Error("只读面与可写面共用同一份提示词")
	}
}

// TestWritableSubAgentSharesGateAndNotFileState 钉住可写子任务的两条装配：
// 一道门（父子共用，子级每次写入照样逐个确认）、两份文件状态（父级读过文件不等于替子级开门）。
func TestWritableSubAgentSharesGateAndNotFileState(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("原始内容\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir) // 工具的工作目录取自进程 cwd，父子必须落在同一个临时目录里
	a, _ := mustApp(t)

	parent, ok := a.registry.Get("read_file")
	if !ok {
		t.Fatal("主 agent 没有 read_file")
	}
	if _, err := parent.Execute(context.Background(), map[string]any{"file_path": "a.txt"}); err != nil {
		t.Fatalf("父级读取失败: %v", err)
	}

	// 一次 subRegistry 调用就是一副面孔：里面 read_file 与 write_file 共用子级那份状态。
	reg := a.subRegistry(true)
	tool := mustTool(t, reg, "write_file")
	ctx := context.Background()

	// 一、FileState 不共用：门预放行，"先读后改"仍要拦住——
	// 已读记录属于父级那一份状态，父级读过一遍不能替子级背书。
	a.gate.Mode = tools.ModeAuto
	if _, err := tool.Execute(ctx, map[string]any{
		"file_path": "a.txt", "content": "子 agent 想改",
	}); err == nil || !strings.Contains(err.Error(), "read_file") {
		t.Errorf("父级读过不该替子级开门，期望先读后改的拒绝，实际 %v", err)
	}

	// 二、子级自己读一遍之后就放行：证明拦它的是状态，不是门
	if _, err := mustTool(t, reg, "read_file").Execute(ctx,
		map[string]any{"file_path": "a.txt"}); err != nil {
		t.Fatalf("子级读取失败: %v", err)
	}
	out, err := tool.Execute(ctx, map[string]any{
		"file_path": "a.txt", "content": "子 agent 想改",
	})
	if err != nil {
		t.Fatalf("子级自己读过之后应当可以写: %v", err)
	}
	if !strings.Contains(out, "已覆盖") {
		t.Errorf("out = %q", out)
	}

	// 三、回到逐个确认的模式：门是共用的，子级每次写入都要问 a.gate，
	// 且父级那次"批准派发"不能替它挡下这一次。
	a.gate.Mode = tools.ModeAsk
	asked := 0
	a.gate.Ask = func(title, detail string) bool {
		asked++
		return false // 用户拒绝
	}
	if _, err := tool.Execute(ctx, map[string]any{
		"file_path": "a.txt", "content": "再改一次",
	}); err == nil {
		t.Fatal("用户拒绝后写入却成功了")
	}
	if asked != 1 {
		t.Errorf("子 agent 的写入问了 %d 次，期望恰好 1 次（父子共用一道门）", asked)
	}
	// 拒绝必须真的没落盘
	data, err := os.ReadFile(filepath.Join(dir, "a.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "子 agent 想改" {
		t.Errorf("被拒绝的写入落盘了，文件内容是 %q", data)
	}
}

// mustTool 从注册表取工具，取不到就直接失败。
func mustTool(t *testing.T, reg *tools.Registry, name string) tools.Tool {
	t.Helper()
	tool, ok := reg.Get(name)
	if !ok {
		t.Fatalf("注册表里没有 %q", name)
	}
	return tool
}

func TestVersionFlagExitsCleanly(t *testing.T) {
	isolateEnv(t)
	a, prompt, err := newApp([]string{"-version"})
	if a != nil || prompt != "" {
		t.Error("-version 不应装配应用")
	}
	var e errExit
	if !errors.As(err, &e) {
		t.Fatalf("err = %v，期望 errExit", err)
	}
	if e.code != 0 {
		t.Errorf("退出码 = %d，期望 0", e.code)
	}
}

func TestProviderErrors(t *testing.T) {
	cases := []struct {
		name string
		env  map[string]string
		args []string
		want string
	}{
		{"没有任何密钥", map[string]string{}, nil, "缺少密钥"},
		{"anthropic 缺密钥", map[string]string{}, []string{"-provider", "anthropic"}, "ANTHROPIC_API_KEY"},
		{"openai 缺密钥", map[string]string{}, []string{"-provider", "openai"}, "OPENAI_API_KEY"},
		{"provider 名写错", map[string]string{"DEEPSEEK_API_KEY": "k"}, []string{"-provider", "gemini"}, "未知 provider"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			isolateEnv(t)
			for k := range tc.env {
				t.Setenv(k, tc.env[k])
			}
			_, _, err := newApp(tc.args)
			if err == nil {
				t.Fatal("期望报错，实际装配成功")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("错误 = %q，应包含 %q", err, tc.want)
			}
			var e errExit
			if errors.As(err, &e) {
				t.Errorf("配置错误应是真错误，不是 errExit: %v", err)
			}
		})
	}
}

func TestDefaultModelPerProvider(t *testing.T) {
	cases := []struct {
		provider string
		key      string
		want     string
	}{
		{"deepseek", "DEEPSEEK_API_KEY", agent.DefaultModel},
		{"anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-5"},
		{"openai", "OPENAI_API_KEY", "gpt-4o-mini"},
	}
	for _, tc := range cases {
		t.Run(tc.provider, func(t *testing.T) {
			isolateEnv(t)
			t.Setenv(tc.key, "test-key")
			a, _, err := newApp([]string{"-provider", tc.provider})
			if err != nil {
				t.Fatalf("newApp 失败: %v", err)
			}
			if a.cfg.Model != tc.want {
				t.Errorf("%s 默认模型 = %q，期望 %q", tc.provider, a.cfg.Model, tc.want)
			}
		})
	}
}

func TestDeepseekKeyMissingFallsBackToAnthropic(t *testing.T) {
	isolateEnv(t)
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	a, _, err := newApp(nil)
	if err != nil {
		t.Fatalf("newApp 失败: %v", err)
	}
	if a.provider.Name() != "anthropic" {
		t.Errorf("provider = %q，只有 Anthropic 密钥时应自动切过去", a.provider.Name())
	}
	if a.cfg.Model != "claude-sonnet-4-5" {
		t.Errorf("Model = %q，自动切换后模型也该跟着变", a.cfg.Model)
	}
}

func TestFlagParsingErrors(t *testing.T) {
	t.Run("-h 是正常退出", func(t *testing.T) {
		isolateEnv(t)
		_, _, err := newApp([]string{"-h"})
		if !errors.Is(err, flag.ErrHelp) {
			t.Errorf("err = %v，期望 flag.ErrHelp", err)
		}
	})
	t.Run("未知选项报错", func(t *testing.T) {
		isolateEnv(t)
		t.Setenv("DEEPSEEK_API_KEY", "k")
		_, _, err := newApp([]string{"--nope"})
		if err == nil {
			t.Fatal("未知选项应报错，静默忽略会让用户以为选项生效了")
		}
		var e errExit
		if errors.As(err, &e) {
			t.Errorf("选项错误应走错误分支: %v", err)
		}
	})
	t.Run("选项值缺失", func(t *testing.T) {
		isolateEnv(t)
		t.Setenv("DEEPSEEK_API_KEY", "k")
		if _, _, err := newApp([]string{"-max-turns"}); err == nil {
			t.Error("-max-turns 后面没值应报错")
		}
	})
}

func TestReportText(t *testing.T) {
	res := &agent.Result{
		Iterations:  4,
		Usage:       llm.Usage{InputTokens: 1200, OutputTokens: 300},
		CostUSD:     0.0123,
		Compactions: 2,
	}
	out := report(res)
	for _, want := range []string{"4 轮", "1200", "300", "$0.0123", "折叠过 2 次"} {
		if !strings.Contains(out, want) {
			t.Errorf("报告缺少 %q:\n%s", want, out)
		}
	}

	clean := report(&agent.Result{Iterations: 1, Usage: llm.Usage{InputTokens: 1, OutputTokens: 1}})
	if strings.Contains(clean, "折叠") {
		t.Errorf("没折叠过历史却提到折叠:\n%s", clean)
	}
}

// ============================================================================
// 外部工具（MCP）装配
// ============================================================================

// fakeMCPServerEnv 让本测试二进制在子进程里充当一台 MCP 服务器。
// 这样装配链的测试不需要机器上装 npx 或 python，也不碰网络。
const fakeMCPServerEnv = "GO_AGENT_TEST_FAKE_MCP"

func serveFakeMCPStdio() {
	in := bufio.NewScanner(os.Stdin)
	for in.Scan() {
		var req struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if err := json.Unmarshal(in.Bytes(), &req); err != nil || len(req.ID) == 0 {
			continue // 通知不用回应，但必须读掉，否则对端以为我们挂了
		}
		var result any
		switch req.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "fake", "version": "1"},
			}
		case "tools/list":
			result = map[string]any{"tools": []any{map[string]any{
				"name":        "ping",
				"description": "回一个 pong",
				"inputSchema": map[string]any{
					"type":       "object",
					"properties": map[string]any{"msg": map[string]any{"type": "string"}},
					"required":   []string{"msg"},
				},
			}}}
		case "tools/call":
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "pong"}}}
		default:
			continue
		}
		frame, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
		os.Stdout.Write(append(frame, '\n'))
	}
	os.Exit(0)
}

// writeMCPConfig 写一份指向"本测试二进制自己"的服务器配置。
func writeMCPConfig(t *testing.T, body map[string]any) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if servers, ok := body["mcpServers"].(map[string]any); ok {
		for _, v := range servers {
			if s, ok := v.(map[string]any); ok {
				if _, hasCmd := s["command"]; !hasCmd {
					s["command"] = exe
				}
				if _, hasArgs := s["args"]; !hasArgs {
					s["args"] = []string{"-test.run", "^TestMCPToolsAreImported$", "-test.timeout", "60s"}
				}
			}
		}
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "mcp.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestMCPToolsAreImported 走通整条链：配置 → 起子进程 → 握手 → 注册 → 确认门 → 退出收尸。
func TestMCPToolsAreImported(t *testing.T) {
	if os.Getenv(fakeMCPServerEnv) == "1" {
		serveFakeMCPStdio() // 子进程模式
		return
	}

	path := writeMCPConfig(t, map[string]any{"mcpServers": map[string]any{
		"fake": map[string]any{"env": map[string]string{fakeMCPServerEnv: "1"}, "timeout": "20s"},
	}})

	a, _ := mustApp(t, "-mcp-config", path)
	defer a.closeMCPServers()

	tool, ok := a.registry.Get("mcp_fake_ping")
	if !ok {
		t.Fatalf("远端工具没导入，当前 = %v", a.registry.Names())
	}
	// 本地工具一个都不能少：外部工具是加法，不是替换
	if got := len(a.registry.Names()); got != 12 {
		t.Errorf("工具总数 = %d，期望 11 个本地 + 1 个远端", got)
	}

	// 服务器没声明 readOnlyHint，就必须按会改动注册（缺省从严）
	if !tool.Mutating() {
		t.Error("未声明只读的远端工具被当成只读注册了，审批门会漏过它")
	}
	def := tool.Definition()
	if !strings.Contains(def.Description, "fake") {
		t.Errorf("工具定义应标出来自哪台服务器: %q", def.Description)
	}
	if !strings.Contains(string(def.RawSchema), `"msg"`) {
		t.Errorf("远端参数 schema 没原样透传: %s", def.RawSchema)
	}

	// 子智能体不许拿到外部工具：两副面孔的注册表都只由本地工具组成
	for _, writable := range []bool{false, true} {
		for _, name := range a.newSubAgent(writable).ToolNames() {
			if strings.HasPrefix(name, "mcp_") {
				t.Errorf("子 agent（可写=%v）拿到了外部工具 %s", writable, name)
			}
		}
	}

	// 真调一次：确认门在 main 装配的这条链上确实站着
	asked := 0
	a.gate.Ask = func(title, detail string) bool {
		asked++
		if !strings.Contains(detail, "mcp_fake_ping") {
			t.Errorf("确认框要写清调的是哪个远端工具: %q", detail)
		}
		return true
	}
	out, err := tool.Execute(t.Context(), map[string]any{"msg": "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if out != "pong" {
		t.Errorf("远端返回 = %q", out)
	}
	if asked != 1 {
		t.Errorf("改动类远端调用应确认一次，实际 %d 次", asked)
	}

	if len(a.mcpClients) != 1 {
		t.Fatalf("已连接的服务器数量 = %d", len(a.mcpClients))
	}
	client := a.mcpClients[0]
	a.closeMCPServers()
	if len(a.mcpClients) != 0 {
		t.Errorf("关闭后还留着 %d 个客户端", len(a.mcpClients))
	}
	// 连接必须报告已断：留着一个"看起来还能用"的客户端，后续调用会挂在管道上
	if client.Err() == nil {
		t.Error("Close 之后客户端仍报告可用")
	}
}

func TestMCPServerProblemsDoNotBreakStartup(t *testing.T) {
	cases := []struct {
		name    string
		server  map[string]any
		wantMCP bool
	}{
		{"disabled 的服务器不启动", map[string]any{"disabled": true}, false},
		{"起不来的服务器只警告，不拦人", map[string]any{"command": "/nonexistent/mcp-server", "args": []string{}}, false},
		{"环境变量没设上也照样启动", map[string]any{}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := map[string]any{}
			for k, v := range tc.server {
				server[k] = v
			}
			if tc.wantMCP {
				// 这一例要真起进程，让它走 self-exec 假服务器
				server["env"] = map[string]string{fakeMCPServerEnv: "1",
					"FAKE_TOKEN": "${GO_AGENT_TEST_NOT_SET}"}
			}
			a, _ := mustApp(t, "-mcp-config", writeMCPConfig(t,
				map[string]any{"mcpServers": map[string]any{"srv": server}}))
			defer a.closeMCPServers()

			want := 11
			if tc.wantMCP {
				want = 12
			}
			// 本地 11 个工具永远在：外部工具出问题不能把 CLI 弄瘫
			if got := len(a.registry.Names()); got != want {
				t.Errorf("工具数 = %d，期望 %d（当前 %v）", got, want, a.registry.Names())
			}
			if hasMCP := a.registry.Has("mcp_srv_ping"); hasMCP != tc.wantMCP {
				t.Errorf("外部工具导入 = %v，期望 %v", hasMCP, tc.wantMCP)
			}
			if len(a.mcpClients) != 0 && !tc.wantMCP {
				t.Errorf("不该有存活的连接: %d", len(a.mcpClients))
			}
		})
	}
}

// TestMCPStartupTellsYouWhatGotImported：启动那几行是人判断"配置到底生效没有"的唯一依据，
// 所以导入了什么、哪台没起来、哪个变量没设上，都得点名说清。
func TestMCPStartupTellsYouWhatGotImported(t *testing.T) {
	isolateEnv(t)
	var buf bytes.Buffer
	a := &app{out: &buf, registry: tools.NewRegistry(), gate: tools.NewGate(tools.ModeAsk)}
	defer a.closeMCPServers()

	path := writeMCPConfig(t, map[string]any{"mcpServers": map[string]any{
		"good": map[string]any{"env": map[string]string{
			fakeMCPServerEnv: "1",
			"TOKEN":          "${GO_AGENT_TEST_NOT_SET}",
		}},
		"bad": map[string]any{"command": "/nonexistent/mcp-server", "args": []string{}},
	}})
	if err := a.connectMCPServers(path); err != nil {
		t.Fatal(err)
	}

	said := buf.String()
	for _, want := range []string{
		"GO_AGENT_TEST_NOT_SET", // 环境变量没设上，当场说，别让人去猜远端为什么一直失败
		"mcp_good_ping",         // 导入了哪个工具要说得出名字
		"bad 启动失败",              // 坏的那台点名，不能只说"部分服务器未就绪"
	} {
		if !strings.Contains(said, want) {
			t.Errorf("启动输出里应包含 %q，实际：\n%s", want, said)
		}
	}
	if len(a.mcpClients) != 1 {
		t.Errorf("存活的连接 = %d，期望只有 good 那台", len(a.mcpClients))
	}
	if !a.registry.Has("mcp_good_ping") || a.registry.Has("mcp_bad_ping") {
		t.Errorf("导入结果与预期不符: %v", a.registry.Names())
	}
}

// TestMCPBadConfigStopsStartup：配置写错了要当场失败。
// 带着坏配置继续跑，人看到的是"没有外部工具"，而不是"配置没生效"。
func TestMCPBadConfigStopsStartup(t *testing.T) {
	isolateEnv(t)
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	for _, bad := range []string{
		`{"mcpServers": {"非法 名字": {"command": "/bin/true"}}}`,
		`{"mcpServers": {"srv": {"args": []}}}`,
		`{"mcpServers": {"srv": {"command": "/bin/true", "timeout": "soon"}}}`,
		`{"mcpServers": {"srv": {"command": "/bin/true"}, "extra": 1}}`,
	} {
		path := filepath.Join(t.TempDir(), "mcp.json")
		if err := os.WriteFile(path, []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, _, err := newApp([]string{"-mcp-config", path}); err == nil {
			t.Errorf("坏配置本该报错: %s", bad)
		}
	}
}
