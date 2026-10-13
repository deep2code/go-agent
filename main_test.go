package main

import (
	"errors"
	"flag"
	"io"
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

// TestEveryToolIsDocumented 是提示词与工具表的同步门：
// 往 registry 加新工具却忘了写进系统提示词，模型就永远调不到它。
func TestEveryToolIsDocumented(t *testing.T) {
	a, _ := mustApp(t)

	want := []string{"delete_file", "edit_file", "list_directory", "move_file",
		"read_file", "run_command", "search_code", "update_plan", "write_file"}
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
