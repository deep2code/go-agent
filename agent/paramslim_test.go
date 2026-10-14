package agent

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"go-agent/llm"
	"go-agent/tools"
)

// ============================================================================
// 脚手架
// ============================================================================

// writeRound 造一轮"模型调 write_file 写了 body，并拿到了成功结果"。
func writeRound(id, filePath, body string) []llm.Message {
	isErr := false
	return []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: id, Name: "write_file",
			Input: map[string]any{"file_path": filePath, "content": body},
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: "tool_result", ToolUseID: id,
			Content: "✅ 已创建 " + filePath, IsError: &isErr,
		}}},
	}
}

// agentWithWriter 装配一个能看到真写入/读取工具的 agent；瘦身判定认的是工具自己的声明，
// 用桩子就等于在测桩子。
func agentWithWriter(t *testing.T, dir string) *Agent {
	t.Helper()
	gate := tools.NewGate(tools.ModeAuto)
	state := tools.NewFileState()
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(
		tools.NewWriteFileTool(dir, gate, state),
		tools.NewEditFileTool(dir, gate, state),
		tools.NewReadFileTool(dir, state),
		tools.NewRunCommandTool(dir, gate, time.Second, io.Discard),
	), func(cfg *Config) {
		cfg.Cache = cacheIn(t)
		cfg.Writer = io.Discard
	})
	return a
}

// callID 生成第 i 次调用的 tool_use ID（a、b、c…）。
func callID(i int) string { return "toolu_" + string(rune('a'+i)) }

func paramOf(t *testing.T, messages []llm.Message, id, name string) string {
	t.Helper()
	for _, m := range messages {
		for _, b := range m.Content {
			if b.Type == "tool_use" && b.ID == id {
				v, _ := b.Input[name].(string)
				return v
			}
		}
	}
	t.Fatalf("历史里找不到 ID 为 %s 的 tool_use", id)
	return ""
}

// rounds 造 n 轮写入，返回从 0 起的每一轮消息。
func writeRounds(n int, body string) []llm.Message {
	history := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("写文件")}}}
	for i := 0; i < n; i++ {
		history = append(history, writeRound(callID(i), "f.go", body)...)
	}
	return history
}

// ============================================================================
// 大正文参数瘦身
// ============================================================================

func TestSlimBodyParamsDropsWrittenContent(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	body := strings.Repeat("package main\n", 400) // 5600 字节，远超门槛

	history := writeRounds(6, body)
	freed := a.SlimBodyParams(history)
	if freed <= 0 {
		t.Fatal("六次成功写入之后一个字节都没省下来")
	}

	got := paramOf(t, history, callID(0), "content")
	if !strings.Contains(got, "已落盘在 f.go") {
		t.Errorf("早期参数没换成回读说明: %q", got)
	}
	if len(got) > 300 {
		t.Errorf("瘦身后的说明太长（%d 字节），省不下什么", len(got))
	}
	// 最近 l2KeepRecent 次保持原样：那几轮正是模型下一步动作的参照
	if got := paramOf(t, history, callID(5), "content"); got != body {
		t.Error("最近一次写入的正文被砍了")
	}
	// file_path 必须留着，否则那句"已落盘在 …"就是空头支票
	if got := paramOf(t, history, callID(0), "file_path"); got != "f.go" {
		t.Errorf("file_path 不该被动，实际 %q", got)
	}
}

func TestSlimBodyParamsKeepsFailedWrite(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	body := strings.Repeat("x", l2ParamFrom+500)

	isErr := true
	history := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("写文件")}}}
	for i := 0; i < 6; i++ {
		id := callID(i)
		history = append(history,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
				Type: "tool_use", ID: id, Name: "write_file",
				Input: map[string]any{"file_path": "f.go", "content": body},
			}}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
				Type: "tool_result", ToolUseID: id, Content: "路径越界，写入失败", IsError: &isErr,
			}}})
	}

	if freed := a.SlimBodyParams(history); freed != 0 {
		t.Errorf("失败的写入也被瘦身了（省下 %d）：参数是模型下一轮重试的唯一依据", freed)
	}
	if got := paramOf(t, history, callID(0), "content"); got != body {
		t.Error("失败调用的正文被改动了")
	}
}

// 还在等审批（或被打断、结果还没回来）的调用绝不能换：diff 要从参数里生成。
func TestSlimBodyParamsKeepsCallWithoutResult(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	body := strings.Repeat("y", l2ParamFrom+500)

	history := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("写文件")}}}
	// 第一次调用故意不配结果，且它足够老，能过"最近几次"那道门
	history = append(history, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
		Type: "tool_use", ID: callID(0), Name: "write_file",
		Input: map[string]any{"file_path": "f.go", "content": body},
	}}})
	for i := 1; i < 6; i++ {
		history = append(history, writeRound(callID(i), "f.go", body)...)
	}
	a.SlimBodyParams(history)

	if got := paramOf(t, history, callID(0), "content"); got != body {
		t.Error("没有结果的调用被瘦身了")
	}
	if got := paramOf(t, history, callID(1), "content"); len(got) > 300 {
		t.Error("同一批里已成功的早期调用没被瘦身——门做成了全有或全无")
	}
}

// 命令参数再长也不能换：那是"执行过什么"的唯一记录，而且没有任何文件里存着它。
func TestSlimBodyParamsIgnoresUndeclaredTools(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	cmd := "echo " + strings.Repeat("a", l2ParamFrom+500)

	isErr := false
	history := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("跑命令")}}}
	for i := 0; i < 6; i++ {
		id := callID(i)
		history = append(history,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
				Type: "tool_use", ID: id, Name: "run_command",
				Input: map[string]any{"command": cmd},
			}}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
				Type: "tool_result", ToolUseID: id, Content: "✅ 退出码 0", IsError: &isErr,
			}}})
	}

	if freed := a.SlimBodyParams(history); freed != 0 {
		t.Errorf("未声明大正文参数的工具被瘦身了（省下 %d）", freed)
	}
	if got := paramOf(t, history, callID(0), "command"); got != cmd {
		t.Error("run_command 的参数被改写了")
	}
}

// edit_file 只换 new_string，old_string 必须整份留着：磁盘上已经没有那段原文了。
func TestSlimBodyParamsKeepsEditOldString(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	oldStr := strings.Repeat("old", 700)
	newStr := strings.Repeat("new", 700)

	isErr := false
	history := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("改文件")}}}
	for i := 0; i < 6; i++ {
		id := callID(i)
		history = append(history,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
				Type: "tool_use", ID: id, Name: "edit_file",
				Input: map[string]any{"file_path": "f.go", "old_string": oldStr, "new_string": newStr},
			}}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
				Type: "tool_result", ToolUseID: id, Content: "✅ 已修改 f.go", IsError: &isErr,
			}}})
	}
	a.SlimBodyParams(history)

	if got := paramOf(t, history, callID(0), "new_string"); len(got) > 300 {
		t.Errorf("new_string 没瘦身（%d 字节）：它已经在文件里，read_file 能读回来", len(got))
	}
	if got := paramOf(t, history, callID(0), "old_string"); got != oldStr {
		t.Errorf("old_string 被动了（%d 字节）", len(got))
	}
}

// 门槛之下不值得动：省不下几个字节，却让那次改动在历史里更难追。
func TestSlimBodyParamsLeavesSmallContent(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	body := strings.Repeat("z", l2ParamFrom-1)

	history := writeRounds(6, body)
	if freed := a.SlimBodyParams(history); freed != 0 {
		t.Errorf("未过门槛的参数被瘦身了（省下 %d）", freed)
	}
}

// Input 是引用类型：就地改会让同一张参数表的其他持有者看到变了的内容。
func TestSlimBodyParamsDoesNotMutateSharedMap(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	body := strings.Repeat("w", l2ParamFrom+500)
	shared := map[string]any{"file_path": "f.go", "content": body}

	history := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("写文件")}}}
	isErr := false
	for i := 0; i < 6; i++ {
		id := callID(i)
		history = append(history,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
				Type: "tool_use", ID: id, Name: "write_file", Input: shared,
			}}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
				Type: "tool_result", ToolUseID: id, Content: "✅ 已创建", IsError: &isErr,
			}}})
	}
	if freed := a.SlimBodyParams(history); freed == 0 {
		t.Fatal("前置条件：这一批应当被瘦身，否则本用例什么都没证明")
	}
	if got, _ := shared["content"].(string); got != body {
		t.Error("瘦身改动了调用方持有的那张参数表")
	}
}

// 瘦身后历史的配对结构必须毫发无损：它只换参数值，不碰块、不碰 ID、不碰顺序。
func TestSlimmedHistoryStillPairs(t *testing.T) {
	a := agentWithWriter(t, t.TempDir())
	body := strings.Repeat("q", l2ParamFrom+800)

	history := writeRounds(6, body)
	a.SlimBodyParams(history)
	assertValid(t, RepairHistory(history))
}

// 瘦身必须发生在估算之前（和 L2 同一侧）：放在折叠之后，超预算的判断已经按肥历史做完了，
// 那一轮折叠就是白做的。
func TestSlimRunsBeforeBudgetCheck(t *testing.T) {
	dir := t.TempDir()
	gate := tools.NewGate(tools.ModeAuto)
	state := tools.NewFileState()
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(
		tools.NewWriteFileTool(dir, gate, state),
		tools.NewReadFileTool(dir, state),
	), func(cfg *Config) {
		cfg.Cache = cacheIn(t)
		cfg.ContextWindow = 20000
		cfg.MaxTokens = 8000 // 历史预算 8000：不瘦身就一定超
		cfg.Writer = io.Discard
	})

	a.history = writeRounds(8, strings.Repeat("package main\n", 400))

	freed, err := a.prepareForSend(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if freed != 0 {
		t.Errorf("动用了折叠（省下 %d token）：参数瘦身没赶在预算判断之前", freed)
	}
	if got := paramOf(t, a.history, callID(0), "content"); len(got) > 300 {
		t.Error("prepareForSend 没有瘦身早期写入参数")
	}
}

// 端到端：真正发出去的那份请求里，早期写入参数必须已经换成说明——省的是每一轮都要重付的字节。
func TestRunSendsSlimmedParams(t *testing.T) {
	dir := t.TempDir()
	body := strings.Repeat("package main\n", 200) // 2800 字节

	var steps []scriptedStep
	for i := 0; i < 6; i++ {
		steps = append(steps, toolStep(callID(i), "write_file",
			map[string]any{"file_path": fmt.Sprintf("f%d.go", i), "content": body}, ""))
	}
	steps = append(steps, stopReply("完成", 0, 0))

	p := &scriptedProvider{steps: steps}
	a, _ := newAgent(t, p, registryWith(
		tools.NewWriteFileTool(dir, tools.NewGate(tools.ModeAuto), tools.NewFileState()),
		tools.NewReadFileTool(dir, tools.NewFileState()),
	), func(cfg *Config) {
		cfg.Cache = cacheIn(t)
		cfg.MaxIterations = 8
		cfg.Writer = io.Discard
	})

	if _, err := a.Run(context.Background(), "写六个文件"); err != nil {
		t.Fatal(err)
	}
	// 回读路径站不站得住，取决于文件真的在磁盘上
	if _, err := os.Stat(filepath.Join(dir, "f0.go")); err != nil {
		t.Fatalf("用例前提失效：f0.go 没写出来: %v", err)
	}
	last := p.requests[len(p.requests)-1]
	got := paramOf(t, last.Messages, callID(0), "content")
	if got == body {
		t.Error("最后一次请求仍在重复发送早期写入的全文正文")
	}
	if !strings.Contains(got, "f0.go") {
		t.Errorf("发出去的说明里没点名文件，模型无从回读: %q", got)
	}
}

// ============================================================================
// 重复读取省略
// ============================================================================

// seedGoFile 写一个足够大的文件，让 read_file 的结果过上去重门槛。
func seedGoFile(t *testing.T, dir string) string {
	t.Helper()
	var sb strings.Builder
	for i := 0; i < 80; i++ {
		sb.WriteString("func demo() { return }\n")
	}
	path := filepath.Join(dir, "a.go")
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func readOut(t *testing.T, dir string, input map[string]any) string {
	t.Helper()
	out, err := tools.NewReadFileTool(dir, tools.NewFileState()).Execute(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// 同一个文件、同一份内容读第二遍：全文就在上方历史里，不必再发一遍。
func TestDuplicateReadIsOmitted(t *testing.T) {
	dir := t.TempDir()
	seedGoFile(t, dir)
	a := agentWithWriter(t, dir)
	input := map[string]any{"file_path": "a.go"}

	first := readOut(t, dir, input)
	if len(first) < dupMinBytes {
		t.Fatalf("用例前提失效：这份读取结果只有 %d 字节，够不上去重门槛", len(first))
	}
	a.history = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("看看 a.go")}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file", Input: input,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", first)}},
	}

	second := readOut(t, dir, input)
	notice := a.duplicateNotice("read_file", input, second)
	if notice == "" {
		t.Fatal("逐字相同的重复读取没有省略")
	}
	if !strings.Contains(notice, "a.go") || !strings.Contains(notice, "read_file") {
		t.Errorf("省略说明要点名是哪个文件: %q", notice)
	}
	if len(notice) > len(second)/2 {
		t.Errorf("省略说明本身太长（%d 对 %d）", len(notice), len(second))
	}
}

// 分段读取的另一个范围不是重复：start_line 不同就必须各发各的。
func TestDuplicateReadSeparatesRanges(t *testing.T) {
	dir := t.TempDir()
	seedGoFile(t, dir)
	a := agentWithWriter(t, dir)

	first := readOut(t, dir, map[string]any{"file_path": "a.go", "start_line": "1"})
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file",
			Input: map[string]any{"file_path": "a.go", "start_line": "1"},
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", first)}},
	}

	second := readOut(t, dir, map[string]any{"file_path": "a.go", "start_line": "41"})
	if got := a.duplicateNotice("read_file",
		map[string]any{"file_path": "a.go", "start_line": "41"}, second); got != "" {
		t.Errorf("换了范围却被判成重复: %q", got)
	}
}

// 文件被改过就不是重复：新的判定必须照常把全文发出去。
func TestDuplicateReadWhenContentChanged(t *testing.T) {
	dir := t.TempDir()
	path := seedGoFile(t, dir)
	a := agentWithWriter(t, dir)
	input := map[string]any{"file_path": "a.go"}

	first := readOut(t, dir, input)
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file", Input: input,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", first)}},
	}

	if err := os.WriteFile(path, []byte(strings.Repeat("changed line\n", 80)), 0o644); err != nil {
		t.Fatal(err)
	}
	second := readOut(t, dir, input)
	if got := a.duplicateNotice("read_file", input, second); got != "" {
		t.Errorf("内容已经变了还被省略，模型会看不到改动: %q", got)
	}
}

// 历史里那一份已经被指针化（只剩头尾 + 路径）时不能省略：
// 那等于把模型指向一份它其实看不见的全文。
func TestDuplicateReadAfterPointerizeIsNotOmitted(t *testing.T) {
	dir := t.TempDir()
	seedGoFile(t, dir)
	a := agentWithWriter(t, dir)
	input := map[string]any{"file_path": "a.go"}

	full := readOut(t, dir, input)
	governed := a.cache.Pointer("toolu_1", full)
	if governed == full {
		t.Fatal("用例前提失效：这份结果没有被指针化")
	}
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file", Input: input,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", governed)}},
	}

	if got := a.duplicateNotice("read_file", input, full); got != "" {
		t.Errorf("历史里那一份已经是残缺的指针，还被当成可见全文: %q", got)
	}
}

// 写类工具的结果是"改动确实发生过"的凭据，逐字相同也要逐条留着。
func TestDuplicateReadIgnoresMutatingTool(t *testing.T) {
	body := strings.Repeat("m", dupMinBytes+100)
	write := &fakeTool{name: "write_file", mutating: true, result: body}
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(write), func(cfg *Config) {
		cfg.Writer = io.Discard
	})
	input := map[string]any{"file_path": "a.go"}
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "write_file", Input: input,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", body)}},
	}

	if got := a.duplicateNotice("write_file", input, body); got != "" {
		t.Errorf("写类工具的结果被省略了: %q", got)
	}
}

// 短结果不值得省略：省不下什么，却多让模型背一个"这次怎么没返回内容"。
func TestDuplicateReadIgnoresSmallResults(t *testing.T) {
	body := strings.Repeat("s", dupMinBytes-1)
	read := &fakeTool{name: "read_file", result: body}
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(read), func(cfg *Config) {
		cfg.Writer = io.Discard
	})
	input := map[string]any{"file_path": "a.go"}
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file", Input: input,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", body)}},
	}

	if got := a.duplicateNotice("read_file", input, body); got != "" {
		t.Error("短结果也被省略了")
	}
}

// 上一次是错误结果时不能拿来当"已经看过同样的内容"。
func TestDuplicateReadIgnoresErrorResult(t *testing.T) {
	body := strings.Repeat("e", dupMinBytes+100)
	read := &fakeTool{name: "read_file", result: body}
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(read), func(cfg *Config) {
		cfg.Writer = io.Discard
	})
	input := map[string]any{"file_path": "a.go", "start_line": "1"}
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file", Input: input,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{errorResult("toolu_1", body)}},
	}

	if got := a.duplicateNotice("read_file", input, body); got != "" {
		t.Errorf("那次是失败结果，却被当成可见全文: %q", got)
	}
}

// 参数只在截断点之后才不同时，绝不能被判成同一次调用——那会省掉真数据。
func TestDuplicateReadSeparatesArgsThatSharePrefix(t *testing.T) {
	body := strings.Repeat("d", dupMinBytes+100)
	read := &fakeTool{name: "read_file", result: body}
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(read), func(cfg *Config) {
		cfg.Writer = io.Discard
	})
	long := strings.Repeat("p", 4000)
	first := map[string]any{"file_path": long + "-a.go"}
	a.history = []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: "tool_use", ID: "toolu_1", Name: "read_file", Input: first,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("toolu_1", body)}},
	}

	if got := a.duplicateNotice("read_file", map[string]any{"file_path": long + "-b.go"}, body); got != "" {
		t.Errorf("两个不同的文件被判成同一次调用: %q", got)
	}
	if got := a.duplicateNotice("read_file", first, body); got == "" {
		t.Error("同名同参的重复没被认出来")
	}
}

func readUse(id string, input map[string]any) llm.ContentBlock {
	return llm.ContentBlock{Type: "tool_use", ID: id, Name: "read_file", Input: input}
}

// 省略的结果仍然必须进历史、且配对完好——不能把 tool_result 整条省掉。
func TestDuplicateReadKeepsPairing(t *testing.T) {
	dir := t.TempDir()
	seedGoFile(t, dir)
	read := tools.NewReadFileTool(dir, tools.NewFileState())
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(read), func(cfg *Config) {
		cfg.Cache = cacheIn(t)
		cfg.Writer = io.Discard
	})
	input := map[string]any{"file_path": "a.go"}

	first := a.runOneQuiet(context.Background(), readUse("toolu_1", input), read)
	a.history = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("把 a.go 看两遍")}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{readUse("toolu_1", input)}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{first}},
	}
	second := a.runOneQuiet(context.Background(), readUse("toolu_2", input), read)

	if second.Type != "tool_result" || second.ToolUseID != "toolu_2" {
		t.Fatalf("省略后的返回不再是配对的 tool_result: %+v", second)
	}
	if !strings.Contains(toolResultText(second), "逐字相同") {
		t.Errorf("第二次返回的仍是全文（%d 字节）", len(toolResultText(second)))
	}
	if len(toolResultText(second))*2 > len(toolResultText(first)) {
		t.Error("省略说明太长，省下的都还回去了")
	}
	assertValid(t, RepairHistory(append(a.history,
		llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{readUse("toolu_2", input)}},
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{second}})))
}
