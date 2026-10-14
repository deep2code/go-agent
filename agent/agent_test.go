package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"go-agent/llm"
	"go-agent/log"
	"go-agent/tools"
)

// ReAct 循环的分支很多：工具成功/失败/未知、参数被截断、输出撞上限、
// 用户中断、服务端判定超限。这些分支只能靠脚本化的 Provider 来验，
// 因为真 API 不会配合你复现同一种失败。

func TestMain(m *testing.M) {
	log.SetLevel(log.ERROR)
	os.Exit(m.Run())
}

// ============================================================================
// 脚本化 Provider
// ============================================================================

type scriptedStep struct {
	resp *llm.ChatResponse
	err  error
}

type scriptedProvider struct {
	steps    []scriptedStep
	requests []llm.ChatRequest // 每次调用时的历史快照
	call     int
	onCall   func(call int)
}

func (p *scriptedProvider) Name() string { return "scripted" }

func (p *scriptedProvider) SetHTTPLogger(*log.HTTPLogger) {}

func (p *scriptedProvider) StreamingChat(
	ctx context.Context,
	req llm.ChatRequest,
	onText func(text string),
) (*llm.ChatResponse, error) {
	n := p.call
	p.call++
	if p.onCall != nil {
		p.onCall(n)
	}
	p.requests = append(p.requests, snapshotRequest(req))

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if n >= len(p.steps) {
		return nil, fmt.Errorf("脚本只有 %d 步，第 %d 步无响应", len(p.steps), n+1)
	}
	step := p.steps[n]
	if step.err != nil {
		return nil, step.err
	}
	if onText != nil {
		for _, b := range step.resp.Content {
			if b.Type == "text" && b.Text != "" {
				onText(b.Text)
			}
		}
	}
	return step.resp, nil
}

// snapshotRequest 复制历史，防止后续 append 复用底层数组把快照改掉。
func snapshotRequest(req llm.ChatRequest) llm.ChatRequest {
	out := req
	out.Messages = make([]llm.Message, len(req.Messages))
	for i, m := range req.Messages {
		blocks := make([]llm.ContentBlock, len(m.Content))
		copy(blocks, m.Content)
		out.Messages[i] = llm.Message{Role: m.Role, Content: blocks}
	}
	return out
}

func stopReply(text string, in, out int) scriptedStep {
	return scriptedStep{resp: &llm.ChatResponse{
		FinishReason: llm.FinishStop,
		Content:      []llm.ContentBlock{{Type: "text", Text: text}},
		Usage:        llm.Usage{InputTokens: in, OutputTokens: out},
	}}
}

func toolStep(id, name string, input map[string]any, pre string) scriptedStep {
	var blocks []llm.ContentBlock
	if pre != "" {
		blocks = append(blocks, llm.ContentBlock{Type: "text", Text: pre})
	}
	blocks = append(blocks, llm.ContentBlock{Type: "tool_use", ID: id, Name: name, Input: input})
	return scriptedStep{resp: &llm.ChatResponse{
		FinishReason: llm.FinishToolCalls,
		Content:      blocks,
		Usage:        llm.Usage{InputTokens: 100, OutputTokens: 20},
	}}
}

func lengthStep(content []llm.ContentBlock) scriptedStep {
	return scriptedStep{resp: &llm.ChatResponse{
		FinishReason: llm.FinishLength,
		Content:      content,
		Usage:        llm.Usage{InputTokens: 100, OutputTokens: 500},
	}}
}

// ============================================================================
// 假工具
// ============================================================================

type fakeTool struct {
	name     string
	mutating bool
	result   string
	err      error
	calls    int
	lastIn   map[string]any
	onCall   func(calls int) // 用于在执行中途制造中断
}

func (f *fakeTool) Name() string   { return f.name }
func (f *fakeTool) Mutating() bool { return f.mutating }
func (f *fakeTool) Definition() llm.ToolDef {
	return llm.ToolDef{Name: f.name, Description: "测试用工具"}
}
func (f *fakeTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	f.calls++
	f.lastIn = input
	if f.onCall != nil {
		f.onCall(f.calls)
	}
	if f.err != nil {
		return "", f.err
	}
	return f.result, nil
}

// ============================================================================
// 测试脚手架
// ============================================================================

func newAgent(t *testing.T, p llm.Provider, rs *tools.Registry, tune func(*Config)) (*Agent, *Config) {
	t.Helper()
	cfg := DefaultConfig(t.TempDir(), "deepseek-chat")
	cfg.MaxIterations = 6
	cfg.Writer = io.Discard
	if tune != nil {
		tune(&cfg)
	}
	a := New(cfg, p, rs)
	return a, &cfg
}

func registryWith(ts ...tools.Tool) *tools.Registry {
	r := tools.NewRegistry()
	for _, t := range ts {
		r.Register(t)
	}
	return r
}

// quietConfig 给不需要看过程输出的用例一份不写 stdout 的配置。
func quietConfig(dir string) Config {
	c := DefaultConfig(dir, "deepseek-chat")
	c.MaxIterations = 4
	c.Writer = io.Discard
	return c
}

func lastUserText(t *testing.T, req llm.ChatRequest) string {
	t.Helper()
	if len(req.Messages) == 0 {
		t.Fatal("请求里没有消息")
	}
	m := req.Messages[len(req.Messages)-1]
	if m.Role != llm.RoleUser {
		t.Fatalf("最后一条不是 user: %s", m.Role)
	}
	var parts []string
	for _, b := range m.Content {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, " ")
}

// resultsOf 找出请求历史里最后一条含 tool_result 的 user 消息。
func lastToolResults(t *testing.T, req llm.ChatRequest) []llm.ContentBlock {
	t.Helper()
	for i := len(req.Messages) - 1; i >= 0; i-- {
		var out []llm.ContentBlock
		for _, b := range req.Messages[i].Content {
			if b.Type == "tool_result" {
				out = append(out, b)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	t.Fatal("请求里没有 tool_result")
	return nil
}

func resultText(b llm.ContentBlock) string {
	s, _ := b.Content.(string)
	return s
}

// ============================================================================
// 用例
// ============================================================================

func TestRunSimpleReply(t *testing.T) {
	p := &scriptedProvider{steps: []scriptedStep{stopReply("这是答案", 1000, 50)}}
	a, _ := newAgent(t, p, registryWith(), nil)

	res, err := a.Run(context.Background(), "打个招呼")
	if err != nil {
		t.Fatal(err)
	}
	if res.Reply != "这是答案" || res.StopReason != "completed" || res.Iterations != 1 {
		t.Errorf("结果不对: %+v", res)
	}
	if res.Usage != (llm.Usage{InputTokens: 1000, OutputTokens: 50}) {
		t.Errorf("用量未累计: %+v", res.Usage)
	}
	if res.CostUSD <= 0 {
		t.Errorf("应能估算成本: %v", res.CostUSD)
	}
	if len(p.requests) != 1 {
		t.Fatalf("调用次数 = %d", len(p.requests))
	}
	if lastUserText(t, p.requests[0]) != "打个招呼" {
		t.Errorf("首轮应把用户任务作为最后一条 user 消息")
	}
}

func TestRunToolRoundTrip(t *testing.T) {
	ft := &fakeTool{name: "read_file", result: "package main\n"}
	p := &scriptedProvider{steps: []scriptedStep{
		toolStep("tu_1", "read_file", map[string]any{"file_path": "main.go"}, "我先看看"),
		stopReply("读完了", 200, 10),
	}}
	a, _ := newAgent(t, p, registryWith(ft), nil)

	res, err := a.Run(context.Background(), "读 main.go")
	if err != nil {
		t.Fatal(err)
	}
	if res.Reply != "读完了" {
		t.Errorf("最终回复应只取收尾那一轮的文本，实际 %q（中间步骤的文本不能算答案）", res.Reply)
	}
	if ft.calls != 1 {
		t.Errorf("工具调用次数 = %d", ft.calls)
	}
	if ft.lastIn["file_path"] != "main.go" {
		t.Errorf("参数没透传: %v", ft.lastIn)
	}

	// 第二次请求必须带上配对的 tool_result
	if len(p.requests) != 2 {
		t.Fatalf("调用次数 = %d", len(p.requests))
	}
	results := lastToolResults(t, p.requests[1])
	if len(results) != 1 || results[0].ToolUseID != "tu_1" {
		t.Fatalf("tool_result 配对失败: %+v", results)
	}
	if resultText(results[0]) != "package main\n" {
		t.Errorf("工具输出没有回传给模型: %q", resultText(results[0]))
	}
	if results[0].IsError != nil && *results[0].IsError {
		t.Error("成功结果不该标记 is_error")
	}
}

// 未知工具名要作为错误结果回传，让模型自己纠正，而不是把整轮任务打死。
func TestRunUnknownTool(t *testing.T) {
	p := &scriptedProvider{steps: []scriptedStep{
		toolStep("tu_x", "does_not_exist", nil, ""),
		stopReply("换个办法", 0, 0),
	}}
	a, _ := newAgent(t, p, registryWith(), nil)

	res, err := a.Run(context.Background(), "任务")
	if err != nil || res.Reply != "换个办法" {
		t.Fatalf("未知工具不该终止任务: res=%+v err=%v", res, err)
	}
	r := lastToolResults(t, p.requests[1])[0]
	if r.IsError == nil || !*r.IsError {
		t.Error("未知工具应回传 is_error")
	}
	if !strings.Contains(resultText(r), "read_file") && !strings.Contains(resultText(r), "可用工具") {
		t.Errorf("应把可用工具列表告诉模型: %q", resultText(r))
	}
}

func TestRunToolErrorKeepsGoing(t *testing.T) {
	ft := &fakeTool{name: "boom", err: errors.New("磁盘满了")}
	p := &scriptedProvider{steps: []scriptedStep{
		toolStep("tu_1", "boom", nil, ""),
		stopReply("绕过去了", 0, 0),
	}}
	a, _ := newAgent(t, p, registryWith(ft), nil)

	if _, err := a.Run(context.Background(), "任务"); err != nil {
		t.Fatalf("工具失败应作为信息回传给模型，而不是终止: %v", err)
	}
	r := lastToolResults(t, p.requests[1])[0]
	if r.IsError == nil || !*r.IsError {
		t.Error("失败结果必须标记 is_error，否则模型会以为成功")
	}
	if !strings.Contains(resultText(r), "磁盘满了") {
		t.Errorf("错误原因要原样传回: %q", resultText(r))
	}
}

// 参数只到一半的工具调用绝不能被执行——否则等于执行一个猜测。
func TestRunSkipsPartialToolUse(t *testing.T) {
	ft := &fakeTool{name: "write_file", mutating: true, result: "ok"}
	partial := llm.ContentBlock{
		Type: "tool_use", ID: "tu_p", Name: "write_file",
		Input: map[string]any{"content": "半截"}, Partial: true,
	}
	p := &scriptedProvider{steps: []scriptedStep{
		{resp: &llm.ChatResponse{
			FinishReason: llm.FinishToolCalls,
			Content:      []llm.ContentBlock{partial},
		}},
		stopReply("重新来过", 0, 0),
	}}
	a, _ := newAgent(t, p, registryWith(ft), nil)

	if _, err := a.Run(context.Background(), "写文件"); err != nil {
		t.Fatal(err)
	}
	if ft.calls != 0 {
		t.Errorf("截断的调用被执行了 %d 次", ft.calls)
	}
	// 历史里这条 tool_use 已被 RepairHistory 丢弃，不该留下孤儿 tool_result
	for _, req := range p.requests {
		for _, m := range req.Messages {
			for _, b := range m.Content {
				if b.ToolUseID == "tu_p" {
					t.Errorf("被丢弃的调用留下了孤儿结果: %+v", b)
				}
			}
		}
	}
}

// 撞 max_tokens：普通文本要请求续写；被截断的工具调用要作废并提示拆分。
func TestRunLengthTruncation(t *testing.T) {
	t.Run("文本续写", func(t *testing.T) {
		p := &scriptedProvider{steps: []scriptedStep{
			lengthStep([]llm.ContentBlock{{Type: "text", Text: "写了一半"}}),
			stopReply("写完", 0, 0),
		}}
		a, _ := newAgent(t, p, registryWith(), nil)
		res, err := a.Run(context.Background(), "写一篇长文")
		if err != nil || res.Iterations != 2 {
			t.Fatalf("res=%+v err=%v", res, err)
		}
		// 续写提示之后模型才回答，中间文本不该混进最终答案
		if res.Reply != "写完" {
			t.Errorf("Reply = %q", res.Reply)
		}
		if !strings.Contains(lastUserText(t, p.requests[1]), "max_tokens") {
			t.Errorf("应告知模型上一轮被截断: %q", lastUserText(t, p.requests[1]))
		}
	})

	t.Run("截断的调用要作废", func(t *testing.T) {
		ft := &fakeTool{name: "write_file"}
		p := &scriptedProvider{steps: []scriptedStep{
			lengthStep([]llm.ContentBlock{{
				Type: "tool_use", ID: "t", Name: "write_file",
				Input: map[string]any{"content": "半"}, Partial: true,
			}}),
			stopReply("改小了再写", 0, 0),
		}}
		a, _ := newAgent(t, p, registryWith(ft), nil)
		if _, err := a.Run(context.Background(), "写个大文件"); err != nil {
			t.Fatal(err)
		}
		if ft.calls != 0 {
			t.Error("截断的 write_file 被执行了")
		}
		msg := lastUserText(t, p.requests[1])
		if !strings.Contains(msg, "作废") {
			t.Errorf("应说明被截断的调用已作废: %q", msg)
		}
		// 作废的调用不能留在历史里等结果
		for _, m := range p.requests[1].Messages {
			for _, b := range m.Content {
				if b.ID == "t" || b.ToolUseID == "t" {
					t.Errorf("被作废的 tool_use 仍留在历史中: %+v", b)
				}
			}
		}
	})
}

func TestRunMaxIterations(t *testing.T) {
	var steps []scriptedStep
	for i := 0; i < 10; i++ {
		steps = append(steps, toolStep(fmt.Sprintf("t%d", i), "noop", nil, ""))
	}
	p := &scriptedProvider{steps: steps}
	noop := &fakeTool{name: "noop", result: "ok"}
	a, cfg := newAgent(t, p, registryWith(noop), func(c *Config) { c.MaxIterations = 3 })

	res, err := a.Run(context.Background(), "停不下来的任务")
	if !errors.Is(err, ErrMaxIterations) {
		t.Fatalf("应返回 ErrMaxIterations，实际: %v", err)
	}
	if res.StopReason != "max_iterations" || res.Iterations != 3 {
		t.Errorf("res=%+v", res)
	}
	if noop.calls != 3 {
		t.Errorf("工具执行 %d 次，应停在 %d 轮", noop.calls, cfg.MaxIterations)
	}
}

// Ctrl+C 之后历史必须还能接着用：本轮里没跑完的 tool_use 由 RepairHistory
// 补上带 is_error 的占位结果，用户直接说"继续"就行，不必 /clear 重来。
func TestRunInterruptKeepsHistoryUsable(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 模型一次要跑两个工具，第一个跑完时用户按下 Ctrl+C
	ft := &fakeTool{name: "slow", result: "跑完了的那个结果"}
	ft.onCall = func(n int) {
		if n == 1 {
			cancel()
		}
	}

	p := &scriptedProvider{steps: []scriptedStep{
		{resp: &llm.ChatResponse{
			FinishReason: llm.FinishToolCalls,
			Content: []llm.ContentBlock{
				{Type: "tool_use", ID: "tu_1", Name: "slow"},
				{Type: "tool_use", ID: "tu_2", Name: "slow"},
			},
		}},
	}}
	a, _ := newAgent(t, p, registryWith(ft), nil)

	res, err := a.Run(ctx, "跑个长任务")
	if !errors.Is(err, ErrInterrupted) {
		t.Fatalf("应返回 ErrInterrupted，实际: %v", err)
	}
	if res.StopReason != "interrupted" {
		t.Errorf("StopReason = %s", res.StopReason)
	}
	if ft.calls != 1 {
		t.Errorf("中断后仍继续执行了后面的工具: calls=%d", ft.calls)
	}
	if a.HistoryLen() == 0 {
		t.Fatal("中断后历史被丢弃了")
	}

	// 恢复：把残缺历史交给新的 agent，应能直接续上
	p2 := &scriptedProvider{steps: []scriptedStep{stopReply("接上了", 5, 5)}}
	a2 := New(a.Config(), p2, registryWith(ft))
	a2.LoadHistory(a.History())
	if _, err := a2.Run(context.Background(), "继续"); err != nil {
		t.Fatalf("中断历史应可直接续用: %v", err)
	}

	r := findResult(t, p2.requests[0], "tu_2")
	if r.IsError == nil || !*r.IsError {
		t.Error("没跑完的调用应补上 is_error 占位结果")
	}
	if !strings.Contains(resultText(r), "中断") && !strings.Contains(resultText(r), "截断") {
		t.Errorf("占位结果应说明原因: %q", resultText(r))
	}
	// 已经跑完的那个不能被重复补结果
	if twice := countResults(p2.requests[0], "tu_1"); twice != 1 {
		t.Errorf("tu_1 的结果出现 %d 次，被复制了", twice)
	}
}

func findResult(t *testing.T, req llm.ChatRequest, id string) llm.ContentBlock {
	t.Helper()
	for _, m := range req.Messages {
		for _, b := range m.Content {
			if b.Type == "tool_result" && b.ToolUseID == id {
				return b
			}
		}
	}
	t.Fatalf("请求里找不到 %s 的结果", id)
	return llm.ContentBlock{}
}

func countResults(req llm.ChatRequest, id string) int {
	n := 0
	for _, m := range req.Messages {
		for _, b := range m.Content {
			if b.Type == "tool_result" && b.ToolUseID == id {
				n++
			}
		}
	}
	return n
}

// 服务端判定超限时，压缩历史后重试一次，而不是把错误抛给用户。
func TestRunOverContextRetriesAfterCompaction(t *testing.T) {
	over := &llm.Error{Status: 400, OverContext: true, Message: "prompt is too long"}
	p := &scriptedProvider{steps: []scriptedStep{
		{err: over},
		stopReply("压缩后成功了", 10, 10),
	}}
	a, _ := newAgent(t, p, registryWith(), nil)

	res, err := a.Run(context.Background(), "很长的任务")
	if err != nil {
		t.Fatalf("超限应自愈: %v", err)
	}
	if res.Reply != "压缩后成功了" {
		t.Errorf("Reply = %q", res.Reply)
	}
	if len(p.requests) != 2 {
		t.Fatalf("调用次数 = %d，期望重试一次", len(p.requests))
	}
	if len(p.requests[1].Messages) > len(p.requests[0].Messages) {
		t.Error("重试前应带着压缩过的历史")
	}
}

// 不可重试的错误要原样上抛，由 main 决定怎么呈现。
func TestRunProviderErrorPropagates(t *testing.T) {
	p := &scriptedProvider{steps: []scriptedStep{{err: errors.New("连接被重置")}}}
	a, _ := newAgent(t, p, registryWith(), nil)

	res, err := a.Run(context.Background(), "任务")
	if err == nil || res.StopReason != "error" {
		t.Fatalf("应报错退出: res=%+v err=%v", res, err)
	}
	if !strings.Contains(err.Error(), "连接被重置") {
		t.Errorf("底层原因要透出: %v", err)
	}
}

// 历史超出预算时自动折叠，且第一条任务不能被折掉。
func TestRunCompactsBeforeBudgetBreached(t *testing.T) {
	long := strings.Repeat("很长的工具输出行\n", 400)
	ft := &fakeTool{name: "reader", result: long}

	var steps []scriptedStep
	for i := 0; i < 4; i++ {
		steps = append(steps, toolStep(fmt.Sprintf("t%d", i), "reader", nil, ""))
	}
	steps = append(steps, stopReply("搞定", 0, 0))

	p := &scriptedProvider{steps: steps}
	a, _ := newAgent(t, p, registryWith(ft), func(c *Config) {
		c.ContextWindow = 8000
		c.MaxTokens = 4000
		c.MaxIterations = 5
	})

	res, err := a.Run(context.Background(), "读很多次")
	if err != nil {
		t.Fatal(err)
	}
	if res.Compactions == 0 {
		t.Error("历史早已超预算，却没有触发折叠")
	}
	if len(p.requests) < 2 {
		t.Fatalf("调用次数 = %d", len(p.requests))
	}
	final := p.requests[len(p.requests)-1]
	if !strings.Contains(msgTextOf(final.Messages[0]), "读很多次") {
		t.Error("折叠把用户最初的诉求弄丢了")
	}
}

func msgTextOf(m llm.Message) string {
	var parts []string
	for _, b := range m.Content {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, " ")
}

// 折叠到第二级时，被丢的往返要先摘要再丢弃，摘要要真的写回 history。
// 直接驱动 a.compact 而不是跑整个 Run：摘要调用和对话调用共用同一个
// scripted 队列，跑整循环会让第几步是摘要变得不可控。
func TestCompactSummarizesIntoHistory(t *testing.T) {
	long := strings.Repeat("很长的工具输出行\n", 400)

	// 一步：provider 被摘要调用时返回这段文本
	p := &scriptedProvider{steps: []scriptedStep{
		stopReply("入口在 main.go，已读完 a/b/c 三个文件。", 0, 0),
	}}
	a, _ := newAgent(t, p, registryWith(&fakeTool{name: "reader", result: long}), func(c *Config) {
		c.ContextWindow = 5000
		c.MaxTokens = 2000
	})

	history := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("反复读大文件")}},
	}
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("t%d", i)
		history = append(history,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{toolUse(id, "reader")}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult(id, long)}},
		)
	}
	a.history = history

	if freed := a.compact(context.Background(), 3000); freed <= 0 {
		t.Fatal("compact 没有省下 token")
	}

	if len(p.requests) != 1 || len(p.requests[0].Tools) != 0 {
		t.Fatalf("应恰好发起一次摘要请求（不带工具），实际 %d 次", len(p.requests))
	}
	if !strings.Contains(p.requests[0].System, "压缩") {
		t.Error("摘要请求没带压缩器的系统提示词")
	}

	first := msgTextOf(a.history[0])
	if !strings.Contains(first, "反复读大文件") {
		t.Errorf("第一条任务丢了: %q", first)
	}
	if !strings.Contains(first, "早前对话摘要") || !strings.Contains(first, "入口在 main.go") {
		t.Errorf("摘要没写回 history 第一条:\n%s", first)
	}
	assertValid(t, a.history)
}

// 过程输出走 config.Writer：非交互 JSON 模式靠这个把噪声挡在 stdout 之外。
func TestProcessOutputGoesToWriter(t *testing.T) {
	var buf bytes.Buffer
	ft := &fakeTool{name: "reader", result: "文件内容"}
	p := &scriptedProvider{steps: []scriptedStep{
		toolStep("tu_1", "reader", nil, "我读一下"),
		stopReply("好了", 0, 0),
	}}
	a, _ := newAgent(t, p, registryWith(ft), func(c *Config) {
		c.Quiet = false
		c.Writer = &buf
	})

	if _, err := a.Run(context.Background(), "读文件"); err != nil {
		t.Fatal(err)
	}
	s := buf.String()
	for _, want := range []string{"我读一下", "🔧 reader", "文件内容"} {
		if !strings.Contains(s, want) {
			t.Errorf("过程输出缺少 %q；完整输出:\n%s", want, s)
		}
	}

	quiet := &scriptedProvider{steps: []scriptedStep{stopReply("答案", 0, 0)}}
	var qbuf bytes.Buffer
	a2, _ := newAgent(t, quiet, registryWith(), func(c *Config) {
		c.Quiet = true
		c.Writer = &qbuf
	})
	if _, err := a2.Run(context.Background(), "任务"); err != nil {
		t.Fatal(err)
	}
	if qbuf.Len() != 0 {
		t.Errorf("Quiet 模式不该有过程输出: %q", qbuf.String())
	}
}

func TestHistoryBudget(t *testing.T) {
	c := DefaultConfig(".", "deepseek-chat")
	if got := c.HistoryBudget(); got != 128000-32000-contextReserve {
		t.Errorf("预算 = %d", got)
	}
	// 窗口再小也要留出可用空间，否则每轮都在折叠
	c.ContextWindow = 8000
	c.MaxTokens = 8000
	if got := c.HistoryBudget(); got != 4000 {
		t.Errorf("极小窗口下预算应有下限，实际 %d", got)
	}
	if ContextWindowFor("claude-opus-4-8") != 200000 {
		t.Error("claude 模型窗口识别错误")
	}
	if ContextWindowFor("unknown-model") != 128000 {
		t.Error("未知模型应有保守默认窗口")
	}
}

func TestClearHistoryResetsUsage(t *testing.T) {
	p := &scriptedProvider{steps: []scriptedStep{stopReply("答案", 100, 20)}}
	a, _ := newAgent(t, p, registryWith(), nil)
	if _, err := a.Run(context.Background(), "任务"); err != nil {
		t.Fatal(err)
	}
	if a.Usage().InputTokens != 100 {
		t.Fatal("前置条件：用量应已累计")
	}
	a.ClearHistory()
	if a.HistoryLen() != 0 || a.Usage() != (llm.Usage{}) {
		t.Errorf("清空后 history=%d usage=%+v", a.HistoryLen(), a.Usage())
	}
}
