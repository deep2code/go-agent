package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"go-agent/llm"
)

// 折叠历史的第二种力度是"丢最早的往返"。丢之前先摘一次要，
// 否则模型只是知道自己丢了一段对话，而不是丢掉了哪些结论。
func TestCompactHistorySummarizesDroppedRounds(t *testing.T) {
	const bigPayload = "文件内容行\n"
	long := strings.Repeat(bigPayload, 60)

	messages := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("原始任务：分析这个仓库")}},
	}
	for i := 0; i < 6; i++ {
		id := string(rune('a'+i)) + "_1"
		messages = append(messages,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{toolUse(id, "read_file")}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult(id, long)}},
		)
	}
	messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{text("结论是……")}})

	before := EstimateTokens(messages)

	var called int
	var gotDropped []llm.Message
	summary := Summarizer(func(ctx context.Context, dropped []llm.Message) (string, error) {
		called++
		gotDropped = dropped
		return "已读完 6 个文件，入口在 main.go。", nil
	})

	compact, freed := CompactHistory(context.Background(), messages, before/5, summary)
	if called == 0 {
		t.Fatal("进入了整轮丢弃却没有调用摘要")
	}
	if freed <= 0 {
		t.Error("折叠后没有省下 token")
	}
	if len(gotDropped) == 0 {
		t.Error("摘要没收到被丢弃的消息")
	}
	for _, m := range gotDropped {
		if msgText(m) == "原始任务：分析这个仓库" {
			t.Error("第一条任务被当成可丢弃的内容送进摘要")
		}
	}

	if !strings.Contains(msgText(compact[0]), "原始任务") {
		t.Error("第一条用户任务必须原样保留")
	}
	// 摘要追加在第一条消息里：索引 0 已是 user，另起一条就破坏交替
	if !strings.Contains(msgText(compact[0]), "已读完 6 个文件") {
		t.Errorf("摘要没贴到任务那条消息上:\n%s", msgText(compact[0]))
	}
	if !strings.Contains(msgText(compact[0]), "非原文") {
		t.Error("摘要应标明自己是压缩内容，否则模型会当原文引用")
	}
	if msgText(compact[len(compact)-1]) != "结论是……" {
		t.Error("最近的 assistant 内容不应被动")
	}
	assertValid(t, compact)
}

// 摘要是一次额外的模型调用：可能失败、可能被取消。失败不能让任务整体崩掉，
// 退回"直接丢弃"至少还能继续跑。
func TestCompactHistorySummarizerFailureFallsBackToDrop(t *testing.T) {
	long := strings.Repeat("文件内容行\n", 60)
	messages := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("原始任务")}},
	}
	for i := 0; i < 6; i++ {
		id := string(rune('a'+i)) + "_1"
		messages = append(messages,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{toolUse(id, "read_file")}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult(id, long)}},
		)
	}
	messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{text("结论")}})
	before := EstimateTokens(messages)

	failing := Summarizer(func(ctx context.Context, dropped []llm.Message) (string, error) {
		return "", errors.New("摘要调用失败")
	})
	compact, freed := CompactHistory(context.Background(), messages, before/5, failing)
	if freed <= 0 {
		t.Error("摘要失败也应完成丢弃，省下 token")
	}
	for _, m := range compact {
		if strings.Contains(msgText(m), "早前对话摘要") {
			t.Error("摘要失败却插入了摘要占位")
		}
	}
	assertValid(t, compact)

	// 空摘要同样不插入
	empty := Summarizer(func(ctx context.Context, dropped []llm.Message) (string, error) {
		return "   ", nil
	})
	compact2, _ := CompactHistory(context.Background(), messages, before/5, empty)
	for _, m := range compact2 {
		if strings.Contains(msgText(m), "早前对话摘要") {
			t.Error("空摘要不应插入")
		}
	}
}

// 摘要只该在真的要做第二轮折叠时发生——多一次模型调用是有成本的。
func TestCompactHistorySkipsSummarizerWhenFoldingSuffices(t *testing.T) {
	long := strings.Repeat("文件内容行\n", 60)
	messages := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("原始任务")}},
	}
	for i := 0; i < 6; i++ {
		id := string(rune('a'+i)) + "_1"
		messages = append(messages,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{toolUse(id, "read_file")}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult(id, long)}},
		)
	}
	messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{text("结论")}})
	before := EstimateTokens(messages)

	called := false
	sum := Summarizer(func(ctx context.Context, dropped []llm.Message) (string, error) {
		called = true
		return "不该被调用", nil
	})

	// 预算刚好需要折叠工具输出、但还不至于丢整轮
	compact, freed := CompactHistory(context.Background(), messages, before*3/4, sum)
	if freed <= 0 {
		t.Fatal("这个预算下应该发生第一级折叠")
	}
	if called {
		t.Error("第一级折叠就够了，却额外调用了摘要模型")
	}
	assertValid(t, compact)
}

func TestRenderTranscript(t *testing.T) {
	long := strings.Repeat("长", 1500)
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			text("帮我改 retry"),
			toolResult("t1", long),
		}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: "thinking", Thinking: "内部推理不该进摘要"},
			toolUse("t1", "read_file"),
			text("我看完了"),
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("  ")}},
	}

	out := renderTranscript(msgs)
	for _, want := range []string{"用户: 帮我改 retry", "助手: 我看完了", "工具结果:", "调用工具 read_file"} {
		if !strings.Contains(out, want) {
			t.Errorf("转写缺少 %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "内部推理不该进摘要") {
		t.Error("thinking 内容不该进摘要素材")
	}
	if strings.Contains(out, "长长") && !strings.Contains(out, "…") {
		t.Error("超长块没被截断")
	}
	if len([]rune(out)) > 1400 {
		t.Errorf("截断没生效，转写长度 %d", len([]rune(out)))
	}

	// 全是空块时返回空串，调用方据此跳过一次无谓的模型调用
	if got := renderTranscript([]llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text("   ")}}}); got != "" {
		t.Errorf("空历史应返回空串，实际 %q", got)
	}
}

// 摘要请求本身不该变成被折叠内容的副本：把 write_file 的整份 content 原样摊进去，
// 刚省下来的预算又被摘要自己吃回去了。
func TestRenderTranscriptClipsToolArgs(t *testing.T) {
	body := strings.Repeat("package main\n", 400) // 5600 字节
	msgs := []llm.Message{{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
		Type: "tool_use", ID: "t1", Name: "write_file",
		Input: map[string]any{"file_path": "a.go", "content": body},
	}}}}

	out := renderTranscript(msgs)
	if strings.Contains(out, body) {
		t.Error("工具参数的整份正文进了摘要请求")
	}
	if len([]rune(out)) > 600 {
		t.Errorf("摘要转写被参数撑到 %d 字", len([]rune(out)))
	}
	// 短参数必须原样留着：file_path 是摘要里唯一能点名"改的是哪个文件"的线索
	if !strings.Contains(out, "a.go") {
		t.Errorf("短参数被一起截掉了:\n%s", out)
	}
}

// 同一份参数每次转写必须一模一样：转写文本进的是模型调用，抖动没有意义还更难核对。
func TestClipArgsIsStable(t *testing.T) {
	input := map[string]any{"zeta": "1", "alpha": "2", "mid": "3"}
	first := clipArgs(input, 20)
	for i := 0; i < 20; i++ {
		if clipArgs(input, 20) != first {
			t.Fatal("同一份参数的转写每次不同")
		}
	}
	if !strings.HasPrefix(first, "{alpha=") {
		t.Errorf("键未按字母序排列: %q", first)
	}
	if got := clipArgs(nil, 20); got != "{}" {
		t.Errorf("空参数应转成 {}，实际 %q", got)
	}
}

func TestClip(t *testing.T) {
	if got := clip("中文混abc", 2); got != "中文…" {
		t.Errorf("clip = %q", got)
	}
	if got := clip("短", 10); got != "短" {
		t.Errorf("未超长不应改动: %q", got)
	}
}

// 子 agent 的 token 要能累加回主对话，否则 /cost 会低估真实花费。
func TestAddUsage(t *testing.T) {
	a := New(quietConfig(t.TempDir()), &scriptedProvider{}, registryWith())
	a.AddUsage(llm.Usage{InputTokens: 100, OutputTokens: 20})
	a.AddUsage(llm.Usage{InputTokens: 5, OutputTokens: 3})
	u := a.Usage()
	if u.InputTokens != 105 || u.OutputTokens != 23 {
		t.Errorf("累加后用量 = %+v，期望 {105, 23}", u)
	}
}
