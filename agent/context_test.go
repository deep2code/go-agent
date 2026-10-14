package agent

import (
	"context"
	"strings"
	"testing"

	"go-agent/llm"
)

// 历史修复的判据只有一条：发给 API 的每条 tool_use 都有配对的 tool_result，
// 且 user / assistant 严格交替。这两点在代码里是隐式约定，测试把它们钉住。

func text(s string) llm.ContentBlock { return llm.ContentBlock{Type: "text", Text: s} }

func toolUse(id, name string) llm.ContentBlock {
	return llm.ContentBlock{Type: "tool_use", ID: id, Name: name, Input: map[string]any{"x": 1}}
}

func partialToolUse(id, name string) llm.ContentBlock {
	b := toolUse(id, name)
	b.Partial = true
	return b
}

func toolResult(id, body string) llm.ContentBlock {
	return llm.ContentBlock{Type: "tool_result", ToolUseID: id, Content: body}
}

func assertValid(t *testing.T, msgs []llm.Message) {
	t.Helper()
	if len(msgs) == 0 {
		t.Fatal("修复后历史为空")
	}
	if msgs[0].Role != llm.RoleUser {
		t.Errorf("首条必须是 user，实际 %s", msgs[0].Role)
	}
	var pending []string
	for i, m := range msgs {
		if i > 0 && m.Role == msgs[i-1].Role {
			t.Errorf("第 %d 条与上一条同为 %s，未交替", i, m.Role)
		}
		if m.Role == llm.RoleAssistant {
			if len(pending) > 0 {
				t.Errorf("上一条 assistant 的 tool_use %v 没有等到结果", pending)
			}
			pending = nil
			for _, b := range m.Content {
				if b.Type == "tool_use" {
					if b.Partial {
						t.Errorf("截断的 tool_use %s 不应留在历史里", b.ID)
					}
					pending = append(pending, b.ID)
				}
			}
			continue
		}
		// user：出现的 tool_result 必须命中当前 pending
		for _, b := range m.Content {
			if b.Type != "tool_result" {
				continue
			}
			if !contains(pending, b.ToolUseID) {
				t.Errorf("tool_result %s 没有对应的 tool_use", b.ToolUseID)
			}
		}
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				pending = remove(pending, b.ToolUseID)
			}
		}
	}
	if len(pending) > 0 {
		t.Errorf("结尾仍有未配对的 tool_use %v", pending)
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func remove(s []string, v string) []string {
	out := s[:0]
	for _, x := range s {
		if x != v {
			out = append(out, x)
		}
	}
	return out
}

// 任务被 Ctrl+C 打断后，历史里留着没有结果的 tool_use。
// 带着它再发一次请求会被服务端拒绝——这正是"只能 /clear"的根因。
func TestRepairHistoryAddsMissingResults(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("帮我修一下 bug")}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			text("我先看看"), toolUse("tu_1", "read_file"), toolUse("tu_2", "search_code"),
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("tu_1", "文件内容")}},
	}

	fixed := RepairHistory(history)
	assertValid(t, fixed)

	last := fixed[len(fixed)-1]
	if last.Role != llm.RoleUser {
		t.Fatalf("应以 user 的工具结果收尾，实际 %s", last.Role)
	}
	var matched bool
	for _, b := range last.Content {
		if b.ToolUseID == "tu_2" {
			matched = true
			if b.IsError == nil || !*b.IsError {
				t.Error("补上的占位结果应标记 is_error")
			}
			if !strings.Contains(toolResultTextOf(b), "重新发起") {
				t.Errorf("占位结果应告诉模型可以重试: %v", b.Content)
			}
		}
	}
	if !matched {
		t.Errorf("tu_2 没有补上占位结果: %+v", last.Content)
	}

	// 幂等：修好的历史再修一次不该变
	again := RepairHistory(fixed)
	assertValid(t, again)
	if len(again) != len(fixed) {
		t.Errorf("重复修复改变了历史长度 %d → %d", len(fixed), len(again))
	}
}

func TestRepairHistoryDropsPartialAndOrphans(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("任务")}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			text("被截断的调用"), partialToolUse("tu_half", "write_file"),
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			toolResult("tu_half", "不该存在的结果"), text("用户补充"),
		}},
	}

	fixed := RepairHistory(history)
	assertValid(t, fixed)
	for _, m := range fixed {
		for _, b := range m.Content {
			if b.ToolUseID == "tu_half" || b.ID == "tu_half" {
				t.Errorf("半条工具调用及其结果都应被丢弃，但仍留着 %+v", b)
			}
		}
	}
	if !strings.Contains(msgText(fixed[0]), "任务") || !strings.Contains(msgText(fixed[len(fixed)-1]), "用户补充") {
		t.Error("正常的文本内容被误删了")
	}
}

// 首条是 assistant 的历史（例如从会话文件恢复了半截）不能直接发出去。
func TestRepairHistoryLeadingAssistant(t *testing.T) {
	fixed := RepairHistory([]llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{text("我先看看")}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("好的")}},
	})
	assertValid(t, fixed)
	if msgText(fixed[0]) != "好的" {
		t.Errorf("应以用户消息开头，实际 %q", msgText(fixed[0]))
	}
}

// 同角色必须合并：中断后常出现连续两条 user。
func TestRepairHistoryMergesSameRole(t *testing.T) {
	fixed := RepairHistory([]llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("第一条")}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("第二条")}},
	})
	assertValid(t, fixed)
	if len(fixed) != 1 {
		t.Fatalf("连续 user 应合并为一条，实际 %d 条", len(fixed))
	}
}

func msgText(m llm.Message) string {
	var parts []string
	for _, b := range m.Content {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, " ")
}

func toolResultTextOf(b llm.ContentBlock) string {
	s, _ := b.Content.(string)
	return s
}

func TestEstimateTokensGrows(t *testing.T) {
	messages := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{text("短")}},
	}
	small := EstimateTokens(messages)

	messages[0].Content[0] = text(strings.Repeat("很长的一段中文内容", 200))
	large := EstimateTokens(messages)
	if large <= small {
		t.Errorf("内容变长后估算应当变大: %d → %d", small, large)
	}

	if EstimateTokens(nil) != 0 {
		t.Error("空历史应估算为 0 token")
	}

	// 中文按 ~1 token/字，ASCII 按 ~4 token/字符：同样长度中文应显著更贵
	zh := EstimateTokens([]llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text(strings.Repeat("中", 400))}}})
	en := EstimateTokens([]llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{text(strings.Repeat("a", 400))}}})
	if zh <= en {
		t.Errorf("中文估算偏低: zh=%d en=%d", zh, en)
	}
}

func TestEstimateTools(t *testing.T) {
	defs := []llm.ToolDef{{
		Name:        "read_file",
		Description: "读取文件",
		InputSchema: llm.InputSchema{
			Type:       "object",
			Properties: map[string]llm.Property{"file_path": {Type: "string", Description: "路径"}},
			Required:   []string{"file_path"},
		},
	}}
	if EstimateTools(defs) <= 0 {
		t.Error("工具定义应占用 token")
	}
	if EstimateTools(nil) != 0 {
		t.Error("空工具列表应为 0")
	}
}

// 折叠历史不能改变对话结构，也不能丢掉用户最初的诉求。
func TestCompactHistoryFoldsOldResults(t *testing.T) {
	const bigPayload = "文件内容行\n" // 重复成超过 200 字节的工具输出
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
	// 预算压到 1/5：第一轮折叠不够，必须进入整轮丢弃（summarize 为 nil，退回丢弃）
	compact, freed := CompactHistory(context.Background(), messages, before/5, nil)
	if freed <= 0 {
		t.Fatal("超出预算却没有省下任何 token")
	}
	if got := EstimateTokens(compact); got >= before {
		t.Errorf("折叠后 token 没有下降: %d → %d", before, got)
	}
	if len(compact) >= len(messages) {
		t.Errorf("第二轮应整轮丢弃，长度却没变: %d", len(compact))
	}

	if !strings.Contains(msgText(compact[0]), "原始任务") {
		t.Error("第一条用户任务必须原样保留")
	}
	if msgText(compact[len(compact)-1]) != "结论是……" {
		t.Error("最近的 assistant 内容不应被动")
	}

	// 折叠后的历史仍然要是合法请求体
	assertValid(t, compact)

	// 未超预算时不该动内容
	kept, freed := CompactHistory(context.Background(), messages, before*2, nil)
	if freed != 0 {
		t.Errorf("预算充足却进行了折叠: freed=%d", freed)
	}
	if EstimateTokens(kept) != before {
		t.Error("预算充足时历史内容应保持不变")
	}
}

func TestHasPartialToolUse(t *testing.T) {
	if HasPartialToolUse([]llm.ContentBlock{toolUse("a", "read_file")}) {
		t.Error("完整调用不应判为截断")
	}
	if !HasPartialToolUse([]llm.ContentBlock{text("x"), partialToolUse("a", "write_file")}) {
		t.Error("截断调用未被识别")
	}
}
