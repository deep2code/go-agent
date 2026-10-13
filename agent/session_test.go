package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"go-agent/llm"
)

// 会话落盘的唯一目的：明天开机还能接着昨天的对话。
// 所以成对的 tool_use / tool_result 与 is_error 标记必须原样回来。

func sampleSession(id string) Session {
	isErr := true
	return Session{
		ID:         id,
		WorkingDir: "/tmp/proj",
		Provider:   "deepseek",
		Model:      "deepseek-chat",
		Turns:      2,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: "text", Text: "把 main.go 里的日志换成 zap"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Type: "text", Text: "我先读一下"},
				{Type: "tool_use", ID: "tu_1", Name: "read_file", Input: map[string]any{"file_path": "main.go"}},
			}},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				{Type: "tool_result", ToolUseID: "tu_1", Content: "1 | package main"},
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Type: "tool_use", ID: "tu_2", Name: "edit_file", Input: map[string]any{"file_path": "main.go"}},
			}},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				{Type: "tool_result", ToolUseID: "tu_2", Content: "权限被拒绝", IsError: &isErr},
			}},
		},
		Usage: llm.Usage{InputTokens: 1234, OutputTokens: 56},
	}
}

func TestSessionSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s := sampleSession("20260101-100000-0001")

	if err := Save(dir, s); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(SessionDir(dir), s.ID+".json")); err != nil {
		t.Fatalf("会话没落盘: %v", err)
	}

	got, err := Load(dir, s.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != s.Model || got.WorkingDir != s.WorkingDir || got.Turns != s.Turns {
		t.Errorf("元信息丢失: %+v", got)
	}
	if got.Usage != s.Usage {
		t.Errorf("用量丢失: %+v", got.Usage)
	}
	if len(got.Messages) != len(s.Messages) {
		t.Fatalf("消息条数 %d → %d", len(s.Messages), len(got.Messages))
	}

	// 工具调用的参数与结果必须成对回来
	asst := got.Messages[1].Content
	if asst[1].Type != "tool_use" || asst[1].ID != "tu_1" || asst[1].Name != "read_file" {
		t.Errorf("tool_use 还原失败: %+v", asst[1])
	}
	if asst[1].Input["file_path"] != "main.go" {
		t.Errorf("tool_use 参数丢失: %+v", asst[1].Input)
	}
	res := got.Messages[2].Content[0]
	if res.Type != "tool_result" || res.ToolUseID != "tu_1" || res.Content != "1 | package main" {
		t.Errorf("tool_result 还原失败: %+v", res)
	}
	errRes := got.Messages[4].Content[0]
	if errRes.IsError == nil || !*errRes.IsError {
		t.Error("is_error 标记在落盘后丢了，模型会把失败当成功")
	}

	// 恢复出来的历史应能直接发请求
	a := New(quietConfig(dir), &scriptedProvider{steps: []scriptedStep{stopReply("好", 0, 0)}}, registryWith())
	a.LoadHistory(got.Messages)
	if _, err := a.Run(t.Context(), "继续"); err != nil {
		t.Fatalf("恢复的会话无法直接使用: %v", err)
	}
}

func TestSessionSummaryAndList(t *testing.T) {
	dir := t.TempDir()

	first := sampleSession("s-first")
	first.Messages[0].Content[0].Text = strings.Repeat("很长的任务描述", 40)
	if err := Save(dir, first); err != nil {
		t.Fatal(err)
	}
	time.Sleep(time.Millisecond)

	second := sampleSession("s-second")
	second.Messages[0].Content[0].Text = "第二个任务"
	if err := Save(dir, second); err != nil {
		t.Fatal(err)
	}

	list, err := List(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("会话数 = %d", len(list))
	}
	if list[0].ID != "s-second" {
		t.Errorf("列表应按时间倒序，第一条是 %q", list[0].ID)
	}
	if !strings.HasSuffix(list[0].Preview, "第二个任务") {
		t.Errorf("摘要应取首条用户消息: %q", list[0].Preview)
	}
	if len([]rune(list[1].Preview)) > 61 { // 60 字 + 省略号
		t.Errorf("摘要应截断到 60 字，实际 %d", len([]rune(list[1].Preview)))
	}

	// Load 的空 id / "last" 都取最近一次
	for _, id := range []string{"", "last"} {
		got, err := Load(dir, id)
		if err != nil {
			t.Fatalf("id=%q: %v", id, err)
		}
		if got.ID != "s-second" {
			t.Errorf("id=%q 取到了 %q", id, got.ID)
		}
	}
}

func TestSessionErrors(t *testing.T) {
	dir := t.TempDir()

	if _, err := List(dir); err != nil {
		t.Errorf("目录还不存在时 List 应返回空而不是错: %v", err)
	}
	if _, err := Load(dir, "last"); err == nil {
		t.Error("没有会话时恢复应报错")
	}
	if _, err := Load(dir, "nope"); err == nil || !strings.Contains(err.Error(), "/sessions") {
		t.Errorf("找不到会话时应提示用 /sessions 查看: %v", err)
	}
	if err := Save(dir, Session{}); err == nil {
		t.Error("无 ID 的会话应拒绝落盘")
	}

	// 手改坏掉的文件不能让整个程序崩
	sessions := SessionDir(dir)
	if err := os.MkdirAll(sessions, 0o755); err != nil {
		t.Fatal(err)
	}
	broken := filepath.Join(sessions, "broken.json")
	if err := os.WriteFile(broken, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir, "broken"); err == nil || !strings.Contains(err.Error(), "损坏") {
		t.Errorf("坏文件应报损坏: %v", err)
	}
	if _, err := List(dir); err != nil {
		t.Errorf("列表里有坏文件时 List 不该失败: %v", err)
	}
}

func TestNewSessionIDSortsChronologically(t *testing.T) {
	a := NewSessionID()
	time.Sleep(2 * time.Millisecond)
	b := NewSessionID()
	if a >= b {
		t.Errorf("ID 应可直接按字典序排出先后: %q vs %q", a, b)
	}
	if a == b {
		t.Error("同一秒内两次 ID 相同，第二个会话会覆盖第一个")
	}
	if len(a) < 8 || strings.ContainsAny(a, " /") {
		t.Errorf("ID 不适合做文件名: %q", a)
	}

	// 同一秒内连续开两个会话（连续 /new）不能撞文件
	dir := t.TempDir()
	first := sampleSession(a)
	second := sampleSession(b)
	for _, s := range []Session{first, second} {
		if err := Save(dir, s); err != nil {
			t.Fatal(err)
		}
	}
	list, err := List(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("两个会话只剩 %d 个: %+v", len(list), list)
	}
}

// Partial 是内部标记，不参与序列化；恢复时由 RepairHistory 兜底。
func TestSessionDropsPartialToolUseOnLoad(t *testing.T) {
	dir := t.TempDir()
	s := sampleSession("partial-case")
	s.Messages = append(s.Messages, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		{Type: "tool_use", ID: "tu_half", Name: "write_file", Input: map[string]any{"content": "半截"}, Partial: true},
	}})
	if err := Save(dir, s); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(filepath.Join(SessionDir(dir), "partial-case.json"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "Partial") || strings.Contains(string(data), `"partial"`) {
		t.Error("内部标记不该写进会话文件")
	}

	got, err := Load(dir, "partial-case")
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("落盘内容不是合法 JSON: %v", err)
	}

	a := New(quietConfig(dir), &scriptedProvider{steps: []scriptedStep{stopReply("好", 0, 0)}}, registryWith())
	a.LoadHistory(got.Messages)
	for _, m := range a.History() {
		for _, b := range m.Content {
			if b.ID == "tu_half" || b.ToolUseID == "tu_half" {
				t.Errorf("半截调用被恢复进了历史: %+v", b)
			}
		}
	}
}
