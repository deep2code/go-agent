package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go-agent/llm"
)

// ============================================================================
// Session — 对话落盘
// ============================================================================

// Session 是一次对话的可恢复快照。
type Session struct {
	ID         string        `json:"id"`
	WorkingDir string        `json:"working_dir"`
	Provider   string        `json:"provider"`
	Model      string        `json:"model"`
	Messages   []llm.Message `json:"messages"`
	Usage      llm.Usage     `json:"usage"`
	Turns      int           `json:"turns"`
	UpdatedAt  time.Time     `json:"updated_at"`
	Summary    string        `json:"summary"` // 首条用户消息的摘要，用于列表展示
}

// SessionDir 返回会话存放目录：<workingDir>/.go-agent/sessions
func SessionDir(workingDir string) string {
	return filepath.Join(workingDir, ".go-agent", "sessions")
}

// NewSessionID 生成一个按时间排序友好的会话 ID。
// 带毫秒：同一秒内连开两个会话（比如连续 /new）时，只用秒会撞成同一个文件，
// 前一个会话就被静默覆盖掉了。
func NewSessionID() string {
	now := time.Now()
	return fmt.Sprintf("%s-%03d-%04d", now.Format("20060102-150405"), now.Nanosecond()/1e6, os.Getpid()%10000)
}

// withoutPartials 丢掉参数被截断的工具调用。
// Partial 是内部标记、不进 JSON，所以落盘时必须显式剔除，
// 否则恢复出来的是一个看着完整、实际内容残缺的调用。
func withoutPartials(messages []llm.Message) []llm.Message {
	var out []llm.Message
	for _, m := range messages {
		var blocks []llm.ContentBlock
		for _, b := range m.Content {
			if b.Partial {
				continue
			}
			blocks = append(blocks, b)
		}
		if len(blocks) == 0 {
			continue
		}
		out = append(out, llm.Message{Role: m.Role, Content: blocks})
	}
	return out
}

// Save 把会话写成 JSON。目录不存在时自动创建。
func Save(workingDir string, s Session) error {
	if s.ID == "" {
		return fmt.Errorf("会话缺少 ID")
	}
	dir := SessionDir(workingDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("创建会话目录失败: %w", err)
	}
	s.UpdatedAt = time.Now()
	s.Messages = withoutPartials(s.Messages)
	if s.Summary == "" {
		s.Summary = summarize(s.Messages)
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化会话失败: %w", err)
	}
	path := filepath.Join(dir, s.ID+".json")
	return os.WriteFile(path, data, 0o644)
}

// Load 读取会话。id 为空或 "last" 时取最近一次。
func Load(workingDir, id string) (Session, error) {
	dir := SessionDir(workingDir)
	if id == "" || id == "last" {
		list, err := List(workingDir)
		if err != nil {
			return Session{}, err
		}
		if len(list) == 0 {
			return Session{}, fmt.Errorf("没有可恢复的会话")
		}
		id = list[0].ID
	}

	path := filepath.Join(dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Session{}, fmt.Errorf("找不到会话 %s（用 /sessions 查看可用会话）", id)
		}
		return Session{}, err
	}

	var s Session
	if err := json.Unmarshal(data, &s); err != nil {
		return Session{}, fmt.Errorf("会话文件损坏: %w", err)
	}
	return s, nil
}

// Summary 是会话列表里的一行。
type Summary struct {
	ID        string
	Model     string
	Turns     int
	UpdatedAt time.Time
	Preview   string
}

// List 返回按时间倒序排列的会话摘要。
func List(workingDir string) ([]Summary, error) {
	dir := SessionDir(workingDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var out []Summary
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var s Session
		if json.Unmarshal(data, &s) != nil {
			continue
		}
		out = append(out, Summary{
			ID:        s.ID,
			Model:     s.Model,
			Turns:     s.Turns,
			UpdatedAt: s.UpdatedAt,
			Preview:   s.Summary,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].UpdatedAt.After(out[j].UpdatedAt) })
	return out, nil
}

// summarize 取第一条用户消息的前 60 个字符作为摘要。
func summarize(messages []llm.Message) string {
	for _, m := range messages {
		if m.Role != llm.RoleUser {
			continue
		}
		for _, b := range m.Content {
			if b.Type == "text" && strings.TrimSpace(b.Text) != "" {
				r := []rune(b.Text)
				if len(r) > 60 {
					return string(r[:60]) + "…"
				}
				return string(r)
			}
		}
	}
	return ""
}
