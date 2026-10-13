package tools

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"

	"go-agent/llm"
)

// ============================================================================
// PlanTool — 任务清单
// ============================================================================

// PlanTool 让模型把复杂任务拆成可跟踪的步骤并汇报进度。
// 清单只存在于内存和终端，不落盘、不影响文件系统。
type PlanTool struct {
	mu    sync.Mutex
	steps []planStep
	out   io.Writer // 为 nil 时不回显
}

type planStep struct {
	Title  string
	Status string // todo / doing / done
}

// NewPlanTool 创建 update_plan 工具。
func NewPlanTool(out io.Writer) *PlanTool {
	return &PlanTool{out: out}
}

// Name 返回工具名。
func (t *PlanTool) Name() string { return "update_plan" }

// Mutating 报告该工具不改动文件系统。
func (t *PlanTool) Mutating() bool { return false }

// Definition 返回 update_plan 工具的 JSON Schema。
func (t *PlanTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "update_plan",
		Description: "维护当前任务的步骤清单，用于 3 步以上的复杂任务。每次调用都提交完整清单（覆盖旧的）。每行一步，格式：'- [todo] 步骤描述'，状态可为 todo、doing、done。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"plan": {
					Type:        "string",
					Description: "完整清单文本，每行一步，例如：\n- [done] 读取现有实现\n- [doing] 补上重试\n- [todo] 写测试",
				},
			},
			Required: []string{"plan"},
		},
	}
}

// Execute 实现 update_plan 的执行逻辑。
func (t *PlanTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	raw, ok := input["plan"].(string)
	if !ok || strings.TrimSpace(raw) == "" {
		return "", fmt.Errorf("参数 plan 缺失或为空")
	}

	var steps []planStep
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		title, status := parsePlanLine(line)
		if title == "" {
			continue
		}
		steps = append(steps, planStep{Title: title, Status: status})
	}
	if len(steps) == 0 {
		return "", fmt.Errorf("没有解析出任何步骤，每行请使用 '- [todo] 描述' 格式")
	}

	t.mu.Lock()
	t.steps = steps
	rendered := t.renderLocked()
	t.mu.Unlock()

	if t.out != nil {
		fmt.Fprint(t.out, rendered)
	}
	return fmt.Sprintf("✅ 清单已更新（%d 步）\n%s", len(steps), rendered), nil
}

// renderLocked 生成清单文本，调用方需持有锁。
func (t *PlanTool) renderLocked() string {
	done := 0
	var sb strings.Builder
	sb.WriteString("  📋 任务清单\n")
	for _, s := range t.steps {
		mark := " "
		if s.Status == "done" {
			mark = "✓"
			done++
		} else if s.Status == "doing" {
			mark = "→"
		}
		sb.WriteString(fmt.Sprintf("     [%s] %s\n", mark, s.Title))
	}
	sb.WriteString(fmt.Sprintf("     进度 %d/%d\n", done, len(t.steps)))
	return sb.String()
}

// parsePlanLine 解析一行清单。
func parsePlanLine(line string) (title, status string) {
	line = strings.TrimPrefix(line, "-")
	line = strings.TrimSpace(line)

	if after, ok := strings.CutPrefix(line, "["); ok {
		if idx := strings.Index(after, "]"); idx >= 0 {
			tag := strings.ToLower(strings.TrimSpace(after[:idx]))
			title = strings.TrimSpace(after[idx+1:])
			switch tag {
			case "x", "done", "✓", "完成":
				status = "done"
			case ">", "doing", "wip", "进行中":
				status = "doing"
			default:
				status = "todo"
			}
			return title, status
		}
	}
	return line, "todo"
}

// Steps 返回当前清单的副本，供 /plan 命令展示。
func (t *PlanTool) Steps() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]string, 0, len(t.steps))
	for _, s := range t.steps {
		out = append(out, fmt.Sprintf("[%s] %s", s.Status, s.Title))
	}
	return out
}
