package tools

import (
	"context"
	"fmt"
	"io"
	"strings"

	"go-agent/llm"
)

// ============================================================================
// TaskTool — 把聚焦的子任务交给一个独立上下文的子 agent
// ============================================================================

// SubAgentRunner 执行一个子任务并返回结论。
//
// 定义在 tools 层、由上层装配：tools 不能反向依赖 agent，
// 所以这里只约定"给我一个任务、还我一段结论"，具体怎么起子 agent 由调用方决定。
type SubAgentRunner interface {
	// RunSubAgent 在全新的上下文里执行 task，只回传最终结论。
	// 子 agent 的中间工具往返不应出现在返回值里——那正是隔离上下文的意义。
	//
	// writable 为 true 时，子 agent 额外拿到写文件与精确编辑：
	// 它能改文件，但每一次改动照样单独过审批门。
	RunSubAgent(ctx context.Context, task string, writable bool) (string, error)
}

// TaskTool 让主 agent 派出一个子 agent 处理聚焦的子任务。
//
// 意义在于上下文隔离：子 agent 读十个文件、跑一堆搜索，
// 这些过程留在它自己的历史里，主上下文只收到一段结论。
type TaskTool struct {
	runner SubAgentRunner
	gate   *Gate // 派发可写子任务时用它提前征询；为 nil 时可写子任务一律拒绝
	out    io.Writer
}

// NewTaskTool 创建 task 工具。runner 为空会让调用直接报错，
// 而不是等到模型真的调用它时才在运行时炸出来。
func NewTaskTool(runner SubAgentRunner, gate *Gate, out io.Writer) *TaskTool {
	return &TaskTool{runner: runner, gate: gate, out: out}
}

// Name 返回工具名。
func (t *TaskTool) Name() string { return "task" }

// Mutating 报告"派发"这一步本身不改动文件系统。
//
// 可写子任务的真正改动发生在子 agent 内部，每一次写入单独过门；
// 这里为 false 不代表派出去无害，而是不该由这一个布尔值替整棵改动树放行——
// 放行与否由 Execute 里的 writable 分支决定。
func (t *TaskTool) Mutating() bool { return false }

// Definition 返回 task 工具的 JSON Schema。
func (t *TaskTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name: "task",
		Description: "派一个独立上下文的子智能体去做一件聚焦的子任务，把它得出的结论还给你。" +
			"子任务的完整过程不会占用你的上下文，你只收到结论。" +
			"适合『在这个仓库里找出 X 在哪／梳理某模块的调用链』这类需要翻很多文件的问题。" +
			"缺省只读（读文件、搜索、列目录、仓库骨架）。只有子任务确实要改文件时才传 writable=true：" +
			"那会额外给它写文件与精确编辑，仍然没有执行命令、删除、移动，也不能再派孙 agent，" +
			"并且每一次写入都要用户逐个确认。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"task": {
					Type:        "string",
					Description: "交给子智能体的完整任务描述。要具体、自包含：子智能体看不到你当前的对话历史。",
				},
				"writable": {
					Type: "boolean",
					Description: "是否允许子智能体改文件，缺省 false。传 true 会先向用户确认一次这次派发，" +
						"之后每个文件的写入仍各自确认。只读探查不要传。",
				},
			},
			Required: []string{"task"},
		},
	}
}

// Execute 运行子 agent 并返回其结论。
func (t *TaskTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	task, _ := input["task"].(string)
	if strings.TrimSpace(task) == "" {
		return "", fmt.Errorf("参数 task 缺失或为空")
	}
	if t.runner == nil {
		return "", fmt.Errorf("子智能体未装配")
	}

	writable, _ := input["writable"].(bool)
	if writable {
		// 提前问一次"要不要让它开口写"，而不是等子 agent 自己去问：
		// 只读模式在这里当场拒绝，省掉一整轮已经花出去的模型调用。
		if t.gate == nil {
			return "", fmt.Errorf("%w: 可写子任务需要审批门，但当前未装配", ErrDenied)
		}
		if err := t.gate.Approve("task", "派一个可写子智能体："+firstLine(task)); err != nil {
			return "", err
		}
	}

	if t.out != nil {
		label := "子智能体"
		if writable {
			label = "可写子智能体"
		}
		fmt.Fprintf(t.out, "  🤝 交给%s：%s\n", label, firstLine(task))
	}

	reply, err := t.runner.RunSubAgent(ctx, task, writable)
	if err != nil {
		return "", fmt.Errorf("子智能体失败: %w", err)
	}
	if strings.TrimSpace(reply) == "" {
		return "（子智能体没有返回结论）", nil
	}
	return reply, nil
}

// firstLine 取任务首行，用于过程回显。
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	r := []rune(s)
	if len(r) > 80 {
		return string(r[:80]) + "…"
	}
	return s
}
