package tools

import (
	"context"
	"encoding/json"
	"io"
	"testing"
	"time"

	"go-agent/llm"
)

// 并发白名单是对着真实工具表校验的，不是只在测试里造两个假工具证明判定函数能用：
// 新增工具时最容易忘的正是"它到底能不能同时跑"。
func TestParallelizableCoversRealTools(t *testing.T) {
	dir := t.TempDir()
	gate := NewGate(ModeAuto)
	state := NewFileState()
	idx := NewRepoIndex()

	parallel := []Tool{
		NewReadFileTool(dir, state),
		NewSearchCodeTool(dir).WithIndex(idx),
		NewListDirectoryTool(dir),
		NewRepoOverviewTool(dir),
	}
	serial := []Tool{
		NewWriteFileTool(dir, gate, state).WithIndex(idx),
		NewEditFileTool(dir, gate, state).WithIndex(idx),
		NewMoveFileTool(dir, gate, state).WithIndex(idx),
		NewDeleteFileTool(dir, gate, state).WithIndex(idx),
		NewRunCommandTool(dir, gate, time.Second, io.Discard),
		NewPlanTool(io.Discard),
		NewTaskTool(nil, NewGate(ModeReadOnly), io.Discard),
	}

	for _, tool := range parallel {
		if !Parallelizable(tool) {
			t.Errorf("%s 应当可并发：只读、不占终端、不派子循环", tool.Name())
		}
	}
	for _, tool := range serial {
		if Parallelizable(tool) {
			t.Errorf("%s 被判为可并发。改动类要过审批门，不允许多人同时打字；"+
				"task 和 update_plan 虽然只读，但一个要跑整轮子循环、一个要往终端打印", tool.Name())
		}
	}
}

// 远端工具只有服务器自己声明只读时才能并发；缺省按会改动对待，那条规则在这里同样成立。
func TestParallelizableForRemoteTools(t *testing.T) {
	srv := &fakeMCPServer{name: "srv"}

	mutating := NewMCPTool(srv, NewGate(ModeAuto), time.Second,
		"mcp_srv_x", "x", false, "远端工具", json.RawMessage(`{"type":"object"}`))
	if Parallelizable(mutating) {
		t.Error("未声明只读的远端工具被判为可并发")
	}

	readonly := NewMCPTool(srv, NewGate(ModeAuto), time.Second,
		"mcp_srv_y", "y", true, "远端只读工具", json.RawMessage(`{"type":"object"}`))
	if !Parallelizable(readonly) {
		t.Error("声明只读的远端工具应当可以并发（客户端按 ID 分派响应，写入有锁）")
	}
}

// 没实现 ParallelSafe 接口的工具一律串行——缺省从严，不是缺省可用。
func TestParallelizableDefaultsToSerial(t *testing.T) {
	if Parallelizable(noDeclare{}) {
		t.Error("没声明并行的工具被当成可并发")
	}
	if !Parallelizable(declared{}) {
		t.Error("显式声明并行的只读工具应当可以并发")
	}
	// 声明了并行但会改动：并行判定必须否掉它，否则改动之间会互踩 diff 基线
	if Parallelizable(lying{}) {
		t.Error("会改动的工具即使声明并行也必须串行")
	}
}

// 三个桩子覆盖判定的三种情况：没实现接口、实现且只读、实现但会改动。
type noDeclare struct{}

func (noDeclare) Name() string            { return "no_declare" }
func (noDeclare) Mutating() bool          { return false }
func (noDeclare) Definition() llm.ToolDef { return llm.ToolDef{Name: "no_declare"} }
func (noDeclare) Execute(context.Context, map[string]any) (string, error) {
	return "", nil
}

type declared struct{}

func (declared) Name() string            { return "declared" }
func (declared) Mutating() bool          { return false }
func (declared) ParallelSafe() bool      { return true }
func (declared) Definition() llm.ToolDef { return llm.ToolDef{Name: "declared"} }
func (declared) Execute(context.Context, map[string]any) (string, error) {
	return "", nil
}

type lying struct{}

func (lying) Name() string            { return "lying" }
func (lying) Mutating() bool          { return true }
func (lying) ParallelSafe() bool      { return true }
func (lying) Definition() llm.ToolDef { return llm.ToolDef{Name: "lying"} }
func (lying) Execute(context.Context, map[string]any) (string, error) {
	return "", nil
}
