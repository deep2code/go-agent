package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"go-agent/llm"
	"go-agent/tools"
)

// 并发只给显式声明安全的只读工具，且结果顺序不能变。
// 这里直接调 executeToolCalls，不走 a.Run——端到端断言会被 provider 步数带偏（见 AGENTS.md 测试风格）。

// recorder 跨工具共享，用来证明"这些调用真的同时在场"。
// 每个工具自己数峰值恒等于 1，那是自证，说明不了任何事。
type recorder struct {
	mu      sync.Mutex
	active  int
	peak    int
	history []string // "start:<name>" / "end:<name>"
}

func (r *recorder) start(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.active++
	if r.active > r.peak {
		r.peak = r.active
	}
	r.history = append(r.history, "start:"+name)
}

func (r *recorder) end(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.active--
	r.history = append(r.history, "end:"+name)
}

func (r *recorder) snapshot() (int, []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.peak, append([]string(nil), r.history...)
}

// trackedTool 是一个可配置的假工具：可选并行声明、可控耗时、共享并发记录。
type trackedTool struct {
	name     string
	mutating bool
	parallel bool
	delay    time.Duration
	rec      *recorder
	onCall   func(n int)

	mu    sync.Mutex
	calls int
}

func (t *trackedTool) Name() string       { return t.name }
func (t *trackedTool) Mutating() bool     { return t.mutating }
func (t *trackedTool) ParallelSafe() bool { return t.parallel }
func (t *trackedTool) Definition() llm.ToolDef {
	return llm.ToolDef{Name: t.name, Description: "测试用工具"}
}

func (t *trackedTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	t.mu.Lock()
	t.calls++
	n := t.calls
	t.mu.Unlock()

	if t.onCall != nil {
		t.onCall(n)
	}
	t.rec.start(t.name)
	if t.delay > 0 {
		select {
		case <-time.After(t.delay):
		case <-ctx.Done():
		}
	}
	t.rec.end(t.name)
	return fmt.Sprintf("%s#%d", t.name, n), nil
}

func (t *trackedTool) called() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.calls
}

// toolUse / resultIDs 沿用 context_test.go 与 context.go 里已有的同名助手。

func asTools(ts ...*trackedTool) []tools.Tool {
	out := make([]tools.Tool, 0, len(ts))
	for _, t := range ts {
		out = append(out, t)
	}
	return out
}

// 三个各睡 60ms 的只读调用：串行下限是 180ms，并发应该接近 60ms。
func TestReadOnlyToolsRunInParallel(t *testing.T) {
	rec := &recorder{}
	var ts []*trackedTool
	var uses []llm.ContentBlock
	for i := 1; i <= 3; i++ {
		tool := &trackedTool{
			name: fmt.Sprintf("ro%d", i), parallel: true,
			delay: 60 * time.Millisecond, rec: rec,
		}
		ts = append(ts, tool)
		uses = append(uses, toolUse(fmt.Sprintf("t%d", i), tool.name))
	}

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(ts...)...), nil)
	start := time.Now()
	got, err := a.executeToolCalls(context.Background(), uses, io.Discard)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	if peak, _ := rec.snapshot(); peak < 2 {
		t.Errorf("并发峰值 = %d，三个只读调用没有同时在场", peak)
	}
	if elapsed > 150*time.Millisecond {
		t.Errorf("跑了 %v，仍然接近串行耗时（串行下限 180ms）", elapsed)
	}
	for _, tool := range ts {
		if n := tool.called(); n != 1 {
			t.Errorf("%s 被调用 %d 次", tool.name, n)
		}
	}
	if got := strings.Join(resultIDs(got), ","); got != "t1,t2,t3" {
		t.Errorf("结果顺序 = %s，期望 t1,t2,t3", got)
	}
}

// 并发组里先发起的调用可能后完成：回填顺序必须还是模型给出 tool_use 的顺序。
func TestParallelGroupResultsKeepModelOrder(t *testing.T) {
	rec := &recorder{}
	slow := &trackedTool{name: "slow", parallel: true, delay: 90 * time.Millisecond, rec: rec}
	fast := &trackedTool{name: "fast", parallel: true, delay: 10 * time.Millisecond, rec: rec}

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(slow, fast)...), nil)
	uses := []llm.ContentBlock{toolUse("a", "slow"), toolUse("b", "fast")}

	got, err := a.executeToolCalls(context.Background(), uses, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(resultIDs(got), ",") != "a,b" {
		t.Errorf("结果顺序 = %s，期望 a,b（fast 先完成也要排在后面）", resultIDs(got))
	}
	first, _ := got[0].Content.(string)
	second, _ := got[1].Content.(string)
	if first != "slow#1" || second != "fast#1" {
		t.Errorf("结果配错了：%q / %q", first, second)
	}
}

// 会改动的工具绝不能并发：两次改动同时跑，diff 基线与先读后改的门会互相踩。
func TestMutatingToolsStaySerial(t *testing.T) {
	rec := &recorder{}
	first := &trackedTool{name: "w1", mutating: true, parallel: true, delay: 30 * time.Millisecond, rec: rec}
	second := &trackedTool{name: "w2", mutating: true, parallel: true, delay: 30 * time.Millisecond, rec: rec}

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(first, second)...), nil)
	uses := []llm.ContentBlock{toolUse("1", "w1"), toolUse("2", "w2")}
	got, err := a.executeToolCalls(context.Background(), uses, io.Discard)
	if err != nil {
		t.Fatal(err)
	}

	peak, history := rec.snapshot()
	if peak != 1 {
		t.Errorf("改动类工具并发峰值 = %d，必须为 1；时间线 %v", peak, history)
	}
	want := "start:w1,end:w1,start:w2,end:w2"
	if strings.Join(history, ",") != want {
		t.Errorf("时间线 = %v，期望严格首尾相接：%s", history, want)
	}
	if strings.Join(resultIDs(got), ",") != "1,2" {
		t.Errorf("结果顺序 = %s", resultIDs(got))
	}
}

// 没声明并行的只读工具、以及"声明了并行但会改动"的工具，都必须落到串行。
func TestParallelizableRequiresDeclaration(t *testing.T) {
	rec := &recorder{}
	liar := &trackedTool{name: "liar", mutating: true, parallel: true, delay: 20 * time.Millisecond, rec: rec}
	quiet := &trackedTool{name: "quiet", delay: 20 * time.Millisecond, rec: rec}

	if tools.Parallelizable(liar) {
		t.Error("会改动的工具被当成可并发：两次改动会互相踩 diff 基线")
	}
	if tools.Parallelizable(quiet) {
		t.Error("没实现 ParallelSafe 的只读工具被当成可并发：缺省应当从严")
	}

	// task 这类"只读但会跑一整轮子循环"的工具在产品代码里没有并行声明，
	// 由 tools 包的 TestParallelizableCoversRealTools 直接对着真工具校验。

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(liar, quiet)...), nil)
	uses := []llm.ContentBlock{toolUse("1", "liar"), toolUse("2", "quiet")}
	if _, err := a.executeToolCalls(context.Background(), uses, io.Discard); err != nil {
		t.Fatal(err)
	}
	if peak, history := rec.snapshot(); peak != 1 {
		t.Errorf("并发峰值 = %d，时间线 %v", peak, history)
	}
}

// 只读段和改动段混在一起时：只读那段并发，改动那段仍是串行边界。
func TestMixedRunSplitsAtMutatingBoundary(t *testing.T) {
	rec := &recorder{}
	r1 := &trackedTool{name: "r1", parallel: true, delay: 40 * time.Millisecond, rec: rec}
	r2 := &trackedTool{name: "r2", parallel: true, delay: 40 * time.Millisecond, rec: rec}
	w := &trackedTool{name: "w", mutating: true, parallel: true, delay: 5 * time.Millisecond, rec: rec}

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(r1, r2, w)...), nil)
	uses := []llm.ContentBlock{toolUse("1", "r1"), toolUse("2", "r2"), toolUse("3", "w")}
	got, err := a.executeToolCalls(context.Background(), uses, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(resultIDs(got), ",") != "1,2,3" {
		t.Errorf("结果顺序 = %s", resultIDs(got))
	}
	peak, history := rec.snapshot()
	if peak != 2 {
		t.Errorf("并发峰值 = %d，期望 2（两个只读同时、改动单独）；时间线 %v", peak, history)
	}
	// w 必须独占：它前后不能有别的调用同时在场
	for i, h := range history {
		if h != "start:w" {
			continue
		}
		if i+1 >= len(history) || history[i+1] != "end:w" {
			t.Errorf("改动类调用 %v 与别的调用重叠：%v", h, history)
		}
	}
}

// 上下文已经取消：一个都不能执行，返回 ErrInterrupted，空格子交给 RepairHistory 补占位。
func TestCancelledContextExecutesNothing(t *testing.T) {
	rec := &recorder{}
	r := &trackedTool{name: "r", parallel: true, rec: rec}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(r)...), nil)
	uses := []llm.ContentBlock{toolUse("1", "r"), toolUse("2", "r")}
	got, err := a.executeToolCalls(ctx, uses, io.Discard)
	if err == nil || !errorsIsInterrupted(err) {
		t.Fatalf("应返回 ErrInterrupted，实际 = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("取消后仍有 %d 个结果：%v", len(got), resultIDs(got))
	}
	if n := r.called(); n != 0 {
		t.Errorf("取消后工具仍被执行 %d 次", n)
	}
}

// 中断发生在两组之间：第二组一个都不该启动。
func TestInterruptBetweenGroupsSkipsLaterGroup(t *testing.T) {
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	w := &trackedTool{name: "w", mutating: true, delay: 5 * time.Millisecond, rec: rec}
	w.onCall = func(int) { cancel() } // 改动类调用跑完就把上下文关掉
	r := &trackedTool{name: "r", parallel: true, delay: 5 * time.Millisecond, rec: rec}

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(asTools(w, r)...), nil)
	uses := []llm.ContentBlock{toolUse("1", "w"), toolUse("2", "r")}
	got, err := a.executeToolCalls(ctx, uses, io.Discard)
	if err == nil || !errorsIsInterrupted(err) {
		t.Fatalf("应返回 ErrInterrupted，实际 = %v", err)
	}
	if strings.Join(resultIDs(got), ",") != "1" {
		t.Errorf("已执行的结果 = %v，只应有第一组：%v", resultIDs(got), got)
	}
	if n := r.called(); n != 0 {
		t.Errorf("中断后第二组还是被执行了 %d 次", n)
	}
}

func errorsIsInterrupted(err error) bool { return errors.Is(err, ErrInterrupted) }
