package tools

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

type recordingRunner struct {
	got      string
	gotWrite bool
	err      error
	reply    string
	calls    int
}

func (r *recordingRunner) RunSubAgent(ctx context.Context, task string, writable bool) (string, error) {
	r.calls++
	r.got = task
	r.gotWrite = writable
	return r.reply, r.err
}

// autoTask 造一个不问话的 task 工具：测派发逻辑时用，避免每个用例都要搭审批门。
func autoTask(r SubAgentRunner) *TaskTool {
	return NewTaskTool(r, NewGate(ModeAuto), io.Discard)
}

func TestTaskToolContract(t *testing.T) {
	noop := &recordingRunner{reply: "结论"}
	tt := autoTask(noop)

	if tt.Name() != "task" {
		t.Errorf("Name = %q", tt.Name())
	}
	if tt.Definition().Name != tt.Name() {
		t.Error("Definition().Name 与 Name() 不一致，Registry 会 panic")
	}
	if tt.Mutating() {
		t.Error("派发这一步本身不改动文件系统，Mutating 必须为 false")
	}
	// 缺省必须站在只读那一侧：布尔参数缺省若是有害的，模型漏传一次就等於偷偷派了个能改文件的子 agent
	def := tt.Definition()
	if strings.Contains(def.Description, "缺省可写") || strings.Contains(def.Description, "默认 writable=true") {
		t.Error("task 的描述应把只读当成缺省面孔")
	}
	writable, ok := def.InputSchema.Properties["writable"]
	if !ok {
		t.Fatal("描述里承诺了可写子任务，schema 却没有 writable 参数")
	}
	if writable.Type != "boolean" {
		t.Errorf("writable 应为 boolean，实际 %q", writable.Type)
	}
	for _, req := range def.InputSchema.Required {
		if req == "writable" {
			t.Error("writable 不该是必填：绝大多数子任务是只读探查")
		}
	}
}

func TestTaskToolExecute(t *testing.T) {
	t.Run("透传结论", func(t *testing.T) {
		r := &recordingRunner{reply: "入口在 main.go"}
		tt := autoTask(r)
		out, err := tt.Execute(context.Background(), map[string]any{"task": "找入口"})
		if err != nil {
			t.Fatal(err)
		}
		if out != "入口在 main.go" {
			t.Errorf("out = %q", out)
		}
		if r.got != "找入口" {
			t.Errorf("子任务没原样传下去: %q", r.got)
		}
		if r.gotWrite {
			t.Error("不传 writable 时子任务必须按只读派发")
		}
	})

	t.Run("缺参数报错", func(t *testing.T) {
		tt := autoTask(&recordingRunner{})
		for _, in := range []map[string]any{nil, {}, {"task": "   "}} {
			if _, err := tt.Execute(context.Background(), in); err == nil {
				t.Errorf("task=%v 应报错", in)
			}
		}
	})

	t.Run("runner 失败被包装", func(t *testing.T) {
		tt := autoTask(&recordingRunner{err: errors.New("上下文超限")})
		_, err := tt.Execute(context.Background(), map[string]any{"task": "x"})
		if err == nil || !strings.Contains(err.Error(), "上下文超限") {
			t.Errorf("底层原因要透出: %v", err)
		}
	})

	t.Run("空结论给占位", func(t *testing.T) {
		tt := autoTask(&recordingRunner{reply: "  "})
		out, err := tt.Execute(context.Background(), map[string]any{"task": "x"})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out, "没有返回结论") {
			t.Errorf("空结论应给明确占位，实际 %q", out)
		}
	})

	t.Run("未装配 runner 直接报错", func(t *testing.T) {
		tt := autoTask(nil)
		if _, err := tt.Execute(context.Background(), map[string]any{"task": "x"}); err == nil {
			t.Error("runner 为 nil 应在调用时报错")
		}
	})

	t.Run("取消的上下文传到子任务", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		// 子 agent 的 Run 会自行检查 ctx；TaskTool 只负责把 ctx 透传下去
		called := false
		tt := autoTask(ctxRunnerFunc(func(c context.Context) error {
			called = true
			return c.Err()
		}))
		if _, err := tt.Execute(ctx, map[string]any{"task": "x"}); err == nil {
			t.Error("已取消的上下文应让子任务失败")
		}
		if !called {
			t.Error("ctx 未透传给子任务")
		}
	})
}

// 可写子任务是这道门唯一"批准一件事却没做完一件事"的调用：
// 所以它必须当场问人，而且拒绝时一个字节的文件都不该被碰过。
func TestWritableSubAgentNeedsApproval(t *testing.T) {
	t.Run("ask 模式先问再派", func(t *testing.T) {
		var asked []string
		gate := NewGate(ModeAsk)
		gate.Ask = func(title, detail string) bool {
			asked = append(asked, title+"|"+detail)
			return true
		}
		r := &recordingRunner{reply: "改好了"}
		out, err := NewTaskTool(r, gate, io.Discard).Execute(context.Background(),
			map[string]any{"task": "把三个包的日志前缀统一", "writable": true})
		if err != nil {
			t.Fatal(err)
		}
		if out != "改好了" {
			t.Errorf("out = %q", out)
		}
		if len(asked) != 1 || !strings.HasPrefix(asked[0], "task|") {
			t.Fatalf("应恰好问一次 task，实际 %v", asked)
		}
		if !strings.Contains(asked[0], "把三个包的日志前缀统一") {
			t.Errorf("确认详情要带上子任务内容，实际 %q", asked[0])
		}
		if !r.gotWrite {
			t.Error("用户批准后应当以可写面孔派发")
		}
	})

	t.Run("用户拒绝则子 agent 一次都不跑", func(t *testing.T) {
		gate := NewGate(ModeAsk)
		gate.Ask = func(string, string) bool { return false }
		r := &recordingRunner{reply: "不该跑到"}
		_, err := NewTaskTool(r, gate, io.Discard).Execute(context.Background(),
			map[string]any{"task": "改点东西", "writable": true})
		if !errors.Is(err, ErrDenied) {
			t.Errorf("应返回拒绝，实际 %v", err)
		}
		if r.calls != 0 {
			t.Errorf("被拒绝后仍跑了 %d 次子 agent：那笔 token 已经花了，文件却没人批准过", r.calls)
		}
	})

	t.Run("只读模式当场拒绝", func(t *testing.T) {
		// -p 不带 --yes 时缺省只读。若这里放行，子 agent 里的 write_file 才在深处撞门，
		// 整轮模型调用就白烧了。
		r := &recordingRunner{}
		_, err := NewTaskTool(r, NewGate(ModeReadOnly), io.Discard).Execute(context.Background(),
			map[string]any{"task": "改点东西", "writable": true})
		if !errors.Is(err, ErrDenied) {
			t.Errorf("只读模式应拒绝可写子任务，实际 %v", err)
		}
		if r.calls != 0 {
			t.Errorf("只读模式仍派出了子 agent（%d 次）", r.calls)
		}
	})

	t.Run("只读子任务不打扰用户", func(t *testing.T) {
		// 这条是反面的门：可写要问，但绝大多数探查若每次都问，人会把 --yes 当成默认，
		// 那时真正的危险改动也被一起预放行了。
		asked := 0
		gate := NewGate(ModeAsk)
		gate.Ask = func(string, string) bool { asked++; return true }
		r := &recordingRunner{reply: "结论"}
		if _, err := NewTaskTool(r, gate, io.Discard).Execute(context.Background(),
			map[string]any{"task": "找找看"}); err != nil {
			t.Fatal(err)
		}
		if asked != 0 {
			t.Error("只读子任务不该弹确认")
		}
		if r.gotWrite {
			t.Error("只读子任务被派发成可写")
		}
	})

	t.Run("没装配审批门时可写子任务一律拒绝", func(t *testing.T) {
		r := &recordingRunner{}
		_, err := NewTaskTool(r, nil, io.Discard).Execute(context.Background(),
			map[string]any{"task": "改点东西", "writable": true})
		if !errors.Is(err, ErrDenied) {
			t.Errorf("缺门即无渠道确认，必须拒绝，实际 %v", err)
		}
		if r.calls != 0 {
			t.Error("没有确认渠道却跑起了可写子 agent")
		}
	})

	t.Run("没装配审批门时只读子任务照常", func(t *testing.T) {
		r := &recordingRunner{reply: "结论"}
		out, err := NewTaskTool(r, nil, io.Discard).Execute(context.Background(),
			map[string]any{"task": "找找看"})
		if err != nil || out != "结论" {
			t.Errorf("只读派发不应依赖审批门: out=%q err=%v", out, err)
		}
	})
}

// ctxRunnerFunc 把函数适配成 SubAgentRunner。
type ctxRunnerFunc func(ctx context.Context) error

func (f ctxRunnerFunc) RunSubAgent(ctx context.Context, _ string, _ bool) (string, error) {
	return "", f(ctx)
}

func TestFirstLine(t *testing.T) {
	if got := firstLine("第一行\n第二行"); got != "第一行" {
		t.Errorf("firstLine = %q", got)
	}
	long := strings.Repeat("汉", 100)
	got := firstLine(long)
	if len([]rune(got)) != 81 { // 80 + 省略号
		t.Errorf("应按 80 字截断，实际长度 %d: %q", len([]rune(got)), got)
	}
	if !strings.HasSuffix(got, "…") {
		t.Error("超长首行应带省略号")
	}
}
