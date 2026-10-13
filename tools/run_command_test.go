package tools

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// run_command 是破坏面最大的工具：它直接走 bash -c。
// 这里验的是"门在命令之前"、"超时会杀掉整组进程"、"命令失败不当作工具错误"。

func runIn(t *testing.T, dir string, g *Gate) *RunCommandTool {
	t.Helper()
	return NewRunCommandTool(dir, g, 0, io.Discard)
}

func TestRunCommandSuccess(t *testing.T) {
	dir := t.TempDir()
	out, err := runIn(t, dir, autoGate()).Execute(context.Background(),
		map[string]any{"command": "echo hello-from-tool"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "hello-from-tool") || !strings.Contains(out, "✅ 执行成功") {
		t.Errorf("输出不对: %q", out)
	}
}

// 命令失败对模型是有效信息，不该以 tool_error 形式吞掉退出码和日志。
func TestRunCommandNonZeroExitIsInformational(t *testing.T) {
	out, err := runIn(t, t.TempDir(), autoGate()).Execute(context.Background(),
		map[string]any{"command": "echo 编译失败 >&2; exit 3"})
	if err != nil {
		t.Fatalf("命令失败不该成为工具错误: %v", err)
	}
	if !strings.Contains(out, "退出码: 3") {
		t.Errorf("应报告退出码: %q", out)
	}
	if !strings.Contains(out, "[stderr] 编译失败") && !strings.Contains(out, "编译失败") {
		t.Errorf("stderr 内容也要回传给模型: %q", out)
	}
}

func TestRunCommandRunsInWorkingDir(t *testing.T) {
	dir := t.TempDir()
	out, err := runIn(t, dir, autoGate()).Execute(context.Background(),
		map[string]any{"command": "pwd"})
	if err != nil {
		t.Fatal(err)
	}
	// macOS 的临时目录是软链接，pwd 打印的是解析后的真实路径
	real, _ := filepath.EvalSymlinks(dir)
	if !strings.Contains(strings.TrimSpace(out), filepath.Base(real)) {
		t.Errorf("命令没有在工作目录里执行: %q", out)
	}
}

func TestRunCommandTimeoutKills(t *testing.T) {
	dir := t.TempDir()
	start := time.Now()
	out, err := runIn(t, dir, autoGate()).Execute(context.Background(),
		map[string]any{"command": "sleep 30", "timeout_seconds": "1"})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "超时") {
		t.Errorf("应报告超时: %q", out)
	}
	if elapsed > 5*time.Second {
		t.Errorf("超时后没有终止进程，耗时 %v", elapsed)
	}
}

// 超时必须杀掉整个进程组：只杀 bash 本身的话，它生下的子进程会继续写文件、
// 继续占端口，模型看到的"已终止"是假的。
func TestRunCommandTimeoutKillsProcessGroup(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "marker")

	// 后台子进程 3 秒后落一个文件；命令本身 5 秒，超时设 1 秒
	_, err := runIn(t, dir, autoGate()).Execute(context.Background(), map[string]any{
		"command":         "(sleep 3; touch marker) & sleep 5",
		"timeout_seconds": "1",
	})
	if err != nil {
		t.Fatal(err)
	}

	time.Sleep(3500 * time.Millisecond)
	if _, statErr := os.Stat(marker); statErr == nil {
		t.Error("超时只杀了 bash，后台子进程还活着并把文件写了出来")
	} else if !os.IsNotExist(statErr) {
		t.Fatal(statErr)
	}
}

// 审批门必须在 exec 之前：被拒绝的命令一个字都不该执行。
func TestRunCommandGateRunsBeforeExec(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "side-effect")

	ro := &Gate{Mode: ModeReadOnly, Out: io.Discard}
	_, err := runIn(t, dir, ro).Execute(context.Background(),
		map[string]any{"command": "touch side-effect"})
	if !errors.Is(err, ErrDenied) {
		t.Fatalf("只读模式应拒绝: %v", err)
	}
	if _, statErr := os.Stat(marker); statErr == nil {
		t.Error("被拒绝的命令仍然执行了")
	}

	// 禁止名单同理：即使用户在 Ask 里手滑点了同意
	var asked string
	ask := &Gate{Mode: ModeAsk, Out: io.Discard, Ask: func(_, detail string) bool {
		asked = detail
		return true
	}}
	if _, err := runIn(t, dir, ask).Execute(context.Background(),
		map[string]any{"command": "rm -rf / --no-preserve-root"}); err == nil {
		t.Fatal("禁止名单的命令应被拒")
	}
	if asked != "" {
		t.Error("禁止名单不该走到用户面前求确认")
	}
}

func TestRunCommandAsksWithPreview(t *testing.T) {
	var title, detail string
	g := &Gate{Mode: ModeAsk, Out: io.Discard, Ask: func(ti, de string) bool {
		title, detail = ti, de
		return true
	}}
	if _, err := runIn(t, t.TempDir(), g).Execute(context.Background(),
		map[string]any{"command": "echo 需要授权"}); err != nil {
		t.Fatal(err)
	}
	if title != "run_command" || !strings.Contains(detail, "echo 需要授权") {
		t.Errorf("确认信息应包含原命令: title=%q detail=%q", title, detail)
	}
}

func TestRunCommandLongOutputTruncated(t *testing.T) {
	out, err := runIn(t, t.TempDir(), autoGate()).Execute(context.Background(),
		map[string]any{"command": "seq 1 3000"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "省略") {
		t.Errorf("超长输出应做头尾保留: %q", out[:200])
	}
	if len(out) > 9000 {
		t.Errorf("截断后仍达 %d 字节，会挤占上下文", len(out))
	}
	// 头部是第一条、尾部是最后一条，模型才判断得出命令跑到哪一步
	if !strings.Contains(out, "$ seq 1 3000") || !strings.Contains(out, "3000") {
		t.Errorf("头尾内容缺失: %q", out[:120])
	}
}

func TestRunCommandEchoesProgress(t *testing.T) {
	var buf strings.Builder
	tool := NewRunCommandTool(t.TempDir(), autoGate(), 0, &buf)
	if _, err := tool.Execute(context.Background(),
		map[string]any{"command": "printf 'first\\nsecond\\n'; echo 出错了 >&2"}); err != nil {
		t.Fatal(err)
	}
	echoed := buf.String()
	for _, want := range []string{"first", "second"} {
		if !strings.Contains(echoed, want) {
			t.Errorf("实时回显缺少 %q: %q", want, echoed)
		}
	}
	// 给人看的回显里要分得清哪一路输出，否则报错信息会被当成正常输出
	if !strings.Contains(echoed, "[stderr] │ 出错了") {
		t.Errorf("stderr 回显缺少标记: %q", echoed)
	}
}

func TestRunCommandMissingParam(t *testing.T) {
	if _, err := runIn(t, t.TempDir(), autoGate()).Execute(context.Background(), map[string]any{}); err == nil {
		t.Error("缺少 command 参数应报错")
	}
}

// 命令不能继承会话的终端输入：git 要口令、ssh 要确认时，继承 stdin 会把整个
// 会话挂住，而不是让命令自己失败。cmd.Stdin = nil 就是防这个。
func TestRunCommandDoesNotInheritStdin(t *testing.T) {
	start := time.Now()
	out, err := runIn(t, t.TempDir(), autoGate()).Execute(context.Background(),
		map[string]any{"command": "cat", "timeout_seconds": "5"})
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("cat 在等待输入，说明 stdin 被继承了：卡了 %v（输出 %q）", elapsed, out)
	}
}
