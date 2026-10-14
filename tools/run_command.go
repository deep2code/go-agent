package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go-agent/llm"
)

// ============================================================================
// RunCommandTool — 在审批门下执行 shell 命令
// ============================================================================

// RunCommandTool 执行 shell 命令，带超时、输出上限和实时进度。
type RunCommandTool struct {
	workingDir string
	gate       *Gate
	timeout    time.Duration
	out        io.Writer // 实时输出回显的目标，可为 nil
}

// NewRunCommandTool 创建 run_command 工具。timeout<=0 时使用默认值。
func NewRunCommandTool(workingDir string, gate *Gate, timeout time.Duration, out io.Writer) *RunCommandTool {
	if timeout <= 0 {
		timeout = defaultCommandTimeout
	}
	return &RunCommandTool{workingDir: workingDir, gate: gate, timeout: timeout, out: out}
}

// defaultCommandTimeout 给构建和测试留够时间，同时不至于让挂死的命令拖住整个会话。
const defaultCommandTimeout = 2 * time.Minute

// Name 返回工具名。
func (t *RunCommandTool) Name() string { return "run_command" }

// Mutating 报告该工具可能改动任何内容。
func (t *RunCommandTool) Mutating() bool { return true }

// Definition 返回 run_command 工具的 JSON Schema。
func (t *RunCommandTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "run_command",
		Description: fmt.Sprintf("执行 shell 命令并返回输出（bash -c），默认 %v 超时，超过 8000 字节的输出会被截断。禁止破坏性命令；需要较长时间的构建或测试请用 timeout_seconds 放宽。", t.timeout),
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"command": {
					Type:        "string",
					Description: "要执行的 shell 命令",
				},
				"timeout_seconds": {
					Type:        "string",
					Description: fmt.Sprintf("超时秒数，默认 %d，最大 900", int(t.timeout.Seconds())),
				},
			},
			Required: []string{"command"},
		},
	}
}

// Execute 实现 run_command 的执行逻辑。
func (t *RunCommandTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	command, ok := input["command"].(string)
	if !ok || command == "" {
		return "", fmt.Errorf("参数 command 缺失或不是字符串")
	}

	timeout := t.timeout
	if v, ok := inputNumber(input, "timeout_seconds"); ok && v > 0 {
		timeout = time.Duration(v) * time.Second
		if timeout > maxCommandTimeout {
			timeout = maxCommandTimeout
		}
	}

	if err := t.gate.CheckCommand(command); err != nil {
		return "", err
	}

	absWorkingDir, err := filepath.Abs(t.workingDir)
	if err != nil {
		return "", fmt.Errorf("无法获取工作目录: %w", err)
	}

	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(execCtx, "bash", "-c", command)
	cmd.Dir = absWorkingDir
	// 交互式命令（git 要口令、ssh 要确认）没有终端可读，会直接失败而不是挂住整个会话
	cmd.Stdin = nil
	cmd.Env = []string{
		"HOME=" + os.Getenv("HOME"),
		"PATH=" + os.Getenv("PATH"),
		"USER=" + os.Getenv("USER"),
		"PWD=" + absWorkingDir,
		"LANG=C.UTF-8",
		"GIT_TERMINAL_PROMPT=0",
		"GOFLAGS=",
	}
	useProcessGroup(cmd)

	var mu sync.Mutex
	var chunks [][]byte
	const captureLimit = 512 * 1024
	captured := 0

	capture := func(p []byte) {
		mu.Lock()
		defer mu.Unlock()
		if captured >= captureLimit {
			return
		}
		keep := min(len(p), captureLimit-captured)
		chunks = append(chunks, append([]byte(nil), p[:keep]...))
		captured += keep
	}

	collect := writerFunc(func(p []byte) (int, error) { capture(p); return len(p), nil })
	stdout := &lineEcho{prefix: "", out: t.out, mu: &mu}
	stderr := &lineEcho{prefix: "[stderr] ", out: t.out, mu: &mu}

	cmd.Stdout = io.MultiWriter(stdout, collect)
	cmd.Stderr = io.MultiWriter(stderr, collect)

	startTime := time.Now()
	runErr := cmd.Run()
	elapsed := time.Since(startTime)
	stdout.flush()
	stderr.flush()

	output := strings.TrimRight(string(bytes.Join(chunks, nil)), "\n")
	dropped := captured >= captureLimit

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("💻 $ %s\n", command))

	// 输出截断：保留头尾，中间省略——头通常是命令回显与首个错误，尾通常是退出原因
	const maxOutput = 8000
	if len(output) > maxOutput {
		head := output[:maxOutput/2]
		tail := output[len(output)-maxOutput/2:]
		sb.WriteString(head)
		sb.WriteString(fmt.Sprintf("\n… 中间 %d 字节省略 …\n", len(output)-maxOutput))
		sb.WriteString(tail)
	} else {
		sb.WriteString(output)
	}
	if dropped {
		sb.WriteString("\n… 输出超过 512KB，之后的内容未记录 …\n")
	}

	switch {
	case runErr == nil:
		sb.WriteString(fmt.Sprintf("\n✅ 执行成功，耗时 %s", elapsed.Round(time.Millisecond)))
	case execCtx.Err() == context.DeadlineExceeded:
		sb.WriteString(fmt.Sprintf("\n⏱ 命令超时（%v）已终止，耗时 %s；如需更久请带 timeout_seconds 重试",
			timeout, elapsed.Round(time.Millisecond)))
	case ctx.Err() != nil:
		sb.WriteString(fmt.Sprintf("\n⚠️ 命令被用户中断，耗时 %s", elapsed.Round(time.Millisecond)))
	default:
		if exitErr, ok := runErr.(*exec.ExitError); ok {
			sb.WriteString(fmt.Sprintf("\n❌ 退出码: %d，耗时 %s", exitErr.ExitCode(), elapsed.Round(time.Millisecond)))
		} else {
			sb.WriteString(fmt.Sprintf("\n❌ 执行失败: %v，耗时 %s", runErr, elapsed.Round(time.Millisecond)))
		}
	}

	// 命令本身失败不作为工具错误返回：退出码和输出对模型就是有效信息，
	// 让它自己读日志决定下一步。
	// 输出同样是外部数据——被攻击的仓库里的构建脚本可以打印一段伪装成指令的文字。
	return GuardExternalContent(sb.String(), "", nil), nil
}

// lineEcho 把命令输出按行实时回显到终端。
// stdout 与 stderr 共用一把锁，避免两路输出的行互相交错撕碎。
type lineEcho struct {
	prefix  string
	out     io.Writer
	mu      *sync.Mutex
	pending []byte
}

func (e *lineEcho) Write(p []byte) (int, error) {
	if e.out == nil {
		return len(p), nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	e.pending = append(e.pending, p...)
	for {
		idx := bytes.IndexByte(e.pending, '\n')
		if idx < 0 {
			break
		}
		e.printLine(string(e.pending[:idx]))
		e.pending = e.pending[idx+1:]
	}
	return len(p), nil
}

// flush 输出结尾没有换行的残留内容。
func (e *lineEcho) flush() {
	if e.out == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if rest := strings.TrimRight(string(e.pending), "\r"); rest != "" {
		e.printLine(rest)
	}
	e.pending = e.pending[:0]
}

func (e *lineEcho) printLine(line string) {
	line = strings.TrimRight(line, "\r")
	if line == "" {
		return
	}
	const maxEcho = 200
	if len(line) > maxEcho {
		line = line[:maxEcho] + "…"
	}
	fmt.Fprintf(e.out, "  %s│ %s\n", e.prefix, line)
}

// maxCommandTimeout 单次命令允许的最长执行时间。
const maxCommandTimeout = 15 * time.Minute

// writerFunc 把函数适配成 io.Writer。
type writerFunc func(p []byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }
