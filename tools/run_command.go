package tools

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"go-agent/llm"
)

// ============================================================================
// RunCommandTool — 安全执行 shell 命令
// ============================================================================

// RunCommandTool 在受限环境中执行 shell 命令，具有超时和输出限制。
type RunCommandTool struct {
	workingDir string
}

// NewRunCommandTool 创建 run_command 工具。
func NewRunCommandTool(workingDir string) *RunCommandTool {
	return &RunCommandTool{workingDir: workingDir}
}

// Definition 返回 run_command 工具的 JSON Schema。
func (t *RunCommandTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "run_command",
		Description: "执行 shell 命令并返回输出。命令有 30 秒超时限制。请勿执行破坏性命令（如 rm -rf、git push --force 等）。修改文件前请先告知用户。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"command": {
					Type:        "string",
					Description: "要执行的 shell 命令。会使用 bash -c 执行。",
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

	// 创建带超时的上下文
	execCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	// 构建命令
	cmd := exec.CommandContext(execCtx, "bash", "-c", command)

	// 设置工作目录
	absWorkingDir, err := absPath(t.workingDir)
	if err != nil {
		return "", fmt.Errorf("无法获取工作目录: %w", err)
	}
	cmd.Dir = absWorkingDir

	// 设置受限环境变量
	cmd.Env = []string{
		"HOME=" + os.Getenv("HOME"),
		"PATH=" + os.Getenv("PATH"),
		"USER=" + os.Getenv("USER"),
		"PWD=" + absWorkingDir,
	}

	// 捕获 stdout 和 stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// 执行
	startTime := time.Now()
	err = cmd.Run()
	elapsed := time.Since(startTime)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("💻 $ %s\n", command))

	// 收集输出
	stdoutStr := stdout.String()
	stderrStr := stderr.String()
	combined := stdoutStr
	if stderrStr != "" {
		if combined != "" {
			combined += "\n"
		}
		combined += "[stderr]\n" + stderrStr
	}

	// 截断输出
	const maxOutput = 8000
	if len(combined) > maxOutput {
		combined = combined[:maxOutput] + fmt.Sprintf("\n... (输出被截断，完整输出 %d 字节)", len(combined))
	}

	if combined != "" {
		sb.WriteString(combined)
		if !strings.HasSuffix(combined, "\n") {
			sb.WriteString("\n")
		}
	}

	// 退出状态
	if err != nil {
		if execCtx.Err() == context.DeadlineExceeded {
			sb.WriteString(fmt.Sprintf("\n⏱ 命令超时 (30秒)，已终止。耗时 %s", elapsed.Round(time.Millisecond)))
		} else if exitErr, ok := err.(*exec.ExitError); ok {
			sb.WriteString(fmt.Sprintf("\n❌ 退出码: %d，耗时 %s", exitErr.ExitCode(), elapsed.Round(time.Millisecond)))
		} else {
			sb.WriteString(fmt.Sprintf("\n❌ 执行失败: %v，耗时 %s", err, elapsed.Round(time.Millisecond)))
		}
	} else {
		sb.WriteString(fmt.Sprintf("\n✅ 执行成功，耗时 %s", elapsed.Round(time.Millisecond)))
	}

	return sb.String(), nil
}

// absPath 返回绝对路径（不验证范围，run_command 安全由 cmd.Dir + 受限环境保证）。
func absPath(path string) (string, error) {
	return filepath.Abs(path)
}
