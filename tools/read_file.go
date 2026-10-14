package tools

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"go-agent/llm"
)

// ============================================================================
// ReadFileTool — 读取文件内容，带行号显示
// ============================================================================

// ReadFileTool 读取指定文件的内容，以带行号的格式返回。
// 大文件按 start_line 分段读取，并记录该文件已被读过。
type ReadFileTool struct {
	workingDir string
	state      *FileState
}

// NewReadFileTool 创建 read_file 工具。
func NewReadFileTool(workingDir string, state *FileState) *ReadFileTool {
	return &ReadFileTool{workingDir: workingDir, state: state}
}

// Name 返回工具名。
func (t *ReadFileTool) Name() string { return "read_file" }

// Mutating 报告该工具不会改变文件系统。
func (t *ReadFileTool) Mutating() bool { return false }

// Definition 返回 read_file 工具的 JSON Schema。
func (t *ReadFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "read_file",
		Description: "读取指定文件的内容，以行号格式返回。默认最多 2000 行；大文件请配合 start_line 分段读取，不要一次读完整文件。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"file_path": {
					Type:        "string",
					Description: "要读取的文件路径，相对于工作目录的相对路径",
				},
				"start_line": {
					Type:        "string",
					Description: "起始行号（1-based），默认从第 1 行开始",
				},
				"max_lines": {
					Type:        "string",
					Description: "最多返回多少行，默认 2000",
				},
			},
			Required: []string{"file_path"},
		},
	}
}

// Execute 实现 read_file 的执行逻辑。
func (t *ReadFileTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	filePath, ok := input["file_path"].(string)
	if !ok || filePath == "" {
		return "", fmt.Errorf("参数 file_path 缺失或不是字符串")
	}

	// 安全检查：路径必须在工作目录内
	absPath, err := resolveSafe(t.workingDir, filePath, false)
	if err != nil {
		return "", err
	}
	if SensitivePath(filePath) {
		return "", ErrSensitivePath(filePath)
	}

	// 读取文件
	data, err := os.ReadFile(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("文件不存在: %s", filePath)
		}
		return "", fmt.Errorf("读取文件失败: %w", err)
	}

	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	t.state.MarkRead(filePath, absPath)

	// 起始行（1-based）
	start := 0
	if v, ok := inputNumber(input, "start_line"); ok {
		if v < 1 {
			return "", fmt.Errorf("参数 start_line 必须是正整数，收到: %v", v)
		}
		start = v - 1
	}
	if start >= len(lines) {
		return "", fmt.Errorf("起始行 %d 超出文件长度（共 %d 行）", start+1, len(lines))
	}

	// 行数上限
	maxLines := defaultMaxLines
	if v, ok := inputNumber(input, "max_lines"); ok && v > 0 {
		maxLines = min(v, defaultMaxLines)
	}

	end := min(start+maxLines, len(lines))
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("📄 %s (第 %d-%d 行，共 %d 行)\n\n",
		filePath, start+1, end, len(lines)))

	const maxLineBytes = 400 // 压缩过的单行文件不会把整个上下文吃掉的保险丝
	for i := start; i < end; i++ {
		line := lines[i]
		if len(line) > maxLineBytes {
			line = line[:maxLineBytes] + fmt.Sprintf(" …（本行共 %d 字节，已截断）", len(lines[i]))
		}
		// 行号按实际宽度输出：定宽的 "%6d | " 让每行多背 9 个字节，
		// 一次 400 行的读取里光前缀就占掉两成——行号是给模型定位用的，不是给终端对齐用的。
		sb.WriteString(fmt.Sprintf("%d| %s\n", i+1, line))
	}

	if end < len(lines) {
		sb.WriteString(fmt.Sprintf("\n… 还有 %d 行未显示，可用 start_line=%d 继续读\n",
			len(lines)-end, end+1))
	}

	return GuardExternalContent(sb.String(), filePath, t.state), nil
}

// defaultMaxLines 单次读取的默认行数上限。
const defaultMaxLines = 2000

// inputNumber 从工具参数里取一个整数，兼容 JSON 数字和字符串两种写法。
func inputNumber(input map[string]any, key string) (int, bool) {
	switch v := input[key].(type) {
	case float64:
		return int(v), true
	case string:
		n, err := strconv.Atoi(v)
		if err != nil {
			return 0, false
		}
		return n, true
	default:
		return 0, false
	}
}

// ParallelSafe 报告该工具可以并发执行：只读、不占终端、不派子循环。
func (t *ReadFileTool) ParallelSafe() bool { return true }
