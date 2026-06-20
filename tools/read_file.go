package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"go-agent/llm"
)

// ============================================================================
// ReadFileTool — 读取文件内容，带行号显示
// ============================================================================

// ReadFileTool 读取指定文件的内容，以带行号的格式返回。
// 超过 2000 行自动截断并添加提示。
type ReadFileTool struct {
	workingDir string
}

// NewReadFileTool 创建 read_file 工具。
func NewReadFileTool(workingDir string) *ReadFileTool {
	return &ReadFileTool{workingDir: workingDir}
}

// Definition 返回 read_file 工具的 JSON Schema。
func (t *ReadFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "read_file",
		Description: "读取指定文件的内容，以行号格式返回。支持读取任意文本文件。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"file_path": {
					Type:        "string",
					Description: "要读取的文件路径，相对于工作目录的相对路径",
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
	absPath, err := t.safePath(filePath)
	if err != nil {
		return "", err
	}

	// 读取文件
	data, err := os.ReadFile(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("文件不存在: %s", filePath)
		}
		return "", fmt.Errorf("读取文件失败: %w", err)
	}

	// 按行分割并添加行号
	lines := strings.Split(string(data), "\n")
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("📄 %s (%d 行)\n\n", filePath, len(lines)))

	const maxLines = 2000
	truncated := len(lines) > maxLines
	displayLines := lines
	if truncated {
		displayLines = lines[:maxLines]
	}

	for i, line := range displayLines {
		sb.WriteString(fmt.Sprintf("%6d | %s\n", i+1, line))
	}

	if truncated {
		sb.WriteString(fmt.Sprintf("\n... (输出被截断，共 %d 行，仅显示前 %d 行)\n", len(lines), maxLines))
	}

	return sb.String(), nil
}

// safePath 将相对路径解析为绝对路径，并验证其位于工作目录内。
func (t *ReadFileTool) safePath(relPath string) (string, error) {
	// 拒绝绝对路径
	if filepath.IsAbs(relPath) {
		return "", fmt.Errorf("不允许使用绝对路径: %s", relPath)
	}

	// 规范化路径
	absWorkingDir, err := filepath.Abs(t.workingDir)
	if err != nil {
		return "", fmt.Errorf("无法获取工作目录的绝对路径: %w", err)
	}

	absPath := filepath.Join(absWorkingDir, relPath)
	absPath = filepath.Clean(absPath)

	// 验证在工作目录内
	if !strings.HasPrefix(absPath, absWorkingDir+string(filepath.Separator)) && absPath != absWorkingDir {
		return "", fmt.Errorf("路径越界: %s 不在工作目录内", relPath)
	}

	return absPath, nil
}
