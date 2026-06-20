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
// WriteFileTool — 写入文件内容
// ============================================================================

// WriteFileTool 将内容写入指定文件，自动创建父目录。
type WriteFileTool struct {
	workingDir string
}

// NewWriteFileTool 创建 write_file 工具。
func NewWriteFileTool(workingDir string) *WriteFileTool {
	return &WriteFileTool{workingDir: workingDir}
}

// Definition 返回 write_file 工具的 JSON Schema。
func (t *WriteFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "write_file",
		Description: "创建或覆盖写入文件。会自动创建目录。如需修改文件，请先使用 read_file 读取当前内容。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"file_path": {
					Type:        "string",
					Description: "要写入的文件路径，相对于工作目录的相对路径",
				},
				"content": {
					Type:        "string",
					Description: "要写入文件的完整内容",
				},
			},
			Required: []string{"file_path", "content"},
		},
	}
}

// Execute 实现 write_file 的执行逻辑。
func (t *WriteFileTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	filePath, ok := input["file_path"].(string)
	if !ok || filePath == "" {
		return "", fmt.Errorf("参数 file_path 缺失或不是字符串")
	}

	content, ok := input["content"].(string)
	if !ok {
		return "", fmt.Errorf("参数 content 缺失或不是字符串")
	}

	absPath, err := t.safePath(filePath)
	if err != nil {
		return "", err
	}

	// 创建父目录
	dir := filepath.Dir(absPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("创建目录失败: %w", err)
	}

	// 检查是否已存在
	_, statErr := os.Stat(absPath)
	existed := statErr == nil

	// 写入文件
	if err := os.WriteFile(absPath, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("写入文件失败: %w", err)
	}

	if existed {
		return fmt.Sprintf("✅ 已更新文件 %s (%d 字节)", filePath, len(content)), nil
	}
	return fmt.Sprintf("✅ 已创建文件 %s (%d 字节)", filePath, len(content)), nil
}

// safePath 与 ReadFileTool 保持一致。
func (t *WriteFileTool) safePath(relPath string) (string, error) {
	if filepath.IsAbs(relPath) {
		return "", fmt.Errorf("不允许使用绝对路径: %s", relPath)
	}

	absWorkingDir, err := filepath.Abs(t.workingDir)
	if err != nil {
		return "", fmt.Errorf("无法获取工作目录的绝对路径: %w", err)
	}

	absPath := filepath.Join(absWorkingDir, relPath)
	absPath = filepath.Clean(absPath)

	if !strings.HasPrefix(absPath, absWorkingDir+string(filepath.Separator)) && absPath != absWorkingDir {
		return "", fmt.Errorf("路径越界: %s 不在工作目录内", relPath)
	}

	return absPath, nil
}
