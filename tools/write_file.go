package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"go-agent/llm"
)

// ============================================================================
// WriteFileTool — 写入文件内容
// ============================================================================

// WriteFileTool 创建或整文件覆盖写入。修改已有文件优先用 edit_file，
// 它只替换指定片段，不会把没读过的内容一起丢掉。
type WriteFileTool struct {
	workingDir string
	gate       *Gate
	state      *FileState
}

// NewWriteFileTool 创建 write_file 工具。
func NewWriteFileTool(workingDir string, gate *Gate, state *FileState) *WriteFileTool {
	return &WriteFileTool{workingDir: workingDir, gate: gate, state: state}
}

// Name 返回工具名。
func (t *WriteFileTool) Name() string { return "write_file" }

// Mutating 报告该工具会写文件。
func (t *WriteFileTool) Mutating() bool { return true }

// Definition 返回 write_file 工具的 JSON Schema。
func (t *WriteFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "write_file",
		Description: "创建新文件或整文件覆盖写入，会自动创建父目录。修改已有文件请优先使用 edit_file；覆盖前必须先 read_file 读过该文件。",
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

	absPath, err := resolveSafe(t.workingDir, filePath, true)
	if err != nil {
		return "", err
	}

	existed, err := t.state.CheckWritable(filePath, absPath)
	if err != nil {
		return "", err
	}
	if len(content) > maxWriteBytes {
		return "", fmt.Errorf("单次写入不得超过 %d 字节，当前 %d 字节；请分多次写或用 edit_file",
			maxWriteBytes, len(content))
	}

	if err := t.gate.Approve(t.Name(), DescribeWrite(filePath, len(content), existed)); err != nil {
		return "", err
	}

	if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
		return "", fmt.Errorf("创建目录失败: %w", err)
	}

	// 覆盖前留一份上一版，用户发现改坏了还能捞回来
	if existed {
		if err := backupFile(absPath); err != nil {
			return "", fmt.Errorf("备份原文件失败: %w", err)
		}
	}

	if err := os.WriteFile(absPath, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("写入文件失败: %w", err)
	}
	t.state.Mark(absPath)

	if existed {
		return fmt.Sprintf("✅ 已覆盖 %s（%d 字节，上一版备份在 %s）",
			filePath, len(content), backupPath(absPath)), nil
	}
	return fmt.Sprintf("✅ 已创建 %s（%d 字节）", filePath, len(content)), nil
}

// maxWriteBytes 单次写入的字节上限，防止一次吐出超大文件把上下文占满。
const maxWriteBytes = 512 * 1024

// backupPath 返回备份文件路径。
func backupPath(absPath string) string {
	return absPath + ".bak"
}

// backupFile 把现有文件复制成 .bak。
func backupFile(absPath string) error {
	data, err := os.ReadFile(absPath)
	if err != nil {
		return err
	}
	return os.WriteFile(backupPath(absPath), data, 0o644)
}
