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
	index      *RepoIndex
}

// NewWriteFileTool 创建 write_file 工具。
func NewWriteFileTool(workingDir string, gate *Gate, state *FileState) *WriteFileTool {
	return &WriteFileTool{workingDir: workingDir, gate: gate, state: state}
}

// WithIndex 挂上会话级元信息缓存：写入后让该路径的结论作废，
// 否则下一趟搜索会拿着旧大小/旧的二进制判定说假话。
func (t *WriteFileTool) WithIndex(idx *RepoIndex) *WriteFileTool {
	t.index = idx
	return t
}

// Name 返回工具名。
func (t *WriteFileTool) Name() string { return "write_file" }

// Mutating 报告该工具会写文件。
func (t *WriteFileTool) Mutating() bool { return true }

// BodyParams 声明 content 是大正文参数：写入成功后全文已经在磁盘上，
// 历史里再留一份等于此后每一轮重复付费。
func (t *WriteFileTool) BodyParams() []string { return []string{"content"} }

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
	// 凭据文件由用户在终端管理：写入会覆盖真凭据，diff 又会把原内容摊进对话
	if SensitivePath(filePath) {
		return "", ErrSensitivePath(filePath)
	}

	existed, err := t.state.CheckWritable(filePath, absPath)
	if err != nil {
		return "", err
	}
	if len(content) > maxWriteBytes {
		return "", fmt.Errorf("单次写入不得超过 %d 字节，当前 %d 字节；请分多次写或用 edit_file",
			maxWriteBytes, len(content))
	}

	// 疑似密钥不许落盘：auto / readonly 下没人看确认框，直接拒绝
	warning, err := CheckSecretWrite(t.gate.Mode, content)
	if err != nil {
		return "", err
	}

	if err := t.gate.Approve(t.Name(), t.approvalDetail(filePath, absPath, content, existed, warning)); err != nil {
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
	t.index.Forget(absPath)
	t.state.MarkRead(filePath, absPath)

	if existed {
		return fmt.Sprintf("✅ 已覆盖 %s（%d 字节，上一版备份在 %s）",
			filePath, len(content), backupPath(absPath)), nil
	}
	return fmt.Sprintf("✅ 已创建 %s（%d 字节）", filePath, len(content)), nil
}

// maxWriteBytes 单次写入的字节上限，防止一次吐出超大文件把上下文占满。
const maxWriteBytes = 512 * 1024

// approvalDetail 组装确认框文本。
//
// 只有真的会问人才展开 diff 和依据文件：auto / readonly 模式下确认不会发生，
// 没必要为一次注定被丢弃的对比去读整个文件。
func (t *WriteFileTool) approvalDetail(relPath, absPath, content string, existed bool, secretWarning string) string {
	detail := DescribeWrite(relPath, len(content), existed)
	if t.gate.Mode != ModeAsk {
		return detail
	}
	var old string
	if existed {
		data, err := os.ReadFile(absPath)
		if err != nil {
			return detail + "\n（读取磁盘上的原文件失败，无法生成 diff）"
		}
		old = string(data)
	}
	if secretWarning != "" {
		detail += "\n" + secretWarning
	}
	if sources := SourcesLine(t.state); sources != "" {
		detail += "\n" + sources
	}
	return detail + "\n" + UnifiedDiff(relPath, old, content)
}

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
