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
// EditFileTool — 精确片段替换
// ============================================================================

// EditFileTool 用精确匹配替换文件中的一段内容。
//
// 相比 write_file 的整文件覆盖，它不需要模型把整个文件重抄一遍：
// 既省 token，也不会把没留意到的部分改掉。
type EditFileTool struct {
	workingDir string
	gate       *Gate
	state      *FileState
	index      *RepoIndex
}

// NewEditFileTool 创建 edit_file 工具。
func NewEditFileTool(workingDir string, gate *Gate, state *FileState) *EditFileTool {
	return &EditFileTool{workingDir: workingDir, gate: gate, state: state}
}

// WithIndex 挂上会话级元信息缓存，改动后让该路径的判定作废。
func (t *EditFileTool) WithIndex(idx *RepoIndex) *EditFileTool {
	t.index = idx
	return t
}

// Name 返回工具名。
func (t *EditFileTool) Name() string { return "edit_file" }

// Mutating 报告该工具会改文件。
func (t *EditFileTool) Mutating() bool { return true }

// Definition 返回 edit_file 工具的 JSON Schema。
func (t *EditFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "edit_file",
		Description: "在已有文件中精确替换一段内容。old_string 必须与文件中的原文逐字符一致（含缩进与换行），且在默认情况下只能出现一次。修改前必须先用 read_file 读过该文件。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"file_path": {
					Type:        "string",
					Description: "文件路径，相对于工作目录",
				},
				"old_string": {
					Type:        "string",
					Description: "要被替换的原文，需与文件内容完全一致",
				},
				"new_string": {
					Type:        "string",
					Description: "替换后的新内容；删除原文时传空字符串",
				},
				"replace_all": {
					Type:        "string",
					Description: "是否替换全部匹配，填 \"true\" 或 \"false\"，默认 false",
				},
			},
			Required: []string{"file_path", "old_string", "new_string"},
		},
	}
}

// Execute 实现 edit_file 的执行逻辑。
func (t *EditFileTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	filePath, ok := input["file_path"].(string)
	if !ok || filePath == "" {
		return "", fmt.Errorf("参数 file_path 缺失或不是字符串")
	}
	oldStr, ok := input["old_string"].(string)
	if !ok || oldStr == "" {
		return "", fmt.Errorf("参数 old_string 缺失或为空")
	}
	newStr, ok := input["new_string"].(string)
	if !ok {
		return "", fmt.Errorf("参数 new_string 缺失（不改动请传与 old_string 相同的内容）")
	}
	replaceAll := boolParam(input, "replace_all")

	absPath, err := resolveSafe(t.workingDir, filePath, false)
	if err != nil {
		return "", err
	}
	// 改密钥类文件会把原内容明文铺在确认框里，等于把凭据搬进对话
	if SensitivePath(filePath) {
		return "", ErrSensitivePath(filePath)
	}
	if !t.state.HasSeen(absPath) {
		return "", &StaleFileError{Path: filePath}
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("读取文件失败: %w", err)
	}
	text := string(data)

	count := strings.Count(text, oldStr)
	switch {
	case count == 0:
		return "", fmt.Errorf("在 %s 中未找到 old_string；请重新 read_file 确认原文（注意缩进和空行）", filePath)
	case count > 1 && !replaceAll:
		return "", fmt.Errorf("old_string 在 %s 中出现 %d 次，无法确定改哪一处；请扩大上下文使其唯一，或指定 replace_all=true",
			filePath, count)
	}

	updated := strings.ReplaceAll(text, oldStr, newStr)
	secretWarning, err := CheckSecretWrite(t.gate.Mode, newStr)
	if err != nil {
		return "", err
	}
	detail := fmt.Sprintf("修改 %s：%d 处替换，%d → %d 字节", filePath, count, len(text), len(updated))
	if t.gate.Mode == ModeAsk {
		if secretWarning != "" {
			detail += "\n" + secretWarning
		}
		if sources := SourcesLine(t.state); sources != "" {
			detail += "\n" + sources
		}
		detail += "\n" + UnifiedDiff(filePath, text, updated)
	}
	if err := t.gate.Approve(t.Name(), detail); err != nil {
		return "", err
	}

	if err := backupFile(absPath); err != nil {
		return "", fmt.Errorf("备份原文件失败: %w", err)
	}
	// 先写临时文件再改名，避免中途失败留下半截内容
	tmp := absPath + ".tmp"
	if err := os.WriteFile(tmp, []byte(updated), 0o644); err != nil {
		os.Remove(tmp)
		return "", fmt.Errorf("写入失败: %w", err)
	}
	if err := os.Rename(tmp, absPath); err != nil {
		os.Remove(tmp)
		return "", fmt.Errorf("替换文件失败: %w", err)
	}
	t.index.Forget(absPath)
	t.state.MarkRead(filePath, absPath)

	return fmt.Sprintf("✅ 已修改 %s（替换 %d 处，%d → %d 字节，备份 %s）",
		filePath, count, len(text), len(updated), backupPath(absPath)), nil
}

// boolParam 读取布尔参数，兼容 true / "true" 两种 JSON 写法。
func boolParam(input map[string]any, key string) bool {
	switch v := input[key].(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(v, "true")
	default:
		return false
	}
}

// ============================================================================
// MoveFileTool / DeleteFileTool
// ============================================================================

// MoveFileTool 重命名或移动文件。
type MoveFileTool struct {
	workingDir string
	gate       *Gate
	state      *FileState
	index      *RepoIndex
}

// NewMoveFileTool 创建 move_file 工具。
func NewMoveFileTool(workingDir string, gate *Gate, state *FileState) *MoveFileTool {
	return &MoveFileTool{workingDir: workingDir, gate: gate, state: state}
}

// WithIndex 挂上会话级元信息缓存，移动后两端路径的判定都作废。
func (t *MoveFileTool) WithIndex(idx *RepoIndex) *MoveFileTool {
	t.index = idx
	return t
}

// Name 返回工具名。
func (t *MoveFileTool) Name() string { return "move_file" }

// Mutating 报告该工具会改文件系统。
func (t *MoveFileTool) Mutating() bool { return true }

// Definition 返回 move_file 工具的 JSON Schema。
func (t *MoveFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "move_file",
		Description: "重命名或移动文件/目录，两个路径都必须在工作目录内。目标已存在时会拒绝。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"from_path": {Type: "string", Description: "原路径，相对于工作目录"},
				"to_path":   {Type: "string", Description: "新路径，相对于工作目录"},
			},
			Required: []string{"from_path", "to_path"},
		},
	}
}

// Execute 实现 move_file 的执行逻辑。
func (t *MoveFileTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	from, ok := input["from_path"].(string)
	if !ok || from == "" {
		return "", fmt.Errorf("参数 from_path 缺失或不是字符串")
	}
	to, ok := input["to_path"].(string)
	if !ok || to == "" {
		return "", fmt.Errorf("参数 to_path 缺失或不是字符串")
	}

	absFrom, err := resolveSafe(t.workingDir, from, false)
	if err != nil {
		return "", err
	}
	absTo, err := resolveSafe(t.workingDir, to, true)
	if err != nil {
		return "", err
	}
	if _, err := os.Lstat(absTo); err == nil {
		return "", fmt.Errorf("目标 %s 已存在，拒绝覆盖", to)
	}

	if err := t.gate.Approve(t.Name(), fmt.Sprintf("移动 %s → %s", from, to)); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(absTo), 0o755); err != nil {
		return "", fmt.Errorf("创建目标目录失败: %w", err)
	}
	if err := os.Rename(absFrom, absTo); err != nil {
		return "", fmt.Errorf("移动失败: %w", err)
	}
	t.index.Forget(absFrom)
	t.index.Forget(absTo)

	return fmt.Sprintf("✅ 已移动 %s → %s", from, to), nil
}

// DeleteFileTool 删除工作目录内的文件或目录。
type DeleteFileTool struct {
	workingDir string
	gate       *Gate
	state      *FileState
	index      *RepoIndex
}

// NewDeleteFileTool 创建 delete_file 工具。
func NewDeleteFileTool(workingDir string, gate *Gate, state *FileState) *DeleteFileTool {
	return &DeleteFileTool{workingDir: workingDir, gate: gate, state: state}
}

// WithIndex 挂上会话级元信息缓存，删除后该路径的判定作废。
func (t *DeleteFileTool) WithIndex(idx *RepoIndex) *DeleteFileTool {
	t.index = idx
	return t
}

// Name 返回工具名。
func (t *DeleteFileTool) Name() string { return "delete_file" }

// Mutating 报告该工具会删除文件。
func (t *DeleteFileTool) Mutating() bool { return true }

// Definition 返回 delete_file 工具的 JSON Schema。
func (t *DeleteFileTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "delete_file",
		Description: "删除工作目录内的一个文件或目录。删除目录需要 recursive=true，且永远不要用它删工作目录本身。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"file_path": {Type: "string", Description: "要删除的路径，相对于工作目录"},
				"recursive": {Type: "string", Description: "删除目录时填 \"true\""},
			},
			Required: []string{"file_path"},
		},
	}
}

// Execute 实现 delete_file 的执行逻辑。
func (t *DeleteFileTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	filePath, ok := input["file_path"].(string)
	if !ok || filePath == "" {
		return "", fmt.Errorf("参数 file_path 缺失或不是字符串")
	}

	absPath, err := resolveSafe(t.workingDir, filePath, false)
	if err != nil {
		return "", err
	}
	absWorking, err := filepath.Abs(t.workingDir)
	if err == nil && absPath == absWorking {
		return "", fmt.Errorf("不允许删除工作目录本身")
	}
	if isGitInternal(absPath) {
		return "", fmt.Errorf("不允许删除 .git 目录内容")
	}

	info, err := os.Lstat(absPath)
	if err != nil {
		return "", fmt.Errorf("路径不存在: %s", filePath)
	}

	if info.IsDir() && !boolParam(input, "recursive") {
		return "", fmt.Errorf("%s 是目录，需要 recursive=true 才能删除", filePath)
	}

	if err := t.gate.Approve(t.Name(), fmt.Sprintf("删除 %s（%s，无法撤销）",
		filePath, humanSize(info.Size()))); err != nil {
		return "", err
	}

	if info.IsDir() {
		err = os.RemoveAll(absPath)
	} else {
		err = os.Remove(absPath)
	}
	if err != nil {
		return "", fmt.Errorf("删除失败: %w", err)
	}
	t.index.Forget(absPath)
	return fmt.Sprintf("✅ 已删除 %s", filePath), nil
}

// isGitInternal 判断路径是否落在 .git 目录里。
func isGitInternal(absPath string) bool {
	for _, seg := range strings.Split(absPath, string(filepath.Separator)) {
		if seg == ".git" {
			return true
		}
	}
	return false
}

// humanSize 把字节数转成简短可读的形式。
func humanSize(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
