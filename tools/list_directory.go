package tools

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"go-agent/llm"
)

// ============================================================================
// ListDirectoryTool — 列出目录内容
// ============================================================================

// ListDirectoryTool 列出指定目录中的文件和子目录。
type ListDirectoryTool struct {
	workingDir string
}

// NewListDirectoryTool 创建 list_directory 工具。
func NewListDirectoryTool(workingDir string) *ListDirectoryTool {
	return &ListDirectoryTool{workingDir: workingDir}
}

// Definition 返回 list_directory 工具的 JSON Schema。
func (t *ListDirectoryTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "list_directory",
		Description: "列出指定目录中的文件和子目录，显示名称、大小、权限和修改时间。目录在前，文件在后。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"directory": {
					Type:        "string",
					Description: "要列出的目录路径，相对于工作目录的相对路径。默认为工作目录根目录 '.'",
				},
			},
			Required: []string{}, // directory 可选，默认为 "."
		},
	}
}

// Execute 实现 list_directory 的执行逻辑。
func (t *ListDirectoryTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	dirPath := "."
	if d, ok := input["directory"].(string); ok && d != "" {
		dirPath = d
	}

	absPath, err := t.safePath(dirPath)
	if err != nil {
		return "", err
	}

	entries, err := os.ReadDir(absPath)
	if err != nil {
		return "", fmt.Errorf("读取目录失败: %w", err)
	}

	// 排序：目录在前，文件在后，同类按名称排序
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].IsDir() != entries[j].IsDir() {
			return entries[i].IsDir()
		}
		return entries[i].Name() < entries[j].Name()
	})

	type fileInfo struct {
		Name      string
		IsDir     bool
		Size      int64
		Mode      fs.FileMode
		ModTime   string
	}

	var files []fileInfo
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}
		files = append(files, fileInfo{
			Name:    entry.Name(),
			IsDir:   entry.IsDir(),
			Size:    info.Size(),
			Mode:    info.Mode(),
			ModTime: info.ModTime().Format("2006-01-02 15:04"),
		})
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("📁 %s (共 %d 项)\n\n", dirPath, len(files)))

	for _, f := range files {
		typeIcon := "📄"
		if f.IsDir {
			typeIcon = "📁"
		}
		sb.WriteString(fmt.Sprintf("  %s %-40s %10s %8d  %s\n",
			f.Mode.String(),
			typeIcon+" "+f.Name,
			formatSize(f.Size),
			f.Size,
			f.ModTime,
		))
	}

	return sb.String(), nil
}

// formatSize 将文件大小格式化为易读形式。
func formatSize(size int64) string {
	const unit = 1024
	if size < unit {
		return fmt.Sprintf("%d B", size)
	}
	div, exp := int64(unit), 0
	for n := size / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(size)/float64(div), "KMGTPE"[exp])
}

// safePath 与其他工具保持一致。
func (t *ListDirectoryTool) safePath(relPath string) (string, error) {
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
