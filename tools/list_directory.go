package tools

import (
	"context"
	"fmt"
	"io/fs"
	"os"
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

// Name 返回工具名。
func (t *ListDirectoryTool) Name() string { return "list_directory" }

// Mutating 报告该工具不会改动文件系统。
func (t *ListDirectoryTool) Mutating() bool { return false }

// Definition 返回 list_directory 工具的 JSON Schema。
func (t *ListDirectoryTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "list_directory",
		Description: "列出指定目录中的文件和子目录，显示名称、大小和修改时间。目录在前，文件在后，最多显示 300 项。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"directory": {
					Type:        "string",
					Description: "要列出的目录路径，相对于工作目录的相对路径。默认为工作目录根目录 '.'",
				},
			},
			Required: []string{},
		},
	}
}

// Execute 实现 list_directory 的执行逻辑。
func (t *ListDirectoryTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	dirPath := "."
	if d, ok := input["directory"].(string); ok && d != "" {
		dirPath = d
	}

	absPath, err := resolveSafe(t.workingDir, dirPath, false)
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
		Name    string
		IsDir   bool
		Size    int64
		Mode    fs.FileMode
		ModTime string
	}

	const maxEntries = 300
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
	sb.WriteString(fmt.Sprintf("📁 %s (共 %d 项", dirPath, len(files)))
	if len(files) > maxEntries {
		sb.WriteString(fmt.Sprintf("，仅显示前 %d 项", maxEntries))
	}
	sb.WriteString(")\n\n")

	for i, f := range files {
		if i >= maxEntries {
			break
		}
		typeIcon := "📄"
		if f.IsDir {
			typeIcon = "📁"
		}
		sb.WriteString(fmt.Sprintf("  %s %-40s %10s  %s\n",
			f.Mode.String(),
			typeIcon+" "+f.Name,
			humanSize(f.Size),
			f.ModTime,
		))
	}

	// 目录名也是外部文字：仓库里可以有人把目录取名叫 "ignore previous instructions…"。
	// relPath 传空、state 传 nil 是有意的：列出的不是某个文件的内容，
	// 把目录记进"依据文件"会让写操作的确认框点名一个 read_file 永远读不到的路径。
	return GuardExternalContent(sb.String(), "", nil), nil
}

// ParallelSafe 报告该工具可以并发执行：只读、不占终端、不派子循环。
func (t *ListDirectoryTool) ParallelSafe() bool { return true }
