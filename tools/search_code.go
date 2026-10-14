package tools

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"go-agent/llm"
)

// ============================================================================
// SearchCodeTool — 正则搜索代码内容
// ============================================================================

// SearchCodeTool 在目录树中使用正则表达式搜索文件内容。
type SearchCodeTool struct {
	workingDir string
	index      *RepoIndex
}

// NewSearchCodeTool 创建 search_code 工具。
func NewSearchCodeTool(workingDir string) *SearchCodeTool {
	return &SearchCodeTool{workingDir: workingDir}
}

// WithIndex 挂上会话级元信息缓存。不挂也能跑，只是每趟搜索都重新判定一遍
// 哪些文件是二进制——大仓库里这一项就是每趟几千次 open+read。
func (t *SearchCodeTool) WithIndex(idx *RepoIndex) *SearchCodeTool {
	t.index = idx
	return t
}

// Name 返回工具名。
func (t *SearchCodeTool) Name() string { return "search_code" }

// Mutating 报告该工具不会改动文件系统。
func (t *SearchCodeTool) Mutating() bool { return false }

// Definition 返回 search_code 工具的 JSON Schema。
func (t *SearchCodeTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name:        "search_code",
		Description: "在目录树中使用正则表达式搜索文件内容（类似 grep）。自动忽略 .git 目录、二进制文件和密钥类文件（.env、*.pem、.ssh/ 等）。默认最多返回 500 条匹配。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"pattern": {
					Type:        "string",
					Description: "正则表达式搜索模式（Go regexp 语法）",
				},
				"directory": {
					Type:        "string",
					Description: "搜索的目录，相对于工作目录的相对路径。默认为工作目录根目录 '.'",
				},
				"file_pattern": {
					Type:        "string",
					Description: "文件名匹配通配符（如 '*.go'、'*.md'）。默认为 '*'（所有文件）",
				},
			},
			Required: []string{"pattern"},
		},
	}
}

// Execute 实现 search_code 的执行逻辑。
func (t *SearchCodeTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	pattern, ok := input["pattern"].(string)
	if !ok || pattern == "" {
		return "", fmt.Errorf("参数 pattern 缺失或不是字符串")
	}

	searchDir := "."
	if d, ok := input["directory"].(string); ok && d != "" {
		searchDir = d
	}

	fileGlob := "*"
	if g, ok := input["file_pattern"].(string); ok && g != "" {
		fileGlob = g
	}

	absSearchDir, err := resolveSafe(t.workingDir, searchDir, false)
	if err != nil {
		return "", err
	}
	absWorkingDir, err := filepath.Abs(t.workingDir)
	if err != nil {
		return "", fmt.Errorf("无法解析工作目录: %w", err)
	}

	// 编译正则
	re, err := regexp.Compile(pattern)
	if err != nil {
		return "", fmt.Errorf("正则表达式编译失败: %w", err)
	}

	const maxMatches = 500

	type match struct {
		file    string
		lineNo  int
		content string
	}

	var matches []match
	totalScanned := 0
	skippedSensitive := 0

	walkErr := filepath.WalkDir(absSearchDir, func(path string, d fs.DirEntry, err error) error {
		// 检查上下文取消
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err != nil {
			return nil // 跳过无法访问的文件/目录
		}

		// 跳过 .git、vendor、node_modules，以及智能体自己落盘的 .go-agent
		if d.IsDir() {
			if isIgnoredDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}

		// 文件名通配符匹配
		if fileGlob != "*" {
			matched, err := filepath.Match(fileGlob, d.Name())
			if err != nil || !matched {
				return nil
			}
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}

		// 跳过超大文件，避免为了搜索把内存吃满
		if info.Size() > maxSearchFileBytes {
			return nil
		}

		// 跳过二进制文件（快速检查：前 512 字节中有无 null 字节）
		if !t.index.IsTextFile(path, info) {
			return nil
		}

		relPath, err := filepath.Rel(absWorkingDir, path)
		if err != nil {
			relPath = path
		}

		// 密钥类文件不参与搜索：grep 到一行就等于把凭据的一个片段搬进对话
		if SensitivePath(relPath) {
			skippedSensitive++
			return nil
		}

		totalScanned++

		// 搜索文件内容
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		lines := strings.Split(string(data), "\n")
		for lineNo, line := range lines {
			if re.MatchString(line) {
				matches = append(matches, match{
					file:    relPath,
					lineNo:  lineNo + 1,
					content: line,
				})
				if len(matches) >= maxMatches {
					return fmt.Errorf("max_matches_reached") // 用于中断遍历
				}
			}
		}

		return nil
	})

	// 忽略因达到上限而中断的错误
	if walkErr != nil && walkErr.Error() != "max_matches_reached" {
		return "", fmt.Errorf("搜索失败: %w", walkErr)
	}

	// 格式化输出
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("🔍 在 %s 中搜索 /%s/ (扫描 %d 个文件)\n\n", searchDir, pattern, totalScanned))

	if len(matches) == 0 {
		sb.WriteString("  (未找到匹配项)\n")
		if skippedSensitive > 0 {
			sb.WriteString(fmt.Sprintf("\n  （另有 %d 个密钥类文件未参与搜索，本工具不读取凭据）\n", skippedSensitive))
		}
		return sb.String(), nil
	}

	for _, m := range matches {
		// 截断过长行
		content := m.content
		if len(content) > 200 {
			content = content[:200] + "..."
		}
		sb.WriteString(fmt.Sprintf("  %s:%d: %s\n", m.file, m.lineNo, content))
	}

	sb.WriteString(fmt.Sprintf("\n  共找到 %d 条匹配", len(matches)))
	if len(matches) >= maxMatches {
		sb.WriteString(" (结果被截断，已达到 500 条上限)")
	}
	sb.WriteString("\n")
	if skippedSensitive > 0 {
		sb.WriteString(fmt.Sprintf("\n  （另有 %d 个密钥类文件未参与搜索，本工具不读取凭据）\n", skippedSensitive))
	}

	// 匹配到的行是别人写下的文字，同样可能是伪装成指令的剧本
	return GuardExternalContent(sb.String(), "", nil), nil
}

// isBinary 通过检查文件是否包含 null 字节判断是否为二进制文件。
func isBinary(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return true // 无法打开，假定为二进制
	}
	defer f.Close()

	buf := make([]byte, 512)
	n, _ := f.Read(buf)
	for _, b := range buf[:n] {
		if b == 0 {
			return true
		}
	}
	return false
}

// maxSearchFileBytes 单个文件参与内容搜索的大小上限。
const maxSearchFileBytes = 2 * 1024 * 1024

// ParallelSafe 报告该工具可以并发执行：只读、不占终端、不派子循环。
func (t *SearchCodeTool) ParallelSafe() bool { return true }
