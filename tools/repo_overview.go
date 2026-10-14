package tools

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"go-agent/llm"
)

// ============================================================================
// RepoOverviewTool — 一次给出仓库骨架
// ============================================================================

// RepoOverviewTool 输出目录骨架：每个目录的文件数、体量和语言分布。
//
// 存在的理由是冷启动那几轮的钱：模型要先摸清仓库结构，往往连开四五次
// list_directory 加 search_code，每次结果都整段进上下文。这里一次给完，
// 而且只给统计——文件内容仍然要靠 read_file，索引不代它臆测。
type RepoOverviewTool struct {
	workingDir string
}

// NewRepoOverviewTool 创建 repo_overview 工具。
func NewRepoOverviewTool(workingDir string) *RepoOverviewTool {
	return &RepoOverviewTool{workingDir: workingDir}
}

// Name 返回工具名。
func (t *RepoOverviewTool) Name() string { return "repo_overview" }

// Mutating 报告该工具不会改动文件系统。
func (t *RepoOverviewTool) Mutating() bool { return false }

// Definition 返回 repo_overview 工具的 JSON Schema。
func (t *RepoOverviewTool) Definition() llm.ToolDef {
	return llm.ToolDef{
		Name: "repo_overview",
		Description: "一次给出仓库骨架：每个目录有多少文件、体量、主要扩展名，以及被跳过的目录" +
			"（.git、vendor、node_modules 等）。用它在动手前摸清结构，代替连开多次 list_directory。" +
			"只给统计，不给文件内容——要看内容用 read_file 或 search_code。",
		InputSchema: llm.InputSchema{
			Type: "object",
			Properties: map[string]llm.Property{
				"directory": {
					Type:        "string",
					Description: "起始目录，相对于工作目录的相对路径，默认 '.'",
				},
				"max_depth": {
					Type:        "integer",
					Description: "最多展开几层，默认 3，上限 5。层级越深输出越长",
				},
			},
			Required: []string{},
		},
	}
}

const (
	overviewDefaultDepth = 3
	overviewMaxDepth     = 5
	overviewMaxDirs      = 150 // 目录行数上限：再宽的语言模型也读不完
)

// Execute 实现 repo_overview 的执行逻辑。
func (t *RepoOverviewTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	dirPath := "."
	if d, ok := input["directory"].(string); ok && d != "" {
		dirPath = d
	}
	maxDepth := overviewDefaultDepth
	if v, ok := input["max_depth"].(float64); ok && v > 0 {
		maxDepth = int(v)
	}
	if maxDepth > overviewMaxDepth {
		maxDepth = overviewMaxDepth
	}

	absDir, err := resolveSafe(t.workingDir, dirPath, false)
	if err != nil {
		return "", err
	}
	absWork, err := filepath.Abs(t.workingDir)
	if err != nil {
		return "", fmt.Errorf("无法解析工作目录: %w", err)
	}

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
	}

	skel := ScanSkeleton(absDir, absWork, maxDepth)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("🗂 %s 的骨架（展开 %d 层，共 %d 个文件 / %s）\n\n",
		dirPath, maxDepth, skel.TotalFiles, humanSize(skel.TotalBytes)))

	if len(skel.Dirs) == 0 {
		sb.WriteString("  (没有可列出的目录)\n")
		return GuardExternalContent(sb.String(), "", nil), nil
	}

	shown := skel.Dirs
	truncated := false
	if len(shown) > overviewMaxDirs {
		shown = shown[:overviewMaxDirs]
		truncated = true
	}

	extTotals := map[string]int{}
	secrets := 0
	for _, d := range skel.Dirs {
		secrets += d.Secrets
	}
	for _, d := range shown {
		indent := strings.Repeat("  ", d.Depth)
		line := fmt.Sprintf("%s%s  (%d 个文件, %s)", indent, d.Path, d.Files, humanSize(d.Bytes))
		if top := d.TopExts(4); len(top) > 0 {
			line += "  " + strings.Join(top, " ")
		}
		if d.Secrets > 0 {
			line += fmt.Sprintf("  [另有 %d 个密钥类文件未列出]", d.Secrets)
		}
		sb.WriteString(line + "\n")
		for e, c := range d.Exts {
			extTotals[e] += c
		}
	}

	if truncated {
		sb.WriteString(fmt.Sprintf("\n  （目录过多，只显示前 %d 行；用 directory 参数指定子树再看一次）\n",
			overviewMaxDirs))
	}

	// 整体语言分布：单看某个目录看不出这个项目由什么写成
	if len(extTotals) > 0 {
		type kv struct {
			ext string
			c   int
		}
		all := make([]kv, 0, len(extTotals))
		for e, c := range extTotals {
			all = append(all, kv{e, c})
		}
		sort.Slice(all, func(i, j int) bool {
			if all[i].c != all[j].c {
				return all[i].c > all[j].c
			}
			return all[i].ext < all[j].ext
		})
		parts := make([]string, 0, 6)
		for i, e := range all {
			if i >= 6 {
				break
			}
			parts = append(parts, fmt.Sprintf("%s:%d", e.ext, e.c))
		}
		sb.WriteString("\n  主要扩展名：" + strings.Join(parts, "  "))
		if len(all) > 6 {
			sb.WriteString(fmt.Sprintf("（共 %d 种）", len(all)))
		}
		sb.WriteString("\n")
	}

	if secrets > 0 {
		sb.WriteString(fmt.Sprintf("  %d 个密钥类文件只计数不列名，本工具不读取凭据\n", secrets))
	}
	if skel.IgnoredDirs > 0 {
		sb.WriteString(fmt.Sprintf("  已跳过 %d 个目录（.git、vendor、node_modules、.go-agent 等）\n", skel.IgnoredDirs))
	}

	// 目录名和文件名同样是外部数据：仓库里可以放一个名字叫伪装指令的文件
	return GuardExternalContent(sb.String(), "", nil), nil
}

// ParallelSafe 报告该工具可以并发执行：只读、不占终端、不派子循环。
func (t *RepoOverviewTool) ParallelSafe() bool { return true }
