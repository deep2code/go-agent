package tools

import (
	"fmt"
	"strings"
)

// 写操作确认框里的内容必须摊开给用户看。只有一句"覆盖 foo.go（2048 字节）"，
// 用户是在盲签——门还在，但签字的人没有信息。
//
// UnifiedDiff 生成行级 unified diff，专为终端确认框设计：输出有行数上限，
// 改动过大时退化成统计信息，而不是把一屏文件刷进终端。

const (
	diffContext     = 3    // 每个 hunk 前后保留的上下文行数
	diffMaxRegion   = 3000 // 参与最小 diff 计算的单边行数上限
	diffMaxCells    = 900000
	diffMaxOutLines = 160 // 确认框里最多显示多少行 diff
)

// UnifiedDiff 返回 old→new 的 diff 文本；name 用作新文件一侧的标题。
// oldContent 传空串表示文件尚不存在（整份新增）。
func UnifiedDiff(name string, oldContent, newContent string) string {
	if oldContent == newContent {
		return "（内容与磁盘上完全一致，没有实际改动）"
	}
	if containsNUL(oldContent) || containsNUL(newContent) {
		return "（看起来是二进制内容，不展开 diff）"
	}

	oldLines := splitLines(oldContent)
	newLines := splitLines(newContent)
	ops, coarse := diffOps(oldLines, newLines)

	var sb strings.Builder
	if len(oldLines) == 0 {
		sb.WriteString("--- (文件尚不存在)\n")
	} else {
		fmt.Fprintf(&sb, "--- %s（磁盘上的当前内容）\n", name)
	}
	fmt.Fprintf(&sb, "+++ %s（将要写入的内容）\n", name)

	hunks := buildHunks(ops)
	if len(hunks) == 0 {
		sb.WriteString("（行内容相同，只有行尾换行的差别）")
		return sb.String()
	}

	emitted := 0
	for i, h := range hunks {
		if emitted >= diffMaxOutLines {
			fmt.Fprintf(&sb, "…（还有 %d 处改动未显示；共 %d 行变 %d 行）\n",
				len(hunks)-i, len(oldLines), len(newLines))
			break
		}
		fmt.Fprintf(&sb, "@@ -%d,%d +%d,%d @@\n", h.startOld, h.oldCount, h.startNew, h.newCount)
		for _, op := range h.ops {
			if emitted == diffMaxOutLines {
				fmt.Fprintf(&sb, "…（diff 过长已截断；共 %d 行变 %d 行）\n", len(oldLines), len(newLines))
				break
			}
			sb.WriteString(diffMarker(op.kind) + op.text + "\n")
			emitted++
		}
	}
	if coarse {
		fmt.Fprintf(&sb, "（改动过大，未做最小化：整段旧内容被替换。共 %d 行变 %d 行）\n",
			len(oldLines), len(newLines))
	}
	return strings.TrimRight(sb.String(), "\n")
}

// splitLines 按行切分；结尾换行不产生空的最后一行，空内容视为零行。
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n")
}

type diffOp struct {
	kind     byte // '=' 相同、'-' 只在旧内容里、'+' 只在新内容里
	text     string
	startOld int
	startNew int
}

// diffOps 返回逐行的操作序列；coarse 为 true 表示改动过大，
// 中间区间整体按"删旧加新"处理（仍是合法 diff，只是不最小）。
func diffOps(oldLines, newLines []string) ([]diffOp, bool) {
	prefix := 0
	for prefix < len(oldLines) && prefix < len(newLines) && oldLines[prefix] == newLines[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < len(oldLines)-prefix && suffix < len(newLines)-prefix &&
		oldLines[len(oldLines)-1-suffix] == newLines[len(newLines)-1-suffix] {
		suffix++
	}

	oldMid := oldLines[prefix : len(oldLines)-suffix]
	newMid := newLines[prefix : len(newLines)-suffix]

	ops := make([]diffOp, 0, prefix+len(oldMid)+len(newMid)+suffix)
	for i := 0; i < prefix; i++ {
		ops = append(ops, diffOp{kind: '=', text: oldLines[i]})
	}

	coarse := len(oldMid) > diffMaxRegion || len(newMid) > diffMaxRegion ||
		len(oldMid)*len(newMid) > diffMaxCells
	if coarse {
		for _, l := range oldMid {
			ops = append(ops, diffOp{kind: '-', text: l})
		}
		for _, l := range newMid {
			ops = append(ops, diffOp{kind: '+', text: l})
		}
	} else {
		ops = append(ops, lcsOps(oldMid, newMid)...)
	}

	offset := len(oldLines) - suffix
	for i := 0; i < suffix; i++ {
		ops = append(ops, diffOp{kind: '=', text: oldLines[offset+i]})
	}

	// 标上行号，供 hunk 头部使用。每个 op 记下"它落在 old/new 的第几行"，
	// 纯新增行也要记，否则以新增行开头的 hunk 会拿到 0 号起点。
	o, n := 1, 1
	for i := range ops {
		ops[i].startOld, ops[i].startNew = o, n
		switch ops[i].kind {
		case '=':
			o++
			n++
		case '-':
			o++
		case '+':
			n++
		}
	}
	return ops, coarse
}

// lcsOps 用最长公共子序列求中间区间的最小 diff。
// 行数乘积已在调用处限幅，这里的 DP 规模可控。
func lcsOps(oldMid, newMid []string) []diffOp {
	m, n := len(oldMid), len(newMid)
	table := make([][]int, m+1)
	for i := range table {
		table[i] = make([]int, n+1)
	}
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			if oldMid[i] == newMid[j] {
				table[i][j] = table[i+1][j+1] + 1
			} else {
				table[i][j] = max(table[i+1][j], table[i][j+1])
			}
		}
	}

	var ops []diffOp
	for i, j := 0, 0; i < m || j < n; {
		switch {
		case i < m && j < n && oldMid[i] == newMid[j]:
			ops = append(ops, diffOp{kind: '=', text: oldMid[i]})
			i++
			j++
		// 平手时先删后加，读起来才是"旧行被新行替掉"而不是反过来。
		case j < n && (i == m || table[i][j+1] > table[i+1][j]):
			ops = append(ops, diffOp{kind: '+', text: newMid[j]})
			j++
		default:
			ops = append(ops, diffOp{kind: '-', text: oldMid[i]})
			i++
		}
	}
	return ops
}

type diffHunk struct {
	ops      []diffOp
	startOld int
	startNew int
	oldCount int
	newCount int
}

// buildHunks 把改动行连同前后各 diffContext 行归并成 hunk。
func buildHunks(ops []diffOp) []diffHunk {
	keep := make([]bool, len(ops))
	for i, op := range ops {
		if op.kind == '=' {
			continue
		}
		for j := i - diffContext; j <= i+diffContext; j++ {
			if j >= 0 && j < len(ops) {
				keep[j] = true
			}
		}
	}

	var hunks []diffHunk
	for i := 0; i < len(ops); {
		if !keep[i] {
			i++
			continue
		}
		start := i
		for i < len(ops) && keep[i] {
			i++
		}
		h := diffHunk{ops: ops[start:i], startOld: ops[start].startOld, startNew: ops[start].startNew}
		for _, op := range h.ops {
			if op.kind != '+' {
				h.oldCount++
			}
			if op.kind != '-' {
				h.newCount++
			}
		}
		hunks = append(hunks, h)
	}
	return hunks
}

func diffMarker(kind byte) string {
	switch kind {
	case '-':
		return "-"
	case '+':
		return "+"
	default:
		return " "
	}
}

func containsNUL(s string) bool {
	return strings.IndexByte(s, 0) >= 0
}
