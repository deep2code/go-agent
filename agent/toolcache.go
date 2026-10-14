package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"go-agent/log"
)

// ============================================================================
// ToolCache — 大输出的两级治理
//
// 这层解决的不是"省 token"，而是重复计费：一趟 run_command 或 read_file 可能吐回
// 几十万字节，原样进历史就等于让一次调用吃掉整轮预算，并且此后每一轮都再付一次。
// 所以超限的输出在**进历史之前**就拆成"头 + 指针 + 尾"，全文落到工作目录里，
// 模型真要看全文时按指针 read_file 取回——只有需要它的那一轮才付全文的代价。
//
// L1（Govern）触发在每次工具输出；L2（Agent.Pointerize，在 agent/context.go）
// 触发在每次构建上下文，把历史里剩下的早期大输出也换成同一形式的指针，
// 只是预览更短、门槛更低。两者共用这一个缓存目录。
// ============================================================================

// pointerPhrase 是指针文本的固定标记，同时用来识别"已经是指针了"。
// 改它等于改协议：L2 会认不出已治理过的正文，转而再存一遍、再套一层指针。
const pointerPhrase = "完整输出已存到 "

const (
	l1CacheFrom = 6000 // 超过这个字节数才外置
	l1HeadBytes = 4000
	l1TailBytes = 1500

	l2CacheFrom  = 3000 // L2 的门槛比 L1 低：更早的正文没资格继续占着整段
	l2KeepRecent = 3    // 最近这么多条工具结果保持原样

	// L2 留下的预览也比 L1 短一截，这组数字是推出来的不是调出来的：
	// 头尾之和必须明显小于 l2CacheFrom，否则一条 3KB 的正文切完"头 + 指针 + 尾"反而比原文长，
	// 这一级就等于没存在过。800 字节也够模型认出"那一步读的是哪个文件、大概什么内容"。
	l2HeadBytes = 800
	l2TailBytes = 300
)

// govShape 是一次治理用哪组门槛与头尾长度；from=0 表示无条件换形式。
type govShape struct {
	from, head, tail int
}

var (
	l1Shape = govShape{l1CacheFrom, l1HeadBytes, l1TailBytes}
	l2Shape = govShape{0, l2HeadBytes, l2TailBytes}
)

// cacheLimits 是外置目录的容量闸门。缓存无限增长等于把"上下文治理"换成"磁盘治理"，
// 用户不会希望每次跑完智能体都多出几百兆没人认领的文件。
type cacheLimits struct {
	maxBytes int64
	maxFiles int
}

var defaultCacheLimits = cacheLimits{maxBytes: 64 << 20, maxFiles: 200}

// ToolCache 把工具全文落盘，返回可回读的相对路径。可并发调用（并行工具执行会同时写）。
type ToolCache struct {
	dir     string // 绝对路径，写入用
	display string // 相对工作目录的前缀，指针文本里给模型看的就是它
	limits  cacheLimits
	usable  bool

	mu sync.Mutex
}

// NewToolCache 创建缓存。
//
// dir 或 display 为空表示不启用：治理退化成就地截断，指针里会直说"没能落盘"，
// 而不是给模型一个读不到的路径。display 必须是相对工作目录的形式——
// 模型拿它去 read_file，而 read_file 只认工作目录内的相对路径。
func NewToolCache(dir, display string) *ToolCache {
	return &ToolCache{dir: dir, display: display, limits: defaultCacheLimits, usable: dir != "" && display != ""}
}

// Govern 是 L1 的入口：一次工具输出进历史之前过这里。
//
// 不超限的原样返回。超限的切成头 + 指针 + 尾，全文落盘。
// **必须在内容守卫之后调用**：落盘的那份必须是已经隐去密钥的版本，
// 否则缓存目录就成了仓库里最亮的一个靶子。
func (c *ToolCache) Govern(id, text string) string {
	return c.govern(id, text, l1Shape)
}

// Pointer 是 L2 的形态：判定权在调用方（"这条够旧了"），这里只负责换形式。
// 已经是指针的原样返回，避免一轮轮往上叠。
func (c *ToolCache) Pointer(id, text string) string {
	if strings.Contains(text, pointerPhrase) {
		return text
	}
	return c.govern(id, text, l2Shape)
}

// govern 按 s.from 决定是否治理（from=0 表示无条件换形式），并按 s.head/s.tail 切预览。
func (c *ToolCache) govern(id, text string, s govShape) string {
	if c == nil || len(text) <= s.from {
		return text
	}
	head := snapBack(text, s.head)
	tail := snapForward(text, len(text)-s.tail)
	if len(head)+len(tail) >= len(text) {
		return text // 头尾已经覆盖全文，套指针只会更长
	}

	relPath := ""
	if c.usable {
		p, err := c.store(id, text)
		if err != nil {
			log.Warn("工具输出外置失败，改为就地截断: %v", err)
		} else {
			relPath = p
		}
	}
	return head + "\n" + pointerLine(relPath, len(text), len(head), len(tail)) + "\n" + tail
}

// store 把全文写进缓存目录，返回相对文件名。
//
// 文件名取自 tool_use 的 ID：它唯一、且同一调用重放时名字不变，
// 所以恢复会话后历史里的老指针仍然指得到东西。
func (c *ToolCache) store(id, text string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return "", fmt.Errorf("创建缓存目录失败: %w", err)
	}
	name := safeCacheName(id) + ".txt"
	if err := os.WriteFile(filepath.Join(c.dir, name), []byte(text), 0o644); err != nil {
		return "", fmt.Errorf("写入缓存失败: %w", err)
	}
	if err := c.pruneLocked(); err != nil {
		// 清理失败不毁掉这次治理：全文已经落盘，指针依然有效。
		log.Warn("外置缓存清理失败: %v", err)
	}
	return c.display + "/" + name, nil
}

// pruneLocked 把目录压回容量上限：按修改时间从旧到新删，但始终至少留一份，
// 免得一次超限就把模型正打算回读的那几条删光。调用方须已持有 c.mu。
func (c *ToolCache) pruneLocked() error {
	type cacheFile struct {
		name string
		size int64
		mod  int64
	}

	entries, err := os.ReadDir(c.dir)
	if err != nil {
		return err
	}
	files := make([]cacheFile, 0, len(entries))
	var total int64
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		files = append(files, cacheFile{e.Name(), info.Size(), info.ModTime().UnixNano()})
		total += info.Size()
	}

	if total <= c.limits.maxBytes && len(files) <= c.limits.maxFiles {
		return nil
	}

	sort.Slice(files, func(i, j int) bool { return files[i].mod < files[j].mod })
	for len(files) > 1 && (total > c.limits.maxBytes || len(files) > c.limits.maxFiles) {
		victim := files[0]
		if err := os.Remove(filepath.Join(c.dir, victim.name)); err != nil && !os.IsNotExist(err) {
			return err
		}
		total -= victim.size
		files = files[1:]
	}
	return nil
}

// foldedPointerNote 挂在被折叠掉的治理正文前面，明说两头预览已经不在了；
// 它同时充当"这条已经折过"的标记，免得每轮往上再叠一层。
const foldedPointerNote = "（较早的输出已折叠，只剩下面这行回读指针）"

// foldToPointer 把一段大输出压到只剩指针那一行：路径留着（还能回读），
// 两头预览去掉（那正是占地方的部分）。
// ok=false 表示正文里没有指针，调用方走普通的"已折叠"占位。
func foldToPointer(text string) (string, bool) {
	if strings.HasPrefix(text, foldedPointerNote) {
		return text, true // 已经折过一次，再动只会把路径弄丢
	}
	if !strings.Contains(text, pointerPhrase) {
		return "", false
	}
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, pointerPhrase) {
			return foldedPointerNote + "\n" + line, true
		}
	}
	// 含指针却没有整行的指针：不猜它被谁改过，交回普通折叠。
	return "", false
}

// pointerLine 生成中间那一行。relPath 为空时如实说明没存下来，并给出可操作的下法。
func pointerLine(relPath string, total, head, tail int) string {
	omitted := total - head - tail
	if relPath == "" {
		// 参数名要写对：这里若说成 end_line，模型照做会传一个并不存在的参数，
		// read_file 静默忽略它、按默认行数往下读——比不告诉它怎么缩小范围更糟。
		return fmt.Sprintf("…（中间 %d 字节已省略，且全文未能落盘。要看更多内容请缩小范围重新调用：read_file 用 start_line + max_lines 分段，"+
			"run_command 加过滤或 head/tail，search_code 收紧正则）", omitted)
	}
	return fmt.Sprintf("…（中间 %d 字节已省略。%s%s（全文 %d 字节），需要时用 read_file 读这个路径；"+
		"若文件已被清理，重新调用原工具即可）", omitted, pointerPhrase, relPath, total)
}

// safeCacheName 把 tool_use ID 收成合法文件名。
// ID 来自服务端，不能直接拼进路径——那等于让远端决定写到哪里。
func safeCacheName(id string) string {
	var sb strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			sb.WriteRune(r)
		default:
			sb.WriteRune('_')
		}
	}
	s := sb.String()
	if s == "" {
		return "output"
	}
	if len(s) > 60 {
		s = s[len(s)-60:]
	}
	return s
}

// snapBack 把 n 退到不超过 n 的最大 rune 边界，避免把汉字切成半个。
func snapBack(s string, n int) string {
	if n >= len(s) {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// snapForward 把 n 进到不小于 n 的最小 rune 边界。
func snapForward(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if n >= len(s) {
		return s
	}
	for n < len(s) && !utf8.RuneStart(s[n]) {
		n++
	}
	return s[n:]
}
