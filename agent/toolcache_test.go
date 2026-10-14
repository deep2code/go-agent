package agent

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"go-agent/llm"
	"go-agent/tools"
)

// ============================================================================
// 测试脚手架
// ============================================================================

// cacheIn 造一个指向临时目录的缓存；display 用智能体真实的那套相对前缀，
// 这样断言过的路径就是模型真能拿去 read_file 的路径。
func cacheIn(t *testing.T) *ToolCache {
	t.Helper()
	return NewToolCache(t.TempDir()+"/.go-agent/toolcache", ".go-agent/toolcache")
}

func agentWithCache(t *testing.T, cache *ToolCache) *Agent {
	t.Helper()
	return agentCacheWith(t, cache)
}

// agentCacheWith 装配一个只挂缓存、不带模型的 agent；ts 是它能看到的那几个工具。
func agentCacheWith(t *testing.T, cache *ToolCache, ts ...tools.Tool) *Agent {
	t.Helper()
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(ts...), func(cfg *Config) {
		cfg.Cache = cache
		cfg.Writer = io.Discard
	})
	return a
}

// pointerPath 从治理后的文本里把指针路径抠出来，用于断言全文真的落在那个位置。
func pointerPath(t *testing.T, governed string) string {
	t.Helper()
	i := strings.Index(governed, pointerPhrase)
	if i < 0 {
		t.Fatalf("没找到指针标记 %q in %q", pointerPhrase, governed)
	}
	rest := governed[i+len(pointerPhrase):]
	end := strings.IndexAny(rest, "（ ")
	if end < 0 {
		end = len(rest)
	}
	return rest[:end]
}

// ============================================================================
// L1：每次工具输出的入口治理
// ============================================================================

func TestGovernPassesSmallOutputThrough(t *testing.T) {
	c := cacheIn(t)
	text := strings.Repeat("a", l1CacheFrom) // 恰好等于门槛：不超过就不动
	if got := c.Govern("toolu_1", text); got != text {
		t.Error("未超限的输出被改动了")
	}
	if _, err := os.Stat(c.dir); err == nil {
		t.Error("一次都没超限，却已经建出了缓存目录")
	}
}

func TestGovernTruncatesAndSpills(t *testing.T) {
	c := cacheIn(t)
	full := strings.Repeat("x", 20000)

	got := c.Govern("toolu_42", full)

	if len(got) >= len(full) {
		t.Fatalf("治理后长度 %d，原文 %d：没省下来", len(got), len(full))
	}
	if !strings.HasPrefix(got, strings.Repeat("x", 100)) {
		t.Error("头部内容被丢了")
	}
	if !strings.HasSuffix(got, strings.Repeat("x", 100)) {
		t.Error("尾部内容被丢了")
	}

	// 指针必须指得到一个真实文件，且文件里是完整原文
	rel := pointerPath(t, got)
	if rel != ".go-agent/toolcache/toolu_42.txt" {
		t.Errorf("指针路径 = %q，期望工作目录内的相对路径", rel)
	}
	if strings.Contains(rel, "..") || strings.HasPrefix(rel, "/") {
		t.Errorf("指针路径不安全，read_file 会拒：%q", rel)
	}
	saved, err := os.ReadFile(c.dir + "/" + strings.TrimPrefix(rel, ".go-agent/toolcache/"))
	if err != nil {
		t.Fatalf("指针指向的文件读不到: %v", err)
	}
	if string(saved) != full {
		t.Errorf("落盘的全文与原文不一致：差 %d 字节", len(saved)-len(full))
	}
}

// 中文注释是大头，切在 rune 中间会让模型看到乱码，也可能让 provider 直接报错。
func TestGovernSplitsOnRuneBoundary(t *testing.T) {
	c := cacheIn(t)
	full := strings.Repeat("代码智能体", 3000) // 15000 字节，每字 3 字节
	if !utf8.ValidString(full) {
		t.Fatal("构造用例失败")
	}
	// 用例成立的前提：原始字节切点落在汉字中间。常量若改成正好整除，这个测试就什么都没证明。
	if utf8.RuneStart(full[l1HeadBytes]) {
		t.Fatalf("用例前提失效：第 %d 字节正好是 rune 边界", l1HeadBytes)
	}

	got := c.Govern("toolu_cn", full)
	if !utf8.ValidString(got) {
		t.Error("治理后的文本里有被切半的汉字")
	}
	head := snapBack(full, l1HeadBytes)
	if !strings.HasPrefix(got, head) {
		t.Error("头部没落在 rune 边界上")
	}
}

// 头尾加起来就覆盖全文时，套指针反而更长——这种情况必须原样返回。
// 只有 L2 会走到这条分支：它的门槛（3000）比 L1 的头尾之和（5500）低。
func TestPointerSkipsWhenHeadTailCoverEverything(t *testing.T) {
	c := cacheIn(t)
	text := strings.Repeat("y", l2HeadBytes+l2TailBytes-1) // 头尾正好盖住
	got := c.Pointer("toolu_cover", text)
	if got != text {
		t.Errorf("头尾已覆盖全文，却仍被改写（%d → %d 字节）", len(text), len(got))
	}
}

// 落盘失败（目录不可写）不能让整个任务挂掉，也不能给出一个读不到的路径。
func TestGovernWithoutCacheDirStillTruncates(t *testing.T) {
	c := NewToolCache("", "")
	text := strings.Repeat("z", 20000)
	got := c.Govern("toolu_x", text)

	if len(got) >= len(text) {
		t.Fatal("缓存不可用时没退化成就地截断")
	}
	if strings.Contains(got, pointerPhrase) {
		t.Error("落盘失败却给出了指针：模型会白烧一轮去读一个不存在的路径")
	}
	if !strings.Contains(got, "未能落盘") {
		t.Errorf("指针缺失时应说明原因并给出下法，实际：%s", got[len(got)/2:])
	}
}

func TestNilCacheIsSafe(t *testing.T) {
	var c *ToolCache
	text := strings.Repeat("q", 100)
	if got := c.Govern("toolu_1", text); got != text {
		t.Error("nil 缓存应当原样返回")
	}
	if got := c.Pointer("toolu_1", text); got != text {
		t.Error("nil 缓存的 Pointer 应当原样返回")
	}
}

// 文件名取自远端给的 tool_use ID，那不能当路径直接拼。
func TestSafeCacheNameSanitizes(t *testing.T) {
	for _, id := range []string{"../../etc/passwd", "a/b\\c", "中文 id", ""} {
		name := safeCacheName(id)
		if strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
			t.Errorf("safeCacheName(%q) = %q，还能越界", id, name)
		}
		if name == "" {
			t.Errorf("safeCacheName(%q) 返回空串，会写出一个目录路径", id)
		}
	}
	long := strings.Repeat("a", 200)
	if got := safeCacheName(long); len(got) != 60 {
		t.Errorf("超长 ID 应截到 60，实际 %d", len(got))
	}
}

// 容量闸门：缓存无限增长只是把上下文问题换成磁盘问题。
func TestCachePrunesByFileCount(t *testing.T) {
	c := NewToolCache(t.TempDir(), ".go-agent/toolcache")
	c.limits = cacheLimits{maxBytes: 1 << 30, maxFiles: 3}

	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("toolu_%d_%s", i, strings.Repeat("p", i)) // 名字各不相同
		if _, err := c.store(id, strings.Repeat("m", 10)); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) > 3 {
		t.Errorf("清理后剩 %d 个文件，上限 3", len(entries))
	}
}

func TestCachePrunesByBytesButKeepsOne(t *testing.T) {
	c := NewToolCache(t.TempDir(), ".go-agent/toolcache")
	c.limits = cacheLimits{maxBytes: 10, maxFiles: 100}

	// 每条 100 字节，容量只有 10：必须删到超不出，但最后一份要留着，
	// 否则模型正要回读的那条会指向一个已被删掉的文件。
	for i := 0; i < 4; i++ {
		if _, err := c.store(fmt.Sprintf("toolu_%d", i), strings.Repeat("k", 100)); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Error("清理把唯一一份也删了：指针全部悬空")
	}
}

// ============================================================================
// L2：构建上下文时的指针替代
// ============================================================================

// historyWithResults 造一串交替的 assistant/tool_result 往返。
func historyWithResults(n int, size int) []llm.Message {
	out := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: "text", Text: "任务"}}}}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("toolu_%d", i)
		out = append(out,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{toolUse(id, "read_file")}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult(id, strings.Repeat("r", size))}},
		)
	}
	return out
}

func TestPointerizeReplacesOlderResultsOnly(t *testing.T) {
	a := agentWithCache(t, cacheIn(t))
	const n = 6
	hist := historyWithResults(n, 4000)

	freed := a.Pointerize(hist)
	if freed <= 0 {
		t.Fatal("一条都没换掉")
	}

	var seen int
	for _, m := range hist {
		for _, b := range m.Content {
			if b.Type != "tool_result" {
				continue
			}
			seen++
			text := toolResultText(b)
			// 从后往前数：最近 l2KeepRecent 条必须保持原样
			fromEnd := n - seen
			if fromEnd >= 0 && fromEnd < l2KeepRecent {
				if strings.Contains(text, pointerPhrase) {
					t.Errorf("最近第 %d 条被换成了指针：那几轮正是下一步动作的依据", fromEnd)
				}
				continue
			}
			if !strings.Contains(text, pointerPhrase) {
				t.Errorf("较早的第 %d 条没换成指针，还占着 %d 字节", seen, len(text))
			}
		}
	}
}

// 第 5 条约定（tool_use/tool_result 配对）是这套历史能发出去的前提。
// L2 只许替换正文，动 ID 或动块就会让 RepairHistory 判出孤儿。
func TestPointerizeKeepsPairingIntact(t *testing.T) {
	a := agentWithCache(t, cacheIn(t))
	hist := historyWithResults(6, 4000)

	before := len(resultIDsAll(hist))
	a.Pointerize(hist)
	repaired := RepairHistory(hist)

	if got := len(resultIDsAll(repaired)); got != before {
		t.Errorf("tool_result 数量从 %d 变成 %d：配对被破坏了", before, got)
	}
	if len(toolUseIDsAll(repaired)) != before {
		t.Error("tool_use 数量变了：L2 不该碰 assistant 的调用块")
	}
	for i := range hist {
		if hist[i].Role != repaired[i].Role {
			t.Fatalf("第 %d 条的角色变了：交替结构被破坏", i)
		}
	}
}

// 一轮一轮叠上去的嵌套指针比原文更难读，而且第二次治理纯属白费。
func TestPointerizeIsIdempotent(t *testing.T) {
	a := agentWithCache(t, cacheIn(t))
	hist := historyWithResults(6, 4000)

	a.Pointerize(hist)
	first := estimatePointerCounts(hist)
	if again := a.Pointerize(hist); again != 0 {
		t.Errorf("第二次又省下 %d 字节：说明在已有指针上再套了一层", again)
	}
	if second := estimatePointerCounts(hist); second != first {
		t.Errorf("指针数量从 %d 涨到 %d", first, second)
	}
	for _, m := range hist {
		for _, b := range m.Content {
			text := toolResultText(b)
			if strings.Count(text, pointerPhrase) > 1 {
				t.Fatalf("一条正文里出现了两枚指针：%q", text)
			}
		}
	}
}

// 小结果不值得换：换了之后指针文本可能比原文还长。
func TestPointerizeLeavesSmallResults(t *testing.T) {
	a := agentWithCache(t, cacheIn(t))
	hist := historyWithResults(6, l2CacheFrom-1)
	if freed := a.Pointerize(hist); freed != 0 {
		t.Errorf("未超限的旧结果被换掉了，省下 %d 字节", freed)
	}
}

// 缓存不可用（目录没配）时 L2 退化成就地截断：照样省上下文，只是再也取不回来。
// 退化必须诚实——不能给出一条模型拿去 read_file 会扑空的路径。
func TestPointerizeWithoutCache(t *testing.T) {
	a := agentCacheWith(t, NewToolCache("", ""))
	hist := historyWithResults(6, 4000)
	before := EstimateTokens(hist)
	freed := a.Pointerize(hist)
	if freed <= 0 {
		t.Errorf("就地截断同样省上下文，却报告省下 %d 字节", freed)
	}

	var replaced int
	for _, m := range hist {
		for _, b := range m.Content {
			text := toolResultText(b)
			if !strings.Contains(text, "未能落盘") {
				continue
			}
			replaced++
			if strings.Contains(text, pointerPhrase) {
				t.Error("落盘失败却给出了指针路径：模型会白烧一轮去读一个不存在的文件")
			}
		}
	}
	if replaced == 0 {
		t.Fatal("旧结果没有被就地截断")
	}
	if EstimateTokens(hist) >= before {
		t.Errorf("截断后估算没变小：%d -> %d", before, EstimateTokens(hist))
	}
}

// L2 必须排在估算之前：否则"要不要折叠"已经按换指针之前的体积判完了，
// 这一级永远轮不到发言。这里用一条刚好卡在预算边上的历史把顺序钉住。
func TestPointerizeRunsBeforeBudgetCheck(t *testing.T) {
	dir := t.TempDir()
	cache := NewToolCache(dir+"/.go-agent/toolcache", ".go-agent/toolcache")

	a, _ := newAgent(t, &scriptedProvider{}, registryWith(&fakeTool{name: "read_file"}), func(cfg *Config) {
		cfg.Cache = cache
		cfg.ContextWindow = 20000 // 预算很小，不治理就一定超
		cfg.MaxTokens = 8000
		cfg.Writer = io.Discard
	})

	// 8 条 4000 字节的旧结果：不换成指针远超预算
	a.history = historyWithResults(8, 4000)

	freed, err := a.prepareForSend(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if freed != 0 {
		t.Errorf("动用了折叠（省下 %d token）：L2 显然没在估算之前生效", freed)
	}
	pointers := estimatePointerCounts(a.history)
	if pointers == 0 {
		t.Fatal("prepareForSend 没做指针替代")
	}
}

// L1 在真实执行路径上：工具返回超限输出时，进历史的必须是治理后的版本。
func TestToolResultIsGovernedOnEntry(t *testing.T) {
	big := &fakeTool{name: "read_file", result: strings.Repeat("g", 30000)}
	a := agentCacheWith(t, cacheIn(t), big)

	blocks, err := a.executeToolCalls(context.Background(),
		[]llm.ContentBlock{toolUse("toolu_big", "read_file")}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("结果数 = %d", len(blocks))
	}
	text := toolResultText(blocks[0])
	if len(text) > l1CacheFrom+500 {
		t.Errorf("超限输出原样进了历史：%d 字节", len(text))
	}
	if !strings.Contains(text, pointerPhrase) {
		t.Error("进历史的那条没有指针")
	}
}

// 错误结果同样可能超长（编译日志），入口治理不放过它。
func TestErrorResultIsGovernedToo(t *testing.T) {
	big := &fakeTool{name: "run_command", err: fmt.Errorf("%s", strings.Repeat("boom ", 5000))}
	a := agentCacheWith(t, cacheIn(t), big)

	blocks, err := a.executeToolCalls(context.Background(),
		[]llm.ContentBlock{toolUse("toolu_err", "read_file")}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	text := toolResultText(blocks[0])
	if len(text) > l1CacheFrom+500 {
		t.Errorf("超长的错误输出原样进了历史：%d 字节", len(text))
	}
	if blocks[0].IsError == nil || !*blocks[0].IsError {
		t.Error("治理不该把 is_error 弄丢：模型靠它判断这一步失败了")
	}
}

// 降级文本里教模型"怎么缩小范围"，点名的参数就必须是真的。
// 参数名一旦改名，模型照着传会被静默忽略，还以为自己限定了范围——比不提更糟。
func TestPointerAdviceNamesRealParams(t *testing.T) {
	c := NewToolCache("", "")
	got := c.Govern("toolu_adv", strings.Repeat("v", 20000))
	if !strings.Contains(got, "start_line") || !strings.Contains(got, "max_lines") {
		t.Fatalf("降级文本没给出可操作的缩小范围办法：%s", got[len(got)/2:])
	}

	read := tools.NewReadFileTool(t.TempDir(), tools.NewFileState()).Definition()
	for _, name := range []string{"start_line", "max_lines"} {
		if _, ok := read.InputSchema.Properties[name]; !ok {
			t.Errorf("指针文本叫了 read_file 的不存在参数 %q", name)
		}
	}
}

// 并行只读执行会同时治理多条输出，落盘和清理都在同一个目录上。
func TestCacheIsConcurrencySafe(t *testing.T) {
	c := cacheIn(t)
	// 每条全文 20000 字节：按上限 64MB 永远轮不到清理，这里把闸门调到 30KB，
	// 让"边写边删"真的发生，才压得出目录状态上的竞争。
	c.limits = cacheLimits{maxBytes: 30000, maxFiles: 5}

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got := c.Govern(fmt.Sprintf("toolu_c%d", i), strings.Repeat("z", 20000))
			if !strings.Contains(got, pointerPhrase) {
				t.Errorf("第 %d 条没产出指针", i)
			}
		}(i)
	}
	wg.Wait()

	entries, err := os.ReadDir(c.dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) > 5 {
		t.Errorf("并发下清理没守住上限：剩 %d 个", len(entries))
	}
}

// ============================================================================
// 折叠与指针的交界
// ============================================================================

// CompactHistory 第一级要把旧正文换成占位说明。对已经治理过的正文，一句"已折叠"会连
// 回读路径一起丢掉——那一轮就再也取不回原文了。所以这里只许砍掉两头预览。
func TestCompactKeepsPointerPath(t *testing.T) {
	c := cacheIn(t)
	governed := c.Govern("toolu_x", strings.Repeat("w", 20000))
	if !strings.Contains(governed, pointerPhrase) {
		t.Fatal("用例前提失效：入口治理没产出指针")
	}

	messages := historyWithResults(6, len(governed))
	for i := range messages {
		for j, b := range messages[i].Content {
			if b.Type == "tool_result" {
				messages[i].Content[j].Content = governed
			}
		}
	}

	// 预算卡在"砍掉 3 条预览就够、还不必丢整轮"的位置：一旦走到第二级，消息下标就变了，
	// 这个测试要钉的是折叠形式，不是丢弃策略。
	budget := EstimateTokens(messages) * 6 / 10
	compacted, freed := CompactHistory(context.Background(), messages, budget, nil)
	if freed <= 0 {
		t.Fatal("没触发折叠")
	}

	var folded, intact int
	for _, m := range compacted {
		for _, b := range m.Content {
			if b.Type != "tool_result" {
				continue
			}
			text := toolResultText(b)
			switch {
			case !strings.Contains(text, pointerPhrase):
				t.Errorf("折叠把回读路径弄丢了：%q", text)
			case strings.HasPrefix(text, foldedPointerNote):
				folded++
				if len(text) > 400 {
					t.Errorf("折叠没砍掉预览，还剩 %d 字节", len(text))
				}
			default:
				intact++
			}
		}
	}
	if folded != 3 || intact != 3 {
		t.Errorf("折叠了 %d 条、原样 %d 条，期望各 3 条", folded, intact)
	}

	// 回到预算内之后不该再动第二次：反复套 foldedPointerNote 会把指针越埋越深。
	again, freed2 := CompactHistory(context.Background(), compacted, budget, nil)
	if freed2 != 0 {
		t.Error("已经够小了还在折叠")
	}
	if toolResultText(again[2].Content[0]) != toolResultText(compacted[2].Content[0]) {
		t.Error("同一条被折了两次")
	}
}

// 普通正文（没治理过、没有指针）走原来的那句占位说明，不能被指针分支误伤。
func TestCompactFoldsPlainResult(t *testing.T) {
	messages := historyWithResults(6, 5000) // 低于 l1CacheFrom：入口治理不参与
	budget := EstimateTokens(messages) * 6 / 10

	compacted, freed := CompactHistory(context.Background(), messages, budget, nil)
	if freed <= 0 {
		t.Fatal("没触发折叠")
	}
	text := toolResultText(compacted[2].Content[0])
	if strings.Contains(text, pointerPhrase) {
		t.Errorf("普通正文被安上了指针，那个路径下根本没有文件：%q", text)
	}
	if !strings.Contains(text, "已折叠") {
		t.Errorf("旧正文没被折叠：%q", text)
	}
}

// ============================================================================
// 辅助
// ============================================================================

func estimatePointerCounts(messages []llm.Message) int {
	n := 0
	for _, m := range messages {
		for _, b := range m.Content {
			if strings.Contains(toolResultText(b), pointerPhrase) {
				n++
			}
		}
	}
	return n
}

func resultIDsAll(messages []llm.Message) []string {
	var out []string
	for _, m := range messages {
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				out = append(out, b.ToolUseID)
			}
		}
	}
	return out
}

func toolUseIDsAll(messages []llm.Message) []string {
	var out []string
	for _, m := range messages {
		for _, b := range m.Content {
			if b.Type == "tool_use" {
				out = append(out, b.ID)
			}
		}
	}
	return out
}
