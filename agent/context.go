package agent

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"go-agent/llm"
	"go-agent/log"
	"go-agent/tools"
)

// ============================================================================
// Token 估算
// ============================================================================

// EstimateTokens 粗估一段消息的 token 数。
//
// 不求精确，只求别在真正超窗之前毫无反应：估算偏保守（宁多算不少算），
// 到阈值时提前折叠历史，比让 API 返回 400 再补救划算。
func EstimateTokens(messages []llm.Message) int {
	total := 0
	for _, m := range messages {
		for _, b := range m.Content {
			total += estimateTextTokens(b.Text)
			total += estimateTextTokens(b.Thinking)
			total += estimateTextTokens(toolResultText(b))
			for k, v := range b.Input {
				total += estimateTextTokens(k)
				total += estimateTextTokens(fmt.Sprintf("%v", v))
			}
			total += 4 // 每个块的包装开销
		}
	}
	return total
}

// EstimateTools 粗估工具定义占用的 token。
func EstimateTools(tools []llm.ToolDef) int {
	total := 0
	for _, t := range tools {
		total += estimateTextTokens(t.Name) + estimateTextTokens(t.Description)
		for name, prop := range t.InputSchema.Properties {
			total += estimateTextTokens(name) + estimateTextTokens(prop.Description) + 6
		}
		for _, req := range t.InputSchema.Required {
			total += estimateTextTokens(req) + 2
		}
	}
	return total
}

// estimateTextTokens 按字符类别折算：非 ASCII（中文为主）约 1 token/字，
// ASCII 约 4 token/词。
func estimateTextTokens(s string) int {
	if s == "" {
		return 0
	}
	var asciiRunes, wideRunes int
	for _, r := range s {
		switch {
		case r > 0x2E7F:
			wideRunes++
		case unicode.IsSpace(r):
			asciiRunes++
		default:
			asciiRunes++
		}
	}
	return wideRunes + asciiRunes/4 + 1
}

// toolResultText 取出 tool_result 块的文本内容。
func toolResultText(b llm.ContentBlock) string {
	if b.Type != "tool_result" {
		return ""
	}
	switch v := b.Content.(type) {
	case string:
		return v
	default:
		return fmt.Sprintf("%v", v)
	}
}

// ============================================================================
// 历史修复
// ============================================================================

// RepairHistory 把历史整理成 API 能接受的形态。
//
// 需要修的场景很具体：任务被 Ctrl+C 打断、LLM 调用在工具执行前失败、
// 输出被 max_tokens 截断——这些都会留下带 tool_use 却没有配对 tool_result
// 的 assistant 消息。带着这种历史再发一次请求，服务端会直接拒绝，
// 于是用户只剩下 /clear 一条路。
func RepairHistory(messages []llm.Message) []llm.Message {
	var out []llm.Message
	var pending []string // 最近一条 assistant 里等待结果的 tool_use id

	push := func(role llm.Role, content []llm.ContentBlock) {
		if len(content) == 0 {
			return
		}
		if len(out) > 0 && out[len(out)-1].Role == role {
			// 同角色必须合并，user/assistant 交替是 API 的硬性要求
			out[len(out)-1].Content = append(out[len(out)-1].Content, content...)
			return
		}
		out = append(out, llm.Message{Role: role, Content: content})
	}

	flushPending := func() {
		if len(pending) == 0 {
			return
		}
		push(llm.RoleUser, missingResults(pending))
		pending = nil
	}

	for _, m := range messages {
		content := normalizeBlocks(m.Content)
		if len(content) == 0 {
			continue
		}

		if m.Role == llm.RoleAssistant {
			flushPending()
			push(llm.RoleAssistant, content)
			pending = toolUseIDs(content)
			continue
		}

		if len(pending) == 0 {
			// 没有对应的 tool_use，这里的 tool_result 是孤儿，只留文本
			var texts []llm.ContentBlock
			for _, b := range content {
				if b.Type != "tool_result" {
					texts = append(texts, b)
				}
			}
			push(llm.RoleUser, texts)
			continue
		}

		kept := keepMatchedResults(content, pending)
		if missing := subtractIDs(pending, resultIDs(kept)); len(missing) > 0 {
			kept = append(kept, missingResults(missing)...)
		}
		pending = nil
		push(llm.RoleUser, kept)
	}

	flushPending()

	// 首条必须是 user
	for len(out) > 0 && out[0].Role != llm.RoleUser {
		out = out[1:]
	}

	return out
}

// normalizeBlocks 丢掉空块和被截断的 tool_use。
func normalizeBlocks(blocks []llm.ContentBlock) []llm.ContentBlock {
	var out []llm.ContentBlock
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if strings.TrimSpace(b.Text) == "" {
				continue
			}
		case "tool_use":
			if b.Partial {
				// 参数 JSON 只到了一半，执行它等于执行一个猜测
				continue
			}
			if b.ID == "" || b.Name == "" {
				continue
			}
		case "tool_result":
			if b.ToolUseID == "" {
				continue
			}
		case "thinking":
			if b.Thinking == "" {
				continue
			}
		default:
			continue
		}
		out = append(out, b)
	}
	return out
}

// HasPartialToolUse 报告响应里是否含被截断的工具调用。
func HasPartialToolUse(blocks []llm.ContentBlock) bool {
	for _, b := range blocks {
		if b.Type == "tool_use" && b.Partial {
			return true
		}
	}
	return false
}

// missingResults 为丢失结果的 tool_use 生成占位 tool_result。
func missingResults(ids []string) []llm.ContentBlock {
	out := make([]llm.ContentBlock, 0, len(ids))
	for _, id := range ids {
		isErr := true
		out = append(out, llm.ContentBlock{
			Type:      "tool_result",
			ToolUseID: id,
			Content:   "该工具调用没有返回结果（任务被中断、调用失败或输出被截断），请重新发起。",
			IsError:   &isErr,
		})
	}
	return out
}

func toolUseIDs(blocks []llm.ContentBlock) []string {
	var ids []string
	for _, b := range blocks {
		if b.Type == "tool_use" {
			ids = append(ids, b.ID)
		}
	}
	return ids
}

func resultIDs(blocks []llm.ContentBlock) []string {
	var ids []string
	for _, b := range blocks {
		if b.Type == "tool_result" {
			ids = append(ids, b.ToolUseID)
		}
	}
	return ids
}

func subtractIDs(all, minus []string) []string {
	drop := make(map[string]bool, len(minus))
	for _, id := range minus {
		drop[id] = true
	}
	var out []string
	for _, id := range all {
		if !drop[id] {
			out = append(out, id)
		}
	}
	return out
}

// keepMatchedResults 只保留 id 在 want 里的 tool_result，其余内容块原样保留。
func keepMatchedResults(blocks []llm.ContentBlock, want []string) []llm.ContentBlock {
	keep := make(map[string]bool, len(want))
	for _, id := range want {
		keep[id] = true
	}
	var out []llm.ContentBlock
	for _, b := range blocks {
		if b.Type == "tool_result" && !keep[b.ToolUseID] {
			continue
		}
		out = append(out, b)
	}
	return out
}

// ============================================================================
// 历史折叠
// ============================================================================

// keepRecentResults 折叠时保留原样的最近工具结果条数。
const keepRecentResults = 3

// Pointerize 是 L2：把历史里**较早的**大段工具输出换成指针，返回省下的字节数。
//
// 触发点在每次构建上下文时、估算之前——它比折叠温和得多：
// 只替换 tool_result 的正文，不动 tool_use、不动 ToolUseID、不动消息顺序，
// 所以配对结构和角色交替都不受影响，RepairHistory 不需要再介入。
//
// 为什么要单独有这一级：L1 只拦得住"单次就超限"的输出，
// 而真正压垮预算的常常是十几条各自不算大、加起来很可观的 read_file 结果。
// 它们按 L1 的门槛原样进了历史，之后每一轮都被重复发送。
// 这里用更低的门槛（l2CacheFrom）和更短的预览（l2HeadBytes/l2TailBytes）把早期的换成
// "头 + 指针 + 尾"，模型要看全文时按指针回读，一轮就只为需要它的那次调用付费。
//
// 最近 l2KeepRecent 条保持原样：那几轮通常正是模型下一步动作的依据，
// 换掉等于逼它再调一次同样的工具，省下的 token 又花回去了。
func (a *Agent) Pointerize(messages []llm.Message) int {
	if a.cache == nil {
		return 0
	}
	total := 0
	seen := 0
	// 从后往前数，最近几条不动
	for i := len(messages) - 1; i >= 0; i-- {
		for j, b := range messages[i].Content {
			if b.Type != "tool_result" {
				continue
			}
			seen++
			if seen <= l2KeepRecent {
				continue
			}
			text := toolResultText(b)
			if len(text) < l2CacheFrom {
				continue
			}
			trimmed := a.cache.Pointer(b.ToolUseID, text)
			if len(trimmed) >= len(text) {
				continue
			}
			total += len(text) - len(trimmed)
			messages[i].Content[j].Content = trimmed
		}
	}
	return total
}

// l2ParamFrom 是 tool_use 大正文参数的瘦身门槛：低于它的参数不值得动，
// 换掉省不下几个字节，却会让那次改动在历史里更难追踪。
const l2ParamFrom = 1500

// SlimBodyParams 把历史里**较早的、已经成功执行的**写入调用参数中的大正文换成一行说明，
// 返回省下的字节数。
//
// 这一级治的是 Pointerize 管不到的那一半：Pointerize 换的是 tool_result 的正文，
// 而 write_file 那次调用里最占地方的往往是 tool_use 的 content 参数——几千字节的文件全文
// 由模型亲手写出来，写完就躺在磁盘上了，却在此后每一轮都被重复发送。
//
// 四道门，缺一不可：
//  1. 工具自己声明了这个参数是大正文（tools.BodyParamTool）——没声明就一律不动，
//     run_command 的 command 再长也不能换，那是"执行过什么"的唯一记录；
//  2. 配对的 tool_result 存在且不是错误：写失败时参数是模型下一轮重试的唯一依据，
//     而且审批门还在等确认时 diff 要从参数里生成；
//  3. 不属于最近 l2KeepRecent 次调用：那几轮正是模型下一步动作的参照；
//  4. 参数值超过 l2ParamFrom。
//
// 换掉之后不另存一份到 .go-agent/toolcache：全文就在 file_path 指向的文件里，
// 再抄一份到缓存目录只是把同一份内容落两次盘，还多出一个可能被翻看的副本。
func (a *Agent) SlimBodyParams(messages []llm.Message) int {
	statuses := resultStatuses(messages)

	total := 0
	seen := 0
	// 从后往前数，最近几次调用不动
	for i := len(messages) - 1; i >= 0; i-- {
		for j := len(messages[i].Content) - 1; j >= 0; j-- {
			b := &messages[i].Content[j]
			if b.Type != "tool_use" {
				continue
			}
			seen++
			if seen <= l2KeepRecent {
				continue
			}
			tool, ok := a.registry.Get(b.Name)
			if !ok {
				continue
			}
			names := tools.BodyParamNames(tool)
			if len(names) == 0 || b.Input == nil {
				continue
			}
			res, ok := statuses[b.ID]
			if !ok || res.isError {
				continue
			}
			replaced := make(map[string]string)
			for _, name := range names {
				v, ok := b.Input[name].(string)
				if !ok || len(v) < l2ParamFrom {
					continue
				}
				replaced[name] = paramNote(b.Input, name, v)
			}
			if len(replaced) == 0 {
				continue
			}
			// 换整个 map 而不是往原 map 上写：Input 是引用类型，
			// 历史、会话快照、子 agent 可能共用同一份，就地改会让别人看到变了的内容。
			fresh := make(map[string]any, len(b.Input))
			for k, v := range b.Input {
				fresh[k] = v
			}
			for k, v := range replaced {
				total += len(fresh[k].(string)) - len(v)
				fresh[k] = v
			}
			b.Input = fresh
		}
	}
	return total
}

// paramNote 生成替换大正文参数的那句话：说清省掉了什么、去哪儿能拿回来。
func paramNote(input map[string]any, name, value string) string {
	path, _ := input["file_path"].(string)
	if path == "" {
		return fmt.Sprintf("（调用已成功，参数 %s 的 %d 字节正文不再重复发送；需要时重新发起同样的调用）",
			name, len(value))
	}
	return fmt.Sprintf("（调用已成功，参数 %s 的 %d 字节正文已落盘在 %s，不再重复发送；需要时用 read_file 读它）",
		name, len(value), path)
}

// resultOutcome 记录一个 tool_use 有没有拿到成功结果。
type resultOutcome struct {
	isError bool
}

// resultStatuses 按 tool_use ID 收集所有 tool_result 的成败。
func resultStatuses(messages []llm.Message) map[string]resultOutcome {
	out := make(map[string]resultOutcome)
	for _, m := range messages {
		for _, b := range m.Content {
			if b.Type != "tool_result" {
				continue
			}
			out[b.ToolUseID] = resultOutcome{isError: b.IsError != nil && *b.IsError}
		}
	}
	return out
}

// dupMinBytes 是去重的最小结果长度：比它短的调用不值得省略，
// 省不下几个字节，却多让模型背一个"这次怎么没返回内容"的疑问。
const dupMinBytes = 800

// duplicateNotice 检查这次调用的结果是否和历史里某一次**同名同参**调用的结果逐字相同，
// 是则返回一句省略说明，否则返回空串。空串意味着"照常把全文发出去"。
//
// 治的是反复读同一个文件：一个任务里模型常为不同目的把同一份源码读上三四遍，
// 每一遍都是几千字节原样进历史、此后再发三遍。
//
// 判定只用"逐字相等"这一条，但这一条同时兜住了三种风险，所以不需要额外的闸门：
//   - 文件被改过：新的 read_file 输出必然不同，照常发全文；
//   - 历史里那一份已被 L2 指针化或被折叠成占位：它的文本是"头 + 指针 + 尾"，
//     和全文不可能相等，所以不会把内容指向一份其实看不见的历史；
//   - 上一次是错误结果：带 is_error，不参与比较。
//
// 因此它是**自愈**的：那一份一旦在后续某轮被折叠掉，下一次同样的调用就会重新发送全文。
func (a *Agent) duplicateNotice(name string, input map[string]any, result string) string {
	if len(result) < dupMinBytes {
		return ""
	}
	// 只给只读工具去重：写类工具的结果是"改动已经发生"的凭据，哪怕内容雷同也要逐条留在历史里。
	tool, ok := a.registry.Get(name)
	if !ok || tool.Mutating() {
		return ""
	}
	target := name + "\x00" + argKey(input)

	calls := make(map[string]string) // tool_use ID -> 工具名 + 规范化参数
	for _, m := range a.history {
		for _, b := range m.Content {
			if b.Type == "tool_use" {
				calls[b.ID] = b.Name + "\x00" + argKey(b.Input)
			}
		}
	}
	for _, m := range a.history {
		for _, b := range m.Content {
			if b.Type != "tool_result" || calls[b.ToolUseID] != target {
				continue
			}
			if b.IsError != nil && *b.IsError {
				continue
			}
			if toolResultText(b) != result {
				continue
			}
			return dupNoticeText(name, input)
		}
	}
	return ""
}

// dupNoticeText 给出省略说明，并把回读路径说清楚。
func dupNoticeText(name string, input map[string]any) string {
	where := "同名同参的更早一次调用"
	if path, _ := input["file_path"].(string); path != "" {
		where = fmt.Sprintf("更早一次对 %s 的调用", path)
	}
	return fmt.Sprintf("（本次 %s 的结果与%s逐字相同，已省略不再重复发送；那一份原文就在上方历史里，"+
		"需要时回读它或重新调用本工具）", name, where)
}

// argKey 把参数规范化成可比较的字符串：键按字母序，值全量保留。
//
// 值不能截断——两个只在截断点之后才不同的参数会被判成同一个，
// 那等于把两份不同的文件内容当成一份，省略掉的就是真数据。
func argKey(input map[string]any) string {
	if len(input) == 0 {
		return ""
	}
	keys := make([]string, 0, len(input))
	for k := range input {
		keys = append(keys, k)
	}
	sortStrings(keys)
	var sb strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&sb, "%s=%v\x00", k, input[k])
	}
	return sb.String()
}

// Summarizer 把即将被丢弃的一段历史压缩成可读摘要。
// 为 nil 时第二级折叠退化成"整轮丢弃"，不留摘要。
type Summarizer func(ctx context.Context, messages []llm.Message) (string, error)

// CompactHistory 在历史超出预算时压缩它，返回压缩后的历史和省下的 token 数。
//
// 两步走，先温和后激烈：
//  1. 把较早的工具输出换成一行占位说明——模型通常只需要知道"那一步查过了"，
//     不需要重读整份文件内容；对话结构完全不变。
//     已经外置过的正文（第 14 条）不套占位说明，只砍掉头尾预览、留下指针那一行：
//     一句"已折叠"会把回读路径一起丢掉，那一轮就再也取不回原文了。
//  2. 还不够就折叠最早的往返（assistant + 其工具结果一起）。给了 summarize
//     就把这些往返压成一段摘要插在任务后面，否则直接丢弃。
//     无论哪种，都永远保留第一条用户任务和最近 keepRecentRounds 轮。
//
// summarize 会发起一次额外的模型调用，只在第一段确实要丢时才触发，不做无谓调用。
func CompactHistory(ctx context.Context, messages []llm.Message, budget int, summarize Summarizer) (compacted []llm.Message, freed int) {
	if ctx == nil {
		ctx = context.Background()
	}
	before := EstimateTokens(messages)
	if before <= budget {
		return messages, 0
	}
	out := cloneMessages(messages)

	// 第一步：折叠旧的工具结果，保持消息结构不变
	totalResults := countToolResultMessages(out)
	resultSeen := 0
	for i := range out {
		if out[i].Role != llm.RoleUser || !hasToolResult(out[i].Content) {
			continue
		}
		resultSeen++
		if resultSeen > totalResults-keepRecentResults {
			continue // 最近几条保持原样
		}
		for j, b := range out[i].Content {
			if b.Type != "tool_result" {
				continue
			}
			size := len(toolResultText(b))
			if size < 200 {
				continue
			}
			if line, ok := foldToPointer(toolResultText(b)); ok {
				// 已经治理过的：只剩指针那一行。回读路径留着，占地方的头尾预览去掉——
				// 换成一句"已折叠"会把路径一起丢掉，那一轮就再也取不回原文了。
				out[i].Content[j].Content = line
				continue
			}
			out[i].Content[j].Content = fmt.Sprintf("（较早的工具输出已折叠，原本 %d 字节）", size)
		}
	}

	if after := EstimateTokens(out); after <= budget {
		return out, before - after
	}

	// 第二步：整轮丢弃最早的往返
	var dropped []llm.Message
	for len(out) > keepRecentRounds+2 && EstimateTokens(out) > budget {
		n := dropRound(out, 1) // 索引 0 是用户的原始任务，保留
		dropped = append(dropped, out[1:1+n]...)
		out = append(out[:1], out[1+n:]...)
	}

	// 丢掉的往返里有结论，压成一段摘要贴在任务后面；拿不到摘要就接受丢弃。
	// 追加进第一条消息而不是另起一条：索引 0 已经是 user，再起一条 user 就破坏了交替。
	if len(dropped) > 0 && summarize != nil {
		note, err := summarize(ctx, dropped)
		switch {
		case err != nil:
			log.Warn("历史摘要失败，本轮按丢弃处理: %v", err)
		case strings.TrimSpace(note) != "":
			out[0].Content = append(out[0].Content, llm.ContentBlock{
				Type: "text",
				Text: "📝 早前对话摘要（为守住上下文预算被压缩，非原文）：\n" + strings.TrimSpace(note),
			})
		}
	}

	return out, before - EstimateTokens(out)
}

// keepRecentRounds 折叠第二阶段至少保留的最近消息条数。
const keepRecentRounds = 4

// dropRound 从索引 i 开始识别一整轮往返，返回应丢弃的条数。
// 一轮 = 一条 assistant（带 tool_use）+ 紧随其后的 tool_result user 消息。
func dropRound(messages []llm.Message, i int) int {
	if i >= len(messages) {
		return 0
	}
	if messages[i].Role == llm.RoleAssistant && i+1 < len(messages) &&
		hasToolResult(messages[i+1].Content) {
		return 2
	}
	return 1
}

func cloneMessages(messages []llm.Message) []llm.Message {
	out := make([]llm.Message, 0, len(messages))
	for _, m := range messages {
		blocks := make([]llm.ContentBlock, len(m.Content))
		copy(blocks, m.Content)
		out = append(out, llm.Message{Role: m.Role, Content: blocks})
	}
	return out
}

func hasToolResult(blocks []llm.ContentBlock) bool {
	for _, b := range blocks {
		if b.Type == "tool_result" {
			return true
		}
	}
	return false
}

func countToolResultMessages(messages []llm.Message) int {
	n := 0
	for _, m := range messages {
		if m.Role == llm.RoleUser && hasToolResult(m.Content) {
			n++
		}
	}
	return n
}
