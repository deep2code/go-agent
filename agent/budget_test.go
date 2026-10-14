package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"go-agent/llm"
)

// 本地估算和服务端真实计数不是一个口径。这组测试盯的就是校准与止损。

func TestCalibrateTracksServerUsage(t *testing.T) {
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(), nil)

	// 未校准时按 1 处理，不能让预算凭空变小。
	if got := a.tokenScale(); got != 1 {
		t.Fatalf("未校准的比例 = %v，期望 1", got)
	}

	a.sentEstimate = 1000
	a.calibrate(2000)
	if got := a.estScale; got != 2 {
		t.Errorf("首个样本比例 = %v，期望 2（服务端 2000 / 估算 1000）", got)
	}

	// 第二个样本折中，逐步收敛而不是被单轮带跑。
	a.sentEstimate = 1000
	a.calibrate(1000)
	if got := a.estScale; got != 1.5 {
		t.Errorf("折中后比例 = %v，期望 1.5", got)
	}

	// 离谱的单轮（比如刚折叠完历史）必须限幅，且限幅后仍落在 [0.4, 3] 内。
	a.sentEstimate = 1000
	a.calibrate(1)
	if got := a.estScale; got < 0.4 || got > 3 {
		t.Errorf("限幅失效，比例 = %v", got)
	}
	a.estScale = 0
	a.sentEstimate = 1000
	a.calibrate(1)
	if got := a.estScale; got != 0.4 {
		t.Errorf("过低样本应被抬到 0.4，得到 %v", got)
	}
	a.estScale = 0
	a.sentEstimate = 1000
	a.calibrate(1_000_000)
	if got := a.estScale; got != 3 {
		t.Errorf("过高样本应被压到 3，得到 %v", got)
	}

	// 服务端没给用量时不该动比例。
	before := a.estScale
	a.sentEstimate = 1000
	a.calibrate(0)
	if a.estScale != before {
		t.Errorf("用量缺失时比例被改写了：%v -> %v", before, a.estScale)
	}
}

// 校准必须真的影响折叠决定：估算偏乐观时，应当在服务端撞墙之前就折。
// 每条输出都刻意低于 l2CacheFrom：入口治理和指针替代一掺和，"省下来"就不知道该记在
// 校准还是治理的账上，这一级在这里必须完全不出场。
func TestCalibratedScaleTriggersCompaction(t *testing.T) {
	long := strings.Repeat("abcdefgh ", 200) // 1800 字节 ≈ 450 token
	var history []llm.Message
	history = append(history, llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{text("把每个文件都读一遍")}})
	for i := 0; i < 8; i++ {
		history = append(history,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{toolUse("u", "reader")}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{toolResult("u", long)}})
	}

	est := EstimateTokens(history)
	p := &scriptedProvider{steps: []scriptedStep{stopReply("摘要", 0, 0)}}
	a, _ := newAgent(t, p, registryWith(&fakeTool{name: "reader", result: "x"}), func(c *Config) {
		c.ContextWindow = 100000
		c.MaxTokens = 1000
	})
	// 把预算设成估算值的 1.2 倍：未校准不该折，校准到 1.5 倍就必须折。
	overhead := EstimateTools(a.registry.Definitions()) + estimateTextTokens(a.config.SystemPrompt)
	a.config.ContextWindow = est*6/5 + overhead + 1000 + contextReserve

	a.history = history
	if freed, err := a.prepareForSend(context.Background()); err != nil {
		t.Fatal(err)
	} else if freed != 0 {
		t.Errorf("未超预算却折掉了 %d token", freed)
	}

	a.history = history
	a.estScale = 1.5
	freed, err := a.prepareForSend(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if freed <= 0 {
		t.Fatalf("真实口径已达 %d（预算 %d），却没有折叠", int(float64(est)*1.5), a.config.HistoryBudget()-overhead)
	}
	if after := EstimateTokens(a.history); after >= est {
		t.Errorf("折叠后估算没变小：%d -> %d", est, after)
	}
	// 折叠完要能报出发出去的真实口径估算，供下一轮校准。
	if a.sentEstimate <= 0 {
		t.Error("prepareForSend 没有记录本次发送的估算值")
	}
}

func TestCostLimitStopsBeforeNextCall(t *testing.T) {
	ft := &fakeTool{name: "reader", result: "内容"}
	expensive := scriptedStep{resp: &llm.ChatResponse{
		FinishReason: llm.FinishToolCalls,
		Content:      []llm.ContentBlock{{Type: "tool_use", ID: "t1", Name: "reader"}},
		// deepseek-chat 输入 $0.27/百万：这一次就烧掉 $0.27
		Usage: llm.Usage{InputTokens: 1_000_000, OutputTokens: 1_000},
	}}
	p := &scriptedProvider{steps: []scriptedStep{
		expensive,
		toolStep("t2", "reader", nil, ""),
		stopReply("不该走到这里", 0, 0),
	}}
	a, _ := newAgent(t, p, registryWith(ft), func(c *Config) { c.MaxCostUSD = 0.1 })

	res, err := a.Run(context.Background(), "干活")
	if !errors.Is(err, ErrCostLimit) {
		t.Fatalf("err = %v，期望 ErrCostLimit", err)
	}
	if res.StopReason != "cost_limit" {
		t.Errorf("StopReason = %q，期望 cost_limit", res.StopReason)
	}
	if len(p.requests) != 1 {
		t.Errorf("撞上限后仍发起了 %d 次调用，期望 1 次", len(p.requests))
	}
	if res.CostUSD < 0.1 {
		t.Errorf("CostUSD = %v，应已超上限", res.CostUSD)
	}
	// 上限拦不住已经花掉的那次，但历史必须留给用户接着用。
	if a.HistoryLen() == 0 {
		t.Error("止损把对话历史弄丢了")
	}
}

// 默认不设限：老行为不能被新字段悄悄改掉。
func TestNoCostLimitByDefault(t *testing.T) {
	p := &scriptedProvider{steps: []scriptedStep{
		{resp: &llm.ChatResponse{
			FinishReason: llm.FinishStop,
			Content:      []llm.ContentBlock{{Type: "text", Text: "完成"}},
			Usage:        llm.Usage{InputTokens: 5_000_000},
		}},
	}}
	a, _ := newAgent(t, p, registryWith(), nil)
	res, err := a.Run(context.Background(), "说一句")
	if err != nil {
		t.Fatalf("默认不该止损: %v", err)
	}
	if res.StopReason != "completed" {
		t.Errorf("StopReason = %q", res.StopReason)
	}
}

// 价目表里没有的模型无法换算金额，上限自然拦不住——这必须说出来，
// 不能让用户以为已经设了止损。
func TestCostLimitWarnsWhenModelUnpriced(t *testing.T) {
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(), func(c *Config) {
		c.Model = "某厂商的新模型"
		c.MaxCostUSD = 1
	})
	a.usage = llm.Usage{InputTokens: 1000, OutputTokens: 100}
	if err := a.checkCostLimit(); err != nil {
		t.Fatalf("无价目表时不该误报超限: %v", err)
	}
	if a.warnedNoPrice != true {
		t.Error("没有告知上限不会生效")
	}
}

func TestEstimateCostAccountsForCache(t *testing.T) {
	u := llm.Usage{
		InputTokens:              1_000_000,
		OutputTokens:             1_000_000,
		CacheCreationInputTokens: 1_000_000,
		CacheReadInputTokens:     1_000_000,
	}
	got := EstimateCost("deepseek-chat", u)
	// 输入 0.27：普通 1.0 + 写 1.25 + 读 0.1 = 2.35 段；输出 1.10。
	want := 2.35*0.27 + 1.10
	if diff := got - want; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("成本 = %v，期望 %v", got, want)
	}
	if got := EstimateCost("未知模型", u); got != 0 {
		t.Errorf("价目表外应返回 0，得到 %v", got)
	}
}

func TestAddUsageIncludesCacheTokens(t *testing.T) {
	a, _ := newAgent(t, &scriptedProvider{}, registryWith(), nil)
	a.AddUsage(llm.Usage{InputTokens: 10, OutputTokens: 5, CacheReadInputTokens: 100, CacheCreationInputTokens: 20})
	a.AddUsage(llm.Usage{InputTokens: 1, CacheReadInputTokens: 2})

	got := a.Usage()
	if got.InputTokens != 11 || got.OutputTokens != 5 ||
		got.CacheReadInputTokens != 102 || got.CacheCreationInputTokens != 20 {
		t.Errorf("累加后的用量 = %+v", got)
	}
	if got.PromptTokens() != 133 {
		t.Errorf("PromptTokens = %d，期望 133（输入 11 + 写 20 + 读 102）", got.PromptTokens())
	}
}
