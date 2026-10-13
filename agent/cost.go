package agent

import (
	"strings"

	"go-agent/llm"
)

// ============================================================================
// 成本估算
// ============================================================================

// price 是每百万 token 的美元价格。
type price struct {
	input  float64
	output float64
}

// prices 是内置价格表，数字取自各厂商公开定价页。它会随调价过期，
// 只用于判断"这轮任务大概花掉多少钱"的量级，不能当账单。
var prices = []struct {
	match string
	cost  price
}{
	{"deepseek-chat", price{0.27, 1.10}},
	{"deepseek-reasoner", price{0.28, 0.42}},
	{"deepseek", price{0.27, 1.10}},
	{"claude-opus", price{15.00, 75.00}},
	{"claude-sonnet", price{3.00, 15.00}},
	{"claude-haiku", price{0.80, 4.00}},
	{"gpt-4o", price{2.50, 10.00}},
	{"gpt-4.1", price{2.00, 8.00}},
	{"qwen", price{0.50, 2.00}},
	{"glm", price{0.60, 2.20}},
}

// EstimateCost 估算 token 用量的美元成本；模型不在价目表里时返回 0。
func EstimateCost(model string, usage llm.Usage) float64 {
	lower := strings.ToLower(model)
	for _, e := range prices {
		if strings.Contains(lower, e.match) {
			return float64(usage.InputTokens)/1e6*e.cost.input +
				float64(usage.OutputTokens)/1e6*e.cost.output
		}
	}
	return 0
}
