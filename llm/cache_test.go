package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func longSystem() string { return strings.Repeat("规则说明。", 900) } // 1350 个汉字 ≈ 1350 token，远超打断点的门槛

func text(s string) ContentBlock { return ContentBlock{Type: "text", Text: s} }

func toolSet(n int) []ToolDef {
	tools := make([]ToolDef, 0, n)
	for i := 0; i < n; i++ {
		tools = append(tools, ToolDef{Name: string(rune('a' + i)), Description: "工具"})
	}
	return tools
}

// 断点只打一个，打在前缀末尾：把 system 和 tools 各打一个会让 system 被写两遍。
func TestCacheBreakpointIsSingle(t *testing.T) {
	tools := toolSet(3)
	req := ChatRequest{Model: "claude-x", MaxTokens: 100, System: longSystem(), Tools: tools}
	wire := buildAnthropicRequest(req)

	if wire.System[0].CacheControl != nil {
		t.Error("有工具时不该在 system 上再打断点，system 会被缓存两遍")
	}
	for i := 0; i < len(wire.Tools)-1; i++ {
		if wire.Tools[i].CacheControl != nil {
			t.Errorf("第 %d 个工具不该带断点", i)
		}
	}
	last := wire.Tools[len(wire.Tools)-1].CacheControl
	if last == nil || last.Type != "ephemeral" {
		t.Errorf("最后一个工具没有断点: %+v", wire.Tools[len(wire.Tools)-1])
	}
}

// 上层把同一份 []ToolDef 用于 token 估算和后续轮次；就地写入会让断点泄漏。
func TestCacheBreakpointDoesNotMutateCaller(t *testing.T) {
	tools := toolSet(2)
	buildAnthropicRequest(ChatRequest{System: longSystem(), Tools: tools})
	for _, d := range tools {
		if d.CacheControl != nil {
			t.Fatalf("调用方的工具定义被改写了: %+v", tools)
		}
	}
}

// 短前缀换不来命中，却照样按 1.25 倍付写入费——摘要那类一次性请求就该跳过。
func TestShortPrefixGetsNoCacheControl(t *testing.T) {
	wire := buildAnthropicRequest(ChatRequest{System: "把要点压成列表", Tools: toolSet(1)})
	if wire.System[0].CacheControl != nil {
		t.Error("短 system 不该带断点")
	}
	for _, d := range wire.Tools {
		if d.CacheControl != nil {
			t.Errorf("短前缀的请求里工具 %q 带了断点", d.Name)
		}
	}

	// 只有 system、没有工具时，断点退到 system 上。
	long := longSystem() + strings.Repeat("x", 4000)
	wire = buildAnthropicRequest(ChatRequest{System: long})
	if len(wire.Tools) != 0 || wire.System[0].CacheControl == nil {
		t.Error("无工具时应在 system 末块打断点")
	}
}

func TestSystemSerializedAsBlockArray(t *testing.T) {
	body, err := json.Marshal(buildAnthropicRequest(ChatRequest{
		Model: "claude-x", MaxTokens: 64, System: "规则", Messages: []Message{{Role: RoleUser, Content: []ContentBlock{text("hi")}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatal(err)
	}
	system, ok := raw["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system 应为单元素数组，实际: %v", raw["system"])
	}
	block := system[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "规则" {
		t.Errorf("system 块内容不对: %v", block)
	}
	if _, present := block["cache_control"]; present {
		t.Error("短 system 不该序列化出 cache_control")
	}
}
