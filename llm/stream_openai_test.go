package llm

import (
	"context"
	"strings"
	"testing"
)

// SSE 是逐块到达的，工具参数会被拆成很多个 delta。
// 这里锁定的都是真实踩过的形状：拼接顺序、缺 [DONE]、被 max_tokens 切一半的 JSON。

func parseOpenAI(t *testing.T, stream string, onText func(string)) (*ChatResponse, error) {
	t.Helper()
	p := newOpenAISSEParser(strings.NewReader(stream), onText)
	return p.parse(context.Background())
}

func TestOpenAISSETextStream(t *testing.T) {
	stream := `data: {"id":"c1","model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"content":"，世界"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	var echoed strings.Builder
	resp, err := parseOpenAI(t, stream, func(s string) { echoed.WriteString(s) })
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "c1" || resp.Model != "deepseek-chat" {
		t.Errorf("元信息丢失: %+v", resp)
	}
	if resp.FinishReason != FinishStop {
		t.Errorf("FinishReason = %s", resp.FinishReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "你好，世界" {
		t.Fatalf("文本增量没有拼好: %+v", resp.Content)
	}
	if echoed.String() != "你好，世界" {
		t.Errorf("回显给用户的增量 = %q", echoed.String())
	}
}

func TestOpenAISSEToolCallAssembly(t *testing.T) {
	// 两个工具调用交错到达，arguments 各被拆成三片，index 从 1 开始且不连续
	stream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"read_file","arguments":""}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"list_directory","arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"file_path\""}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":": \"a.go\"}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	resp, err := parseOpenAI(t, stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != FinishToolCalls {
		t.Errorf("FinishReason = %s", resp.FinishReason)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("应拼出 2 个工具调用，实际 %d: %+v", len(resp.Content), resp.Content)
	}
	// 按 index 升序，index 0 在前
	if got := resp.Content[0].Name; got != "list_directory" {
		t.Errorf("第一个调用是 %q，应按 index 排序", got)
	}
	if resp.Content[0].ID != "call_a" || resp.Content[0].Partial {
		t.Errorf("无参工具解析异常: %+v", resp.Content[0])
	}
	second := resp.Content[1]
	if second.Name != "read_file" || second.ID != "call_b" {
		t.Errorf("第二个调用元信息不对: %+v", second)
	}
	if second.Input["file_path"] != "a.go" {
		t.Errorf("参数分片未拼接: %+v", second.Input)
	}
	if second.Partial {
		t.Error("完整参数被误判为截断")
	}
}

// 输出撞 max_tokens 时 arguments 只到了一半。执行它等于执行一个猜测，
// 必须标记出来让上层拦下。
func TestOpenAISSETruncatedArgs(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","function":{"name":"write_file","arguments":"{\"file_path\": \"a.go\", \"content\": \"package main\\nfunc main() {"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	resp, err := parseOpenAI(t, stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != FinishLength {
		t.Errorf("FinishReason = %s，期望 length", resp.FinishReason)
	}
	if len(resp.Content) != 1 || !resp.Content[0].Partial {
		t.Fatalf("半截 JSON 应标记 Partial: %+v", resp.Content)
	}
}

func TestOpenAISSEEearlyEndIsError(t *testing.T) {
	// 连接被中途掐断：没有 [DONE] 也没有 finish_reason，内容不可信
	stream := `data: {"choices":[{"index":0,"delta":{"content":"说到一半"}}]}

`
	if _, err := parseOpenAI(t, stream, nil); err == nil {
		t.Fatal("提前结束的流应报错，让上层重试，而不是当成一条短回复")
	}

	if _, err := parseOpenAI(t, "", nil); err == nil {
		t.Error("空流应报错")
	}
}

func TestOpenAISSEUsageAndFormatVariants(t *testing.T) {
	// 带 include_usage 时，最后一个 chunk 只有 usage、没有 choices；
	// 同时这里用 "data:{...}"（无空格）写法。
	stream := strings.Join([]string{
		`data:{"id":"u1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		`data:{"id":"u1","usage":{"prompt_tokens":1234,"completion_tokens":56}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	resp, err := parseOpenAI(t, stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.InputTokens != 1234 || resp.Usage.OutputTokens != 56 {
		t.Errorf("用量未捕获: %+v", resp.Usage)
	}
	if resp.Usage.Total() != 1290 {
		t.Errorf("Total = %d", resp.Usage.Total())
	}
}

func TestOpenAISSEGarbageChunk(t *testing.T) {
	// 中间代理偶尔会插入非 data 行（注释、event: 行），不该让整个流解析失败
	stream := `: keep-alive

event: message

data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}

data: [DONE]
`
	if _, err := parseOpenAI(t, stream, nil); err != nil {
		t.Errorf("无关行应被忽略: %v", err)
	}
}

func TestIsPartialJSON(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"", false},
		{"{}", false},
		{`{"a":1}`, false},
		{"  ", false},
		{`{"a":`, true},
		{`{"a":1`, true},
		{`{"a":"unterminated}`, true},
		{`[1,2]`, true}, // 顶层必须是对象
	}
	for _, c := range cases {
		if got := isPartialJSON(c.in); got != c.want {
			t.Errorf("isPartialJSON(%q) = %v want %v", c.in, got, c.want)
		}
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"stop":           FinishStop,
		"tool_calls":     FinishToolCalls,
		"length":         FinishLength,
		"":               FinishStop,
		"content_filter": FinishError,
		"bogus":          FinishError,
	}
	for in, want := range cases {
		if got := mapFinishReason(in); got != want {
			t.Errorf("mapFinishReason(%q) = %s want %s", in, got, want)
		}
	}
}
