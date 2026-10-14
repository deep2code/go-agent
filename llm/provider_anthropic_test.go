package llm

import (
	"context"
	"strings"
	"testing"
)

// Anthropic 的 SSE 用 event/data 两行描述一个事件，工具参数走 input_json_delta。
// 这里的每条流都按真机抓到的形状写，包括 message_stop 缺失这种半截连接。

func sse(pairs ...[2]string) string {
	var b strings.Builder
	for _, p := range pairs {
		b.WriteString("event: " + p[0] + "\n")
		b.WriteString("data: " + p[1] + "\n\n")
	}
	return b.String()
}

func parseAnthropic(t *testing.T, stream string, onText func(string)) (*MessageResponse, error) {
	t.Helper()
	p := newAnthropicSSEParser()
	return p.parseAnthropicStream(context.Background(), strings.NewReader(stream), onText)
}

func TestAnthropicSSETextStream(t *testing.T) {
	stream := sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x"},"usage":{"input_tokens":100,"output_tokens":1}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"，世界"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)

	var echoed strings.Builder
	resp, err := parseAnthropic(t, stream, func(s string) { echoed.WriteString(s) })
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "msg_1" || resp.Model != "claude-x" || resp.StopReason != "end_turn" {
		t.Errorf("元信息丢失: %+v", resp)
	}
	if resp.Usage.InputTokens != 100 || resp.Usage.OutputTokens != 15 {
		t.Errorf("用量应分别取自 message_start / message_delta: %+v", resp.Usage)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "你好，世界" {
		t.Fatalf("文本增量没有拼好: %+v", resp.Content)
	}
	if echoed.String() != "你好，世界" {
		t.Errorf("回显 = %q", echoed.String())
	}
}

// 缓存段只在 message_start 出现，逐字段拷贝会把它们丢掉：
// 剩下的 InputTokens 是"未命中缓存的那一小段"，拿它当真实输入去校准预算会一路偏小。
func TestAnthropicSSECacheUsage(t *testing.T) {
	stream := sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m","model":"claude-x"},"usage":{"input_tokens":20,"cache_creation_input_tokens":1800,"cache_read_input_tokens":12000,"output_tokens":2}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":30}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)

	resp, err := parseAnthropic(t, stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	u := resp.Usage
	if u.InputTokens != 20 || u.CacheCreationInputTokens != 1800 || u.CacheReadInputTokens != 12000 {
		t.Fatalf("缓存用量没接住: %+v", u)
	}
	if u.PromptTokens() != 13820 {
		t.Errorf("PromptTokens = %d，期望 13820", u.PromptTokens())
	}
	// message_delta 只带 output_tokens，不该把缓存段冲成 0。
	if u.OutputTokens != 30 || u.CacheReadInputTokens != 12000 {
		t.Errorf("收尾用量 = %+v", u)
	}
	if u.Total() != 13850 {
		t.Errorf("Total = %d，期望 13850", u.Total())
	}
}

func TestAnthropicSSEThinkingThenTool(t *testing.T) {
	stream := sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"msg_2","model":"claude-x"},"usage":{"input_tokens":8}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"先看看目录"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"我列一下"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_9","name":"list_directory","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"directory\": "}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"src\"}"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)

	resp, err := parseAnthropic(t, stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) != 3 {
		t.Fatalf("块数 = %d，期望 3: %+v", len(resp.Content), resp.Content)
	}
	if resp.Content[0].Type != "thinking" || resp.Content[0].Thinking != "先看看目录" {
		t.Errorf("thinking 块不对: %+v", resp.Content[0])
	}
	if resp.Content[1].Type != "text" || resp.Content[1].Text != "我列一下" {
		t.Errorf("text 块不对: %+v", resp.Content[1])
	}
	tool := resp.Content[2]
	if tool.Type != "tool_use" || tool.ID != "tu_9" || tool.Name != "list_directory" {
		t.Errorf("tool_use 元信息不对: %+v", tool)
	}
	if tool.Input["directory"] != "src" {
		t.Errorf("参数分片未拼接: %+v", tool.Input)
	}
	if tool.Partial {
		t.Error("完整参数被误判为截断")
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("StopReason = %s", resp.StopReason)
	}
}

func TestAnthropicSSEEmptyAndTruncatedToolInput(t *testing.T) {
	t.Run("无参工具", func(t *testing.T) {
		resp, err := parseAnthropic(t, sse(
			[2]string{"message_start", `{"type":"message_start","message":{"id":"m"}}`},
			[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"plan"}}`},
			[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`},
			[2]string{"message_stop", `{"type":"message_stop"}`},
		), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Content[0].Partial {
			t.Errorf("空参数是合法的（无参工具），不该判为截断: %+v", resp.Content[0])
		}
		if resp.Content[0].Input == nil {
			t.Error("Input 应至少是空 map，序列化时不能是 null")
		}
	})

	t.Run("参数被 max_tokens 切一半", func(t *testing.T) {
		resp, err := parseAnthropic(t, sse(
			[2]string{"message_start", `{"type":"message_start","message":{"id":"m"}}`},
			[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"write_file"}}`},
			[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"content\": \"package main"}}`},
			[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4096}}`},
			[2]string{"message_stop", `{"type":"message_stop"}`},
		), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StopReason != "max_tokens" {
			t.Errorf("StopReason = %s", resp.StopReason)
		}
		if !resp.Content[0].Partial {
			t.Fatalf("半截 JSON 必须标记 Partial，否则会被当成一次正常调用执行掉: %+v", resp.Content[0])
		}
	})
}

// 网络在 message_stop 之前断开时，累积到的内容不可信；
// 如果当成正经响应，用户会看到"模型好像说了两句就没动静"。
func TestAnthropicSSEMissingMessageStop(t *testing.T) {
	_, err := parseAnthropic(t, sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m"}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"断在这里"}}`},
	), nil)
	if err == nil {
		t.Fatal("未收到 message_stop 应报错")
	}
	if !strings.Contains(err.Error(), "message_stop") {
		t.Errorf("错误信息应说明流不完整: %v", err)
	}
}

func TestAnthropicSSEBadJSON(t *testing.T) {
	_, err := parseAnthropic(t, sse(
		[2]string{"message_start", `{"type":"message_start"`},
	), nil)
	if err == nil {
		t.Error("坏 JSON 应报错而不是静默忽略")
	}
}

// delta 落在还没 start 的 index 上（乱序/丢事件）不应 panic。
func TestAnthropicSSEDeltaWithoutBlock(t *testing.T) {
	resp, err := parseAnthropic(t, sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":5,"delta":{"type":"text_delta","text":"x"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":5}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) != 0 {
		t.Errorf("凭空出现的 index 不该造出内容块: %+v", resp.Content)
	}
}

func TestMapAnthropicStopReason(t *testing.T) {
	cases := map[string]string{
		"end_turn":      FinishStop,
		"stop_sequence": FinishStop,
		"pause_turn":    FinishStop,
		"refusal":       FinishStop,
		"tool_use":      FinishToolCalls,
		"max_tokens":    FinishLength,
		"":              FinishStop,
		"whatever":      FinishError,
	}
	for in, want := range cases {
		if got := mapAnthropicStopReason(in); got != want {
			t.Errorf("mapAnthropicStopReason(%q) = %s want %s", in, got, want)
		}
	}
}

func TestAnthropicParseErrorMessages(t *testing.T) {
	p := NewAnthropicProvider("k")
	cases := []struct {
		status    int
		body      string
		wantRetry bool
		wantOver  bool
		wantInMsg string
	}{
		{401, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`, false, false, "认证失败"},
		{429, `{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded"}}`, true, false, "频率限制"},
		{529, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, true, false, "服务过载"},
		{400, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 300000 tokens"}}`, false, true, "请求参数错误"},
		{500, `not json at all`, true, false, "服务器错误"},
	}
	for _, c := range cases {
		err := p.parseError(c.status, []byte(c.body), 0)
		var e *Error
		if !asError(err, &e) {
			t.Fatalf("status %d 应返回 *Error，实际 %T", c.status, err)
		}
		if e.Retryable != c.wantRetry {
			t.Errorf("status %d Retryable = %v want %v", c.status, e.Retryable, c.wantRetry)
		}
		if e.OverContext != c.wantOver {
			t.Errorf("status %d OverContext = %v want %v", c.status, e.OverContext, c.wantOver)
		}
		if !strings.Contains(e.Message, c.wantInMsg) {
			t.Errorf("status %d 文案 %q 应包含 %q", c.status, e.Message, c.wantInMsg)
		}
		if c.status == 401 && !strings.Contains(e.Message, "ANTHROPIC_API_KEY") {
			t.Error("认证失败应提示该检查哪个环境变量")
		}
	}
}
