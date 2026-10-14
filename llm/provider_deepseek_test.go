package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func asError(err error, target **Error) bool {
	return errors.As(err, target)
}

// sseChunks 把若干 JSON chunk 包成 OpenAI 格式的 SSE 响应体。
func sseChunks(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func fastRetry(p *DeepSeekProvider) {
	p.SetRetry(Retry{MaxAttempts: 4, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
}

func userMsg(s string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{{Type: "text", Text: s}}}
}

// 走一遍真 HTTP：请求体形状、SSE 拼接、用量捕获，任何一环缺了都会在这里露出来。
func TestDeepSeekStreamingChatOverHTTP(t *testing.T) {
	var mu sync.Mutex
	var bodies []string
	var auth, accept string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		auth = r.Header.Get("Authorization")
		accept = r.Header.Get("Accept")
		bodies = append(bodies, string(body))
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sseChunks(
			`{"id":"c1","model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"好的"}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\"file_"}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"path\": \"main.go\"}"}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			`{"id":"c1","usage":{"prompt_tokens":700,"completion_tokens":20}}`,
		))
	}))
	defer srv.Close()

	p := NewDeepSeekProviderWithBaseURL(srv.URL, "test-key")
	fastRetry(p)

	messages := []Message{
		userMsg("读一下 main.go"),
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: "text", Text: "我看看"},
			{Type: "tool_use", ID: "call_0", Name: "read_file", Input: map[string]any{"file_path": "go.mod"}},
		}},
		{Role: RoleUser, Content: []ContentBlock{
			{Type: "tool_result", ToolUseID: "call_0", Content: "module go-agent"},
		}},
	}

	var echoed string
	resp, err := p.StreamingChat(context.Background(), ChatRequest{
		Model:     "deepseek-chat",
		MaxTokens: 1024,
		System:    "你是助手",
		Messages:  messages,
		Tools: []ToolDef{{
			Name: "read_file", Description: "读取文件",
			InputSchema: InputSchema{
				Type:       "object",
				Properties: map[string]Property{"file_path": {Type: "string", Description: "路径"}},
				Required:   []string{"file_path"},
			},
		}},
		RoundNumber: 1,
	}, func(s string) { echoed += s })
	if err != nil {
		t.Fatal(err)
	}

	if auth != "Bearer test-key" {
		t.Errorf("Authorization 头 = %q", auth)
	}
	if !strings.Contains(accept, "text/event-stream") {
		t.Errorf("Accept 头 = %q", accept)
	}
	if resp.FinishReason != FinishToolCalls {
		t.Errorf("FinishReason = %s", resp.FinishReason)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("应同时拿到文本和工具调用: %+v", resp.Content)
	}
	if resp.Content[0].Text != "好的" || echoed != "好的" {
		t.Errorf("文本或回显不对: content=%+v echoed=%q", resp.Content[0], echoed)
	}
	call := resp.Content[1]
	if call.Name != "read_file" || call.ID != "call_1" {
		t.Errorf("工具调用元信息不对: %+v", call)
	}
	if call.Input["file_path"] != "main.go" {
		t.Errorf("参数跨 chunk 拼接失败: %+v", call.Input)
	}
	if call.Partial {
		t.Error("完整参数被误判为截断")
	}
	if resp.Usage.Total() != 720 {
		t.Errorf("用量 = %+v，流式响应也要能算钱", resp.Usage)
	}

	mu.Lock()
	defer mu.Unlock()
	sent := bodies[0]
	if !strings.Contains(sent, `"stream":true`) {
		t.Error("请求体应带 stream:true")
	}
	if !strings.Contains(sent, `"include_usage":true`) {
		t.Error("必须带 stream_options.include_usage，否则拿不到 token 用量")
	}
	if !strings.Contains(sent, `"system"`) || !strings.Contains(sent, "你是助手") {
		t.Errorf("system 没带上: %s", sent)
	}
	if !strings.Contains(sent, `"read_file"`) || !strings.Contains(sent, `"file_path"`) {
		t.Errorf("工具定义不完整: %s", sent)
	}
	// 历史往返必须转成 OpenAI 形态：assistant.tool_calls + role=tool 结果
	if !strings.Contains(sent, `"tool_calls"`) {
		t.Errorf("assistant 的 tool_use 没转成 tool_calls: %s", sent)
	}
	if !strings.Contains(sent, `"role":"tool"`) || !strings.Contains(sent, `"tool_call_id":"call_0"`) {
		t.Errorf("tool_result 没转成 role=tool: %s", sent)
	}
}

func TestStreamingChatRetriesTransientFailures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
	}{
		{"429 限流", http.StatusTooManyRequests},
		{"503 不可用", http.StatusServiceUnavailable},
		{"529 过载", statusOverloaded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls int
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if calls == 1 {
					w.WriteHeader(tc.status)
					io.WriteString(w, `{"error":{"message":"temporarily unavailable"}}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, sseChunks(
					`{"choices":[{"index":0,"delta":{"content":"恢复了"},"finish_reason":"stop"}]}`,
				))
			}))
			defer srv.Close()

			p := NewDeepSeekProviderWithBaseURL(srv.URL, "k")
			fastRetry(p)

			resp, err := p.StreamingChat(context.Background(),
				ChatRequest{Model: "m", MaxTokens: 100, Messages: []Message{userMsg("hi")}}, nil)
			if err != nil {
				t.Fatalf("第二次就该成功: %v", err)
			}
			if len(resp.Content) == 0 || resp.Content[0].Text != "恢复了" {
				t.Errorf("重试后内容不对: %+v", resp.Content)
			}
			if calls != 2 {
				t.Errorf("调用次数 = %d，期望 2", calls)
			}
		})
	}
}

// 已经往终端吐过增量时再重试，用户会看到同一段话出现两遍。
func TestStreamingChatNoRetryAfterStreaming(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		if calls == 1 {
			// 半条流：发了文本就断，没有 finish_reason 也没有 [DONE]
			io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"说了一半\"}}]}\n\n")
			return
		}
		io.WriteString(w, sseChunks(`{"choices":[{"index":0,"delta":{"content":"第二次"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p := NewDeepSeekProviderWithBaseURL(srv.URL, "k")
	fastRetry(p)

	var echoed strings.Builder
	_, err := p.StreamingChat(context.Background(),
		ChatRequest{Model: "m", MaxTokens: 100, Messages: []Message{userMsg("hi")}},
		func(s string) { echoed.WriteString(s) })
	if err == nil {
		t.Fatal("半条流应报错")
	}
	if calls != 1 {
		t.Errorf("已经回显过内容还重试，会吐出重复文本: calls=%d", calls)
	}
	if echoed.String() != "说了一半" {
		t.Errorf("已回显内容 = %q", echoed.String())
	}
}

// 没吐出任何内容就断掉的流是可以安全重试的。
func TestStreamingChatRetriesBeforeFirstToken(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			return // 立刻 EOF，一个字节都没发
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sseChunks(`{"choices":[{"index":0,"delta":{"content":"补上了"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p := NewDeepSeekProviderWithBaseURL(srv.URL, "k")
	fastRetry(p)

	resp, err := p.StreamingChat(context.Background(),
		ChatRequest{Model: "m", MaxTokens: 100, Messages: []Message{userMsg("hi")}}, nil)
	if err != nil {
		t.Fatalf("未回显前应重试成功: %v", err)
	}
	if resp.Content[0].Text != "补上了" || calls != 2 {
		t.Errorf("calls=%d content=%+v", calls, resp.Content)
	}
}

func TestStreamingChatNoRetryOnFatal(t *testing.T) {
	cases := []struct {
		name     string
		status   int
		body     string
		wantCall int
		check    func(*testing.T, *Error)
	}{
		{
			name: "认证失败", status: http.StatusUnauthorized,
			body: `{"error":{"message":"invalid api key"}}`, wantCall: 1,
			check: func(t *testing.T, e *Error) {
				if e.Retryable {
					t.Error("密钥错误重试只会重复失败")
				}
			},
		},
		{
			name: "上下文超限", status: http.StatusBadRequest,
			body: `{"error":{"message":"This model's maximum context length is 16384 tokens"}}`, wantCall: 1,
			check: func(t *testing.T, e *Error) {
				if !e.OverContext {
					t.Error("应标记为上下文超限，交给上层压缩历史后重来")
				}
				if e.Retryable {
					t.Error("原样重试必然再撞一次")
				}
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var calls int
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				w.WriteHeader(c.status)
				io.WriteString(w, c.body)
			}))
			defer srv.Close()

			p := NewDeepSeekProviderWithBaseURL(srv.URL, "k")
			fastRetry(p)

			_, err := p.StreamingChat(context.Background(),
				ChatRequest{Model: "m", MaxTokens: 100, Messages: []Message{userMsg("hi")}}, nil)
			if err == nil {
				t.Fatal("应返回错误")
			}
			var e *Error
			if !asError(err, &e) {
				t.Fatalf("错误类型 %T: %v", err, err)
			}
			c.check(t, e)
			if calls != c.wantCall {
				t.Errorf("调用 %d 次，期望 %d", calls, c.wantCall)
			}
		})
	}
}

// 超限标记必须能穿过 provider 的包装到达 agent。
func TestOverContextSurvivesWrapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"prompt is too long: 250000 tokens > 200000"}}`)
	}))
	defer srv.Close()

	p := NewDeepSeekProviderWithBaseURL(srv.URL, "k")
	_, err := p.StreamingChat(context.Background(),
		ChatRequest{Model: "m", MaxTokens: 100, Messages: []Message{userMsg("hi")}}, nil)
	if !IsOverContext(err) {
		t.Fatalf("上层拿不到超限标记就只能报错退出: %v", err)
	}
	if IsRetryable(err) {
		t.Error("超限不该被当成普通可重试错误")
	}
}

func TestStreamingChatRespectsContextCancel(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	p := NewDeepSeekProviderWithBaseURL(srv.URL, "k")
	fastRetry(p)

	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	_, err := p.StreamingChat(ctx,
		ChatRequest{Model: "m", MaxTokens: 100, Messages: []Message{userMsg("hi")}}, nil)
	if err == nil {
		t.Fatal("中断应返回错误")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Ctrl+C 的取消信号应原样透出，好让上层保留历史: %v", err)
	}
	if calls != 1 {
		t.Errorf("用户中断后不该再打第二次请求: calls=%d", calls)
	}
}

func TestAnthropicStreamingChatOverHTTP(t *testing.T) {
	var body string
	var auth, version, anthropicBeta string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		auth = r.Header.Get("x-api-key")
		version = r.Header.Get("anthropic-version")
		anthropicBeta = r.Header.Get("anthropic-beta")

		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sse(
			[2]string{"message_start", `{"type":"message_start","message":{"id":"msg_7","model":"claude-x"},"usage":{"input_tokens":42,"output_tokens":1}}`},
			[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"run_command"}}`},
			[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\": \"go vet ./...\"}"}}`},
			[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":18}}`},
			[2]string{"message_stop", `{"type":"message_stop"}`},
		))
	}))
	defer srv.Close()

	p := NewAnthropicProviderWithBaseURL(srv.URL, "secret")
	p.SetRetry(Retry{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})

	resp, err := p.StreamingChat(context.Background(), ChatRequest{
		Model:       "claude-x",
		MaxTokens:   512,
		System:      "规则",
		Messages:    []Message{userMsg("跑一下 vet")},
		RoundNumber: 1,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if auth != "secret" {
		t.Errorf("x-api-key = %q", auth)
	}
	if version != anthropicVersion {
		t.Errorf("anthropic-version = %q，缺失或不对会被服务端拒绝", version)
	}
	_ = anthropicBeta
	if resp.ID != "msg_7" || resp.Model != "claude-x" {
		t.Errorf("元信息丢失: %+v", resp)
	}
	if resp.FinishReason != FinishToolCalls {
		t.Errorf("FinishReason = %s，stop_reason 应映射为 tool_calls", resp.FinishReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Name != "run_command" {
		t.Fatalf("工具调用还原失败: %+v", resp.Content)
	}
	if resp.Content[0].Input["command"] != "go vet ./..." {
		t.Errorf("参数不对: %+v", resp.Content[0].Input)
	}
	if resp.Usage.InputTokens != 42 || resp.Usage.OutputTokens != 18 {
		t.Errorf("用量 = %+v", resp.Usage)
	}

	// system 走块数组而不是裸字符串：只有数组形式能带 cache_control 断点。
	if !strings.Contains(body, `"system":[{"type":"text","text":"规则"}]`) {
		t.Errorf("system 未按块数组下发: %s", body)
	}
	if !strings.Contains(body, `"max_tokens":512`) || !strings.Contains(body, `"stream":true`) {
		t.Errorf("请求体缺字段: %s", body)
	}
}
