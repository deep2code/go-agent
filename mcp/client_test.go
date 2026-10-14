package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

// 协议行为全部在 net.Pipe 起来的内存假服务器上验证：
// 不 fork 进程、不碰真实 MCP 服务器，帧序和错误传递都能钉死。

// fakeServer 是管道另一端的假 MCP 服务器。
type fakeServer struct {
	routes map[string]func(message) *message

	mu     sync.Mutex
	frames []message // 客户端发来的每一帧，含通知
}

func (s *fakeServer) record(msg message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frames = append(s.frames, msg)
}

// received 返回客户端发出的所有帧的方法名，按到达顺序。
func (s *fakeServer) received() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.frames))
	for _, f := range s.frames {
		out = append(out, f.Method)
	}
	return out
}

func (s *fakeServer) saw(method string) bool {
	for _, m := range s.received() {
		if m == method {
			return true
		}
	}
	return false
}

// serve 从 serverEnd 读帧并回应，直到连接关闭。
func (s *fakeServer) serve(conn net.Conn) {
	dec := json.NewDecoder(conn)
	enc := json.NewEncoder(conn)
	for {
		var msg message
		if err := dec.Decode(&msg); err != nil {
			return
		}
		s.record(msg)
		if !msg.isRequest() {
			continue // 通知不回
		}

		var resp *message
		switch msg.Method {
		case "initialize":
			resp = result(map[string]any{
				"protocolVersion": protocolVersion,
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      serverInfo{Name: "fake-server", Version: "9.9"},
				"instructions":    "请先读 README",
			})
		default:
			if fn, ok := s.routes[msg.Method]; ok {
				resp = fn(msg)
			} else {
				resp = rpcErr(codeMethodNotFound, "no such method: "+msg.Method)
			}
		}
		if resp == nil {
			continue // 故意不回应，用于测超时与取消
		}
		resp.JSONRPC, resp.ID = "2.0", msg.ID
		if err := enc.Encode(resp); err != nil {
			return
		}
	}
}

// newTestClient 起假服务器并完成握手。
func newTestClient(t *testing.T, routes map[string]func(message) *message) (*Client, *fakeServer) {
	t.Helper()
	clientEnd, serverEnd := net.Pipe()
	srv := &fakeServer{routes: routes}
	go srv.serve(serverEnd)

	client, err := NewClient(t.Context(), "demo", clientEnd)
	if err != nil {
		t.Fatalf("握手失败: %v", err)
	}
	t.Cleanup(func() {
		client.Close()
		serverEnd.Close()
	})
	return client, srv
}

func result(v any) *message {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return &message{Result: data}
}

func rpcErr(code int, text string) *message {
	return &message{Error: &RPCError{Code: code, Message: text}}
}

func TestHandshakeReadsServerInfo(t *testing.T) {
	client, srv := newTestClient(t, nil)

	name, version, proto := client.Info()
	if name != "fake-server" || version != "9.9" || proto != protocolVersion {
		t.Errorf("服务器自述没拿到: %q %q %q", name, version, proto)
	}
	if client.Instructions() != "请先读 README" {
		t.Errorf("使用说明: %q", client.Instructions())
	}
	if client.Server() != "demo" {
		t.Errorf("配置名丢了: %q", client.Server())
	}
	// 规范要求的帧序：initialize 之后必须补一条 initialized 通知
	if !waitFor(t, 2*time.Second, func() bool { return srv.saw("notifications/initialized") }) {
		t.Fatalf("没发出 initialized 通知，收到的帧：%v", srv.received())
	}
	if got := strings.Join(srv.received(), ","); got != "initialize,notifications/initialized" {
		t.Errorf("握手帧序 = %q", got)
	}
}

func TestHandshakeErrorPropagates(t *testing.T) {
	clientEnd, serverEnd := net.Pipe()
	defer serverEnd.Close()
	go func() {
		dec := json.NewDecoder(serverEnd)
		enc := json.NewEncoder(serverEnd)
		var msg message
		if dec.Decode(&msg) != nil {
			return
		}
		resp := rpcErr(-32000, "服务器拒绝：未授权")
		resp.JSONRPC, resp.ID = "2.0", msg.ID
		enc.Encode(resp)
	}()

	_, err := NewClient(t.Context(), "demo", clientEnd)
	if err == nil || !strings.Contains(err.Error(), "未授权") {
		t.Fatalf("握手错误要原样带出来: %v", err)
	}
}

func TestHandshakeWithoutProtocolVersion(t *testing.T) {
	clientEnd, serverEnd := net.Pipe()
	defer serverEnd.Close()
	go func() {
		dec := json.NewDecoder(serverEnd)
		enc := json.NewEncoder(serverEnd)
		var msg message
		if dec.Decode(&msg) != nil {
			return
		}
		resp := result(map[string]any{"serverInfo": map[string]string{"name": "x"}})
		resp.JSONRPC, resp.ID = "2.0", msg.ID
		enc.Encode(resp)
	}()

	if _, err := NewClient(t.Context(), "demo", clientEnd); err == nil ||
		!strings.Contains(err.Error(), "protocolVersion") {
		t.Fatalf("服务器没自报版本时不能默认兼容: %v", err)
	}
}

func TestServerRequestGetsMethodNotFound(t *testing.T) {
	clientEnd, serverEnd := net.Pipe()
	defer serverEnd.Close()

	replied := make(chan message, 1)
	go func() {
		dec := json.NewDecoder(serverEnd)
		enc := json.NewEncoder(serverEnd)
		reply := func(msg message) {
			resp := result(map[string]any{
				"protocolVersion": protocolVersion,
				"serverInfo":      map[string]any{"name": "n", "version": "1"},
			})
			resp.JSONRPC, resp.ID = "2.0", msg.ID
			enc.Encode(resp)
		}

		var first message
		if dec.Decode(&first) != nil {
			return
		}
		reply(first)
		dec.Decode(&first) // initialized 通知

		id := int64(99)
		enc.Encode(message{JSONRPC: "2.0", ID: &id, Method: "sampling/createMessage"})
		var back message
		if dec.Decode(&back) != nil {
			return
		}
		replied <- back
	}()

	client, err := NewClient(t.Context(), "demo", clientEnd)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	select {
	case back := <-replied:
		if back.Error == nil || back.Error.Code != codeMethodNotFound {
			t.Errorf("服务器发来的请求应得到 method-not-found，得到 %+v", back)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("客户端没有回应服务器发来的请求，管道会一直挂在那儿")
	}
}

func TestToolsPagination(t *testing.T) {
	pages := 0
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/list": func(msg message) *message {
			pages++
			if pages == 1 {
				return result(map[string]any{
					"tools":      []Tool{{Name: "a", Description: "第一个"}},
					"nextCursor": "PAGE2",
				})
			}
			var p listToolsParams
			json.Unmarshal(msg.Params, &p)
			if p.Cursor != "PAGE2" {
				return rpcErr(codeInternalError, "第二页没带上 cursor")
			}
			return result(map[string]any{"tools": []Tool{{Name: "b"}}})
		},
	})

	list, err := client.Tools(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 || list[0].Name != "a" || list[1].Name != "b" {
		t.Fatalf("分页工具列表没有拼齐: %+v", list)
	}
	if pages != 2 {
		t.Errorf("应当恰好请求两页，实际 %d", pages)
	}
}

func TestToolsPaginationRunaway(t *testing.T) {
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/list": func(msg message) *message {
			return result(map[string]any{"tools": []Tool{{Name: "x"}}, "nextCursor": "again"})
		},
	})
	if _, err := client.Tools(t.Context()); err == nil ||
		!strings.Contains(err.Error(), "超过") {
		t.Fatalf("cursor 永不消失时不能无限翻页: %v", err)
	}
}

func TestCallToolFlattenAndTruncate(t *testing.T) {
	head := strings.Repeat("头", 1000)
	tail := "第二段"
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/call": func(msg message) *message {
			var p callToolParams
			if err := json.Unmarshal(msg.Params, &p); err != nil {
				return rpcErr(codeInternalError, "参数解析失败")
			}
			if p.Name != "echo" {
				return rpcErr(codeInternalError, "未知工具 "+p.Name)
			}
			if p.Arguments["q"] != "hi" {
				return rpcErr(codeInternalError, "参数没传过去")
			}
			return result(map[string]any{"content": []contentItem{
				{Type: "text", Text: head},
				{Type: "text", Text: tail},
				{Type: "image", MimeType: "image/png", Data: strings.Repeat("A", 500000)},
			}})
		},
	})

	out, err := client.CallTool(t.Context(), "echo", map[string]any{"q": "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, head) || !strings.Contains(out, tail) {
		t.Error("多个文本段没有按顺序拼起来")
	}
	if strings.Contains(out, strings.Repeat("A", 100)) {
		t.Error("图片的 base64 负载被搬进返回值了")
	}
	if !strings.Contains(out, "非文本内容 type=image") {
		t.Errorf("非文本部分没有说明: %.200s", out)
	}
}

func TestCallToolTruncatesHugeOutput(t *testing.T) {
	huge := strings.Repeat("数据", 40000) // 24 万字节
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/call": func(msg message) *message {
			return result(map[string]any{"content": []contentItem{{Type: "text", Text: huge}}})
		},
	})
	out, err := client.CallTool(t.Context(), "dump", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > maxCallOutputBytes+200 {
		t.Errorf("返回值没有截断，长度 %d", len(out))
	}
	if !strings.Contains(out, "已截断") {
		t.Errorf("没有说明被截断: %s", out[len(out)-200:])
	}
	// 截断必须落在字符边界上，否则模型看到的最后一格是乱码
	if !utf8.ValidString(out) {
		t.Error("截断切在了多字节字符中间")
	}
}

func TestCallToolIsErrorBecomesError(t *testing.T) {
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/call": func(msg message) *message {
			return result(map[string]any{
				"content": []contentItem{{Type: "text", Text: "查询超时了"}},
				"isError": true,
			})
		},
	})
	_, err := client.CallTool(t.Context(), "search", nil)
	if err == nil || !strings.Contains(err.Error(), "查询超时了") {
		t.Fatalf("isError 的内容要作为错误交回模型: %v", err)
	}
}

func TestCallToolRPCError(t *testing.T) {
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/call": func(msg message) *message { return rpcErr(-32602, "未知工具") },
	})
	_, err := client.CallTool(t.Context(), "nope", nil)
	if err == nil || !strings.Contains(err.Error(), "未知工具") ||
		!strings.Contains(err.Error(), "-32602") {
		t.Fatalf("JSON-RPC 的错误码和消息都要带出来: %v", err)
	}
}

func TestCallCancelledByContext(t *testing.T) {
	client, srv := newTestClient(t, map[string]func(message) *message{
		"tools/call": func(msg message) *message { return nil }, // 故意不回
		"tools/list": func(msg message) *message {
			return result(map[string]any{"tools": []Tool{{Name: "ok"}}})
		},
	})

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := client.CallTool(ctx, "slow", nil)
	if err == nil {
		t.Fatal("超时应当返回错误")
	}
	if !strings.Contains(err.Error(), "取消") {
		t.Errorf("错误要能看出是被取消: %v", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("没有按超时返回，等了 %v", time.Since(start))
	}

	// 取消后要补一条 notifications/cancelled，别让对方白算
	if !waitFor(t, 2*time.Second, func() bool { return srv.saw("notifications/cancelled") }) {
		t.Errorf("没有发出取消通知，收到的帧：%v", srv.received())
	}

	// 等待表清空后，同一连接还能继续用
	if _, err := client.Tools(t.Context()); err != nil {
		t.Errorf("取消一次调用不该弄坏连接: %v", err)
	}
}

func TestCloseWakesPendingCalls(t *testing.T) {
	client, _ := newTestClient(t, map[string]func(message) *message{
		"tools/call": func(msg message) *message { return nil },
	})

	go func() {
		time.Sleep(80 * time.Millisecond)
		client.Close()
	}()
	start := time.Now()
	_, err := client.CallTool(t.Context(), "slow", nil)
	if err == nil {
		t.Fatal("连接关闭后调用必须报错")
	}
	if !errors.Is(err, ErrClosed) {
		t.Errorf("错误要能看出连接已断开: %v", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("Close 没有唤醒等待中的调用，等了 %v", time.Since(start))
	}
}

func TestCallAfterClose(t *testing.T) {
	client, _ := newTestClient(t, nil)
	client.Close()
	if _, err := client.Tools(t.Context()); err == nil {
		t.Error("已关闭的连接不应接受新调用")
	}
	// 重复 Close 不该 panic
	_ = client.Close()
}

func TestReadLoopEOFMarksClosed(t *testing.T) {
	// 对端进程挂掉：读端拿到 EOF 后必须 fail 一次，把后续调用全部挡在门外。
	// 没有这一步，写进管道的请求永远不会有人回应，调用会一路挂到超时。
	clientEnd, serverEnd := net.Pipe()
	client := newClientOn("demo", clientEnd)
	defer client.Close()

	if err := serverEnd.Close(); err != nil {
		t.Fatal(err)
	}
	if !waitFor(t, time.Second, func() bool {
		_, err := client.Tools(t.Context())
		return err != nil
	}) {
		t.Fatal("对端关闭后调用仍然能挂住")
	}
	if _, err := client.Tools(t.Context()); !errors.Is(err, ErrClosed) {
		t.Errorf("错误要能认出是连接断开: %v", err)
	}
	if client.Err() == nil && !waitFor(t, time.Second, func() bool { return client.Err() != nil }) {
		t.Error("客户端没有记下断开原因")
	}
}

func TestToolReadOnlyAnnotation(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		name string
		anno *annotations
		want bool
	}{
		{"没有标注", nil, false},
		{"只读", &annotations{ReadOnlyHint: &yes}, true},
		{"显式非只读", &annotations{ReadOnlyHint: &no}, false},
		{"只读但带破坏性", &annotations{ReadOnlyHint: &yes, DestructiveHint: &yes}, false},
	}
	for _, c := range cases {
		if got := (Tool{Name: "x", Annotations: c.anno}).ReadOnly(); got != c.want {
			t.Errorf("%s: ReadOnly = %v，期望 %v", c.name, got, c.want)
		}
	}
}

func TestRPCErrorText(t *testing.T) {
	e := &RPCError{Code: -32601, Message: "没有这个工具", Data: json.RawMessage(`"detail"`)}
	got := e.Error()
	if !strings.Contains(got, "-32601") || !strings.Contains(got, "没有这个工具") || !strings.Contains(got, "detail") {
		t.Errorf("错误文本不完整: %q", got)
	}
}

// waitFor 轮询直到条件成立或超时。
func waitFor(t *testing.T, timeout time.Duration, cond func() bool) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}
