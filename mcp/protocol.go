// Package mcp 实现 MCP（Model Context Protocol）客户端。
//
// 只实现本智能体用得上的那一小块：newline-delimited JSON-RPC 2.0 over stdio、
// 工具发现（tools/list）与调用（tools/call）。resources、prompts、sampling、roots
// 一律不实现——服务器向我们发请求时回 method-not-found，而不是让它挂在那儿等。
//
// 传输抽象成 io.ReadWriteCloser：真正的实现是子进程的 stdin/stdout，
// 测试里换成 net.Pipe() 起来的内存假服务器。
package mcp

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// 协议版本与客户端标识。
const (
	protocolVersion = "2024-11-05"
	clientName      = "go-agent"
	clientVersion   = "0.2"
)

// JSON-RPC 错误码。
const (
	codeMethodNotFound = -32601
	codeInternalError  = -32603
)

// message 是 JSON-RPC 2.0 的一帧，请求/通知/响应合用一个结构，靠字段是否为空区分。
type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

func (m *message) isResponse() bool { return m.Method == "" && m.ID != nil }
func (m *message) isRequest() bool  { return m.Method != "" && m.ID != nil }

// RPCError 是 JSON-RPC 的错误对象。
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	s := fmt.Sprintf("远端错误 %d: %s", e.Code, e.Message)
	if len(e.Data) > 0 {
		s += ": " + strings.TrimSpace(string(e.Data))
	}
	return s
}

// initializeParams 是握手请求的参数。
type initializeParams struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ClientInfo      serverInfo     `json:"clientInfo"`
}

// initializeResult 是握手返回值。
type initializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      serverInfo     `json:"serverInfo"`
	Instructions    string         `json:"instructions,omitempty"`
}

type serverInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// Tool 是服务器上报的一个工具。
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Annotations *annotations    `json:"annotations,omitempty"`
}

type annotations struct {
	ReadOnlyHint    *bool `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool `json:"destructiveHint,omitempty"`
	OpenWorldHint   *bool `json:"openWorldHint,omitempty"`
}

// ReadOnly 报告服务器是否明确声明这个工具不改动任何东西。
//
// 默认值是"会改动"：MCP 规范里 readOnlyHint 缺省为 false、destructiveHint 缺省为 true，
// 而漏判一次的代价是本地文件被远端改掉。只有服务器亲口说只读、且没说破坏性，才算只读。
func (t Tool) ReadOnly() bool {
	if t.Annotations == nil || t.Annotations.ReadOnlyHint == nil || !*t.Annotations.ReadOnlyHint {
		return false
	}
	if t.Annotations.DestructiveHint != nil && *t.Annotations.DestructiveHint {
		return false
	}
	return true
}

// 分页拉取工具列表时的页数上限，防止服务器给出永不消失的 nextCursor 把循环钉住。
const maxListPages = 20

type listToolsParams struct {
	Cursor string `json:"cursor,omitempty"`
}

type listToolsResult struct {
	Tools      []Tool `json:"tools"`
	NextCursor string `json:"nextCursor,omitempty"`
}

type callToolParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

// callToolResult 是 tools/call 的返回值。
type callToolResult struct {
	Content []contentItem `json:"content"`
	IsError bool          `json:"isError"`
}

type contentItem struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// 回给模型的单次调用输出上限：图片之类的 base64 负载绝不该整段进上下文。
const maxCallOutputBytes = 60 * 1024

// flatten 把返回内容拼成一段文本。非文本部分只报"有什么"，不搬运数据。
func (r callToolResult) flatten() string {
	var parts []string
	for _, item := range r.Content {
		switch item.Type {
		case "text", "":
			parts = append(parts, item.Text)
		default:
			parts = append(parts, fmt.Sprintf("[非文本内容 type=%s mime=%s，%d 字节未展示]",
				item.Type, item.MimeType, len(item.Data)))
		}
	}
	out := strings.Join(parts, "\n")
	if len(out) > maxCallOutputBytes {
		total := len(out)
		out = clipUTF8(out, maxCallOutputBytes) +
			fmt.Sprintf("\n… 远端返回 %d 字节，已截断 …", total)
	}
	return out
}

// clipUTF8 在字节上限处截断，但退回到完整的字符边界——
// 切在多字节字符中间会让模型看到乱码。
func clipUTF8(s string, max int) string {
	if len(s) <= max {
		return s
	}
	for max > 0 && !utf8.RuneStart(s[max]) {
		max--
	}
	return s[:max]
}
