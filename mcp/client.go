package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// ErrClosed 表示连接已经断开（进程退出、管道关闭或主动 Close）。
var ErrClosed = errors.New("MCP 连接已断开")

// callOutcome 是等待中的调用拿到的东西：要么一帧响应，要么连接断开。
type callOutcome struct {
	msg message
	err error
}

// Client 是一个 MCP 服务器的连接。
//
// 一个后台 goroutine 独占读，把响应投递给等待中的调用；服务器主动发来的请求
// 一律回 method-not-found（我们不声明任何能力）。
type Client struct {
	server string
	rw     io.ReadWriteCloser

	writeMu sync.Mutex
	enc     *json.Encoder
	dec     *json.Decoder

	mu      sync.Mutex
	nextID  int64
	pending map[int64]chan callOutcome
	closed  bool
	err     error // 断开原因，closed 为 true 时非空

	done      chan struct{}
	closeOnce sync.Once

	info initializeResult
}

// newClientOn 接上传输并启动读端，但不握手——测试里用来单测断开行为。
func newClientOn(server string, rw io.ReadWriteCloser) *Client {
	c := &Client{
		server:  server,
		rw:      rw,
		enc:     json.NewEncoder(rw),
		dec:     json.NewDecoder(rw),
		pending: make(map[int64]chan callOutcome),
		done:    make(chan struct{}),
	}
	go c.readLoop()
	return c
}

// NewClient 在给定传输上完成 initialize 握手。
//
// 握手失败会关掉传输——调用方不需要再 Close 一次。
func NewClient(ctx context.Context, server string, rw io.ReadWriteCloser) (*Client, error) {
	c := newClientOn(server, rw)

	result, err := c.call(ctx, "initialize", initializeParams{
		ProtocolVersion: protocolVersion,
		Capabilities:    map[string]any{"tools": map[string]any{}},
		ClientInfo:      serverInfo{Name: clientName, Version: clientVersion},
	})
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("与 %s 握手失败: %w", server, err)
	}
	if err := json.Unmarshal(result, &c.info); err != nil {
		c.Close()
		return nil, fmt.Errorf("解析 %s 的握手结果失败: %w", server, err)
	}
	if c.info.ProtocolVersion == "" {
		c.Close()
		return nil, fmt.Errorf("%s 没有返回 protocolVersion，无法确认协议兼容", server)
	}

	// 规范要求握手后补一条通知，服务器据此开始正式服务
	if err := c.notify(ctx, "notifications/initialized", nil); err != nil {
		c.Close()
		return nil, fmt.Errorf("通知 %s 初始化完成失败: %w", server, err)
	}
	return c, nil
}

// Server 返回配置里的服务器名，用于拼工具名和展示来源。
func (c *Client) Server() string { return c.server }

// Info 返回握手拿到的服务器自述。
func (c *Client) Info() (name, version, protocol string) {
	return c.info.ServerInfo.Name, c.info.ServerInfo.Version, c.info.ProtocolVersion
}

// Instructions 是服务器给客户端的使用说明。
//
// 它和工具描述一样属于外部内容，只转述给人看，不拼进系统提示词——
// 否则等于让第三方服务器给模型下指令。
func (c *Client) Instructions() string { return c.info.Instructions }

// readLoop 独占读端。退出时唤醒所有等待中的调用。
func (c *Client) readLoop() {
	for {
		var msg message
		if err := c.dec.Decode(&msg); err != nil {
			c.fail(fmt.Errorf("%w: %v", ErrClosed, err))
			return
		}
		switch {
		case msg.isResponse():
			c.deliver(msg)
		case msg.isRequest():
			// 我们没声明能力，服务器不该发请求；必须回一个错误而不是沉默，
			// 否则它那边一直等，把整条管道拖死。
			c.write(message{JSONRPC: "2.0", ID: msg.ID,
				Error: &RPCError{Code: codeMethodNotFound, Message: "客户端未实现 " + msg.Method}})
		default:
			// 通知（如 logs/message、tools/list_changed），直接丢弃
		}
	}
}

// deliver 把响应交给对应的等待者。
func (c *Client) deliver(msg message) {
	c.mu.Lock()
	ch, ok := c.pending[*msg.ID]
	if ok {
		delete(c.pending, *msg.ID)
	}
	c.mu.Unlock()
	if !ok {
		return // 已经超时取消的调用，丢弃即可
	}
	ch <- callOutcome{msg: msg}
}

// fail 记下断开原因并唤醒等待者。重复调用只有第一次生效。
func (c *Client) fail(err error) {
	c.mu.Lock()
	first := !c.closed
	if first {
		c.closed = true
		c.err = err
	}
	waiters := make([]chan callOutcome, 0, len(c.pending))
	for id, ch := range c.pending {
		waiters = append(waiters, ch)
		delete(c.pending, id)
	}
	c.mu.Unlock()

	if first {
		close(c.done)
	}
	for _, ch := range waiters {
		// 每个等待者的通道都带缓冲，这里不会阻塞
		ch <- callOutcome{err: err}
	}
}

// Err 返回连接断开的原因，还活着时返回 nil。
func (c *Client) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

// write 输出一帧。
func (c *Client) write(msg message) error {
	if msg.JSONRPC == "" {
		msg.JSONRPC = "2.0"
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return c.enc.Encode(msg)
}

// call 发一个请求并等待响应。
func (c *Client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	raw, err := marshalParams(params)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, ErrClosed
	}
	c.nextID++
	id := c.nextID
	ch := make(chan callOutcome, 1)
	c.pending[id] = ch
	c.mu.Unlock()

	if err := c.write(message{JSONRPC: "2.0", ID: &id, Method: method, Params: raw}); err != nil {
		c.unregister(id)
		// 管道写不出去就是连接已经不能用了，把它归到 ErrClosed 里，
		// 上层才能用 errors.Is 区分"这台服务器掉了"和"这次调用失败"。
		return nil, fmt.Errorf("%w: 发给 %s 的 %s 没能写出: %v", ErrClosed, c.server, method, err)
	}

	select {
	case <-ctx.Done():
		c.unregister(id)
		// 尽力告知服务器这次调用不必再继续
		c.notify(context.WithoutCancel(ctx), "notifications/cancelled", map[string]any{
			"requestId": id, "reason": "客户端取消",
		})
		return nil, fmt.Errorf("%s 调用被取消: %w", method, ctx.Err())
	case out := <-ch:
		if out.err != nil {
			return nil, out.err
		}
		if out.msg.Error != nil {
			return nil, out.msg.Error
		}
		return out.msg.Result, nil
	}
}

func (c *Client) unregister(id int64) {
	c.mu.Lock()
	delete(c.pending, id)
	c.mu.Unlock()
}

// notify 发一条不需要回应的通知。
func (c *Client) notify(ctx context.Context, method string, params any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	raw, err := marshalParams(params)
	if err != nil {
		return err
	}
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed {
		return ErrClosed
	}
	return c.write(message{JSONRPC: "2.0", Method: method, Params: raw})
}

func marshalParams(params any) (json.RawMessage, error) {
	if params == nil {
		return nil, nil
	}
	data, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("参数序列化失败: %w", err)
	}
	return data, nil
}

// Tools 拉取服务器提供的工具列表（含分页）。
func (c *Client) Tools(ctx context.Context) ([]Tool, error) {
	var all []Tool
	cursor := ""
	for page := 0; page < maxListPages; page++ {
		result, err := c.call(ctx, "tools/list", listToolsParams{Cursor: cursor})
		if err != nil {
			return nil, err
		}
		var page1 listToolsResult
		if err := json.Unmarshal(result, &page1); err != nil {
			return nil, fmt.Errorf("解析 %s 的工具列表失败: %w", c.server, err)
		}
		all = append(all, page1.Tools...)
		if page1.NextCursor == "" {
			return all, nil
		}
		cursor = page1.NextCursor
	}
	return nil, fmt.Errorf("%s 的工具列表分页超过 %d 页，已停止", c.server, maxListPages)
}

// CallTool 调用一个远端工具，返回它的文本输出。
//
// 服务器标了 isError 时把输出内容作为错误返回：调用方会把它当成 tool_result 的
// is_error 交还模型，模型看得到失败原因，而不是拿到一句空字符串。
func (c *Client) CallTool(ctx context.Context, tool string, args map[string]any) (string, error) {
	result, err := c.call(ctx, "tools/call", callToolParams{Name: tool, Arguments: args})
	if err != nil {
		return "", fmt.Errorf("%s/%s 调用失败: %w", c.server, tool, err)
	}
	var parsed callToolResult
	if err := json.Unmarshal(result, &parsed); err != nil {
		return "", fmt.Errorf("解析 %s/%s 的返回失败: %w", c.server, tool, err)
	}
	text := parsed.flatten()
	if parsed.IsError {
		return "", fmt.Errorf("%s/%s 报告失败: %s", c.server, tool, text)
	}
	return text, nil
}

// Close 关掉传输并唤醒所有等待者。可以重复调用。
func (c *Client) Close() error {
	var err error
	c.closeOnce.Do(func() {
		closeErr := c.rw.Close()
		// 读端的 Decode 会因为关闭而报错退出；这里主动 fail 一次，
		// 让正在等待响应的调用立刻拿到 ErrClosed 而不是干等到超时。
		c.fail(fmt.Errorf("%w: 已主动关闭", ErrClosed))
		err = closeErr
	})
	return err
}
