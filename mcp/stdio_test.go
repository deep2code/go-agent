package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// 这一组测的是"真的起一个进程"那一段：管道 framing、env 传入、stderr 回显、
// Close 之后进程确实被收走。协议本身在 client_test.go 用内存管道覆盖。

const fakeServerEnv = "GO_AGENT_FAKE_MCP_SERVER"
const fakePidFileEnv = "GO_AGENT_FAKE_MCP_PIDFILE"
const fakeTokenEnv = "GO_AGENT_FAKE_MCP_TOKEN"

// serveFakeStdio 让当前进程充当一个最小的 MCP 服务器。
//
// 它由测试自己充当（同一个二进制带环境变量重跑），所以不需要外部命令，
// 也不依赖机器上有没有 npx / python。
func serveFakeStdio() {
	pidFile := os.Getenv(fakePidFileEnv)
	if pidFile != "" {
		// 把自己的 pid 落盘，父进程 Close 之后据此确认真被回收了
		_ = os.WriteFile(pidFile, []byte(strconv.Itoa(os.Getpid())), 0o644)
	}

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1024*1024), 8*1024*1024)
	out := os.Stdout
	for in.Scan() {
		var msg message
		if err := json.Unmarshal(in.Bytes(), &msg); err != nil {
			continue
		}
		if !msg.isRequest() {
			continue
		}
		var resp *message
		switch msg.Method {
		case "initialize":
			resp = result(map[string]any{
				"protocolVersion": protocolVersion,
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      serverInfo{Name: "stdio-fake", Version: "1.0"},
			})
		case "tools/list":
			resp = result(map[string]any{"tools": []Tool{{
				Name:        "whoami",
				Description: "返回收到的 token",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
			}}})
		case "tools/call":
			resp = result(map[string]any{"content": []contentItem{
				{Type: "text", Text: "token=" + os.Getenv(fakeTokenEnv)},
			}})
		default:
			resp = rpcErr(codeMethodNotFound, msg.Method)
		}
		resp.JSONRPC, resp.ID = "2.0", msg.ID
		data, _ := json.Marshal(resp)
		out.Write(append(data, '\n'))
	}
	os.Exit(0)
}

func TestStartServerOverStdio(t *testing.T) {
	if os.Getenv(fakeServerEnv) == "1" {
		serveFakeStdio() // 子进程模式：服务完就退出，不跑测试逻辑
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	pidFile := t.TempDir() + "/pid"

	cfg := ServerConfig{
		Name:    "self",
		Command: exe,
		Args:    []string{"-test.run", "^TestStartServerOverStdio$", "-test.timeout", "60s"},
		Env: map[string]string{
			fakeServerEnv:  "1",
			fakePidFileEnv: pidFile,
			fakeTokenEnv:   "abc-from-parent-env",
		},
		Timeout: 20 * time.Second,
	}

	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	client, err := StartServer(ctx, cfg)
	if err != nil {
		t.Fatalf("启动子进程服务器失败: %v", err)
	}

	name, _, _ := client.Info()
	if name != "stdio-fake" {
		t.Errorf("服务器自述 = %q", name)
	}

	tools, err := client.Tools(ctx)
	if err != nil || len(tools) != 1 || tools[0].Name != "whoami" {
		t.Fatalf("工具列表: %+v, %v", tools, err)
	}
	if string(tools[0].InputSchema) == "" {
		t.Error("远端 schema 没有原样带回来")
	}

	out, err := client.CallTool(ctx, "whoami", map[string]any{"q": "x"})
	if err != nil {
		t.Fatal(err)
	}
	// 配置的 env 真的进了子进程环境，而不是只停在配置里
	if out != "token=abc-from-parent-env" {
		t.Errorf("调用结果 = %q", out)
	}

	if err := client.Close(); err != nil {
		t.Logf("Close 返回（子进程被信号终止属正常）: %v", err)
	}

	// 进程必须真的被收走：留下孤儿的话，每次会话都会挂着一个 npx
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("子进程没有记下 pid: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatalf("pid 文件内容异常 %q: %v", data, err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if err := syscall.Kill(pid, 0); err != nil {
			return // ESRCH：进程已经不在了
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("Close 之后 pid %d 仍然活着", pid)
}

func TestStartServerFailureShowsStderr(t *testing.T) {
	// 服务器起不来时，stderr 是唯一能看出原因的线索
	cfg := ServerConfig{
		Name:    "broken",
		Command: "/bin/sh",
		Args:    []string{"-c", "echo 'npx: command not found' 1>&2; exit 1"},
		Timeout: 5 * time.Second,
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	_, err := StartServer(ctx, cfg)
	if err == nil {
		t.Fatal("进程退出后握手必须失败")
	}
	if !strings.Contains(err.Error(), "command not found") {
		t.Errorf("错误里要带上服务器的 stderr: %v", err)
	}
}

func TestStartServerMissingBinary(t *testing.T) {
	cfg := ServerConfig{Name: "ghost", Command: "/nonexistent/mcp-server", Timeout: time.Second}
	_, err := StartServer(t.Context(), cfg)
	if err == nil || !strings.Contains(err.Error(), "启动 MCP 服务器") {
		t.Fatalf("找不到可执行文件时要报错: %v", err)
	}
}

func TestBoundedBufKeepsTail(t *testing.T) {
	b := &boundedBuf{limit: 10}
	b.Write([]byte("0123456789"))
	b.Write([]byte("abcdef"))
	if got := b.String(); got != "6789abcdef" || len(got) != b.limit {
		t.Errorf("只应保留最后 10 字节，得到 %q", got)
	}
	// 话痨服务器不能把内存吃光：反复写入后长度仍封顶
	for i := 0; i < 1000; i++ {
		b.Write([]byte("xxxxxxxxxxxxxxxxxxxx"))
	}
	if got := b.String(); len(got) != b.limit {
		t.Errorf("写了 1000 次后长度 = %d，应为 %d", len(got), b.limit)
	}
}
