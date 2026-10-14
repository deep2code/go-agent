package mcp

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

// StartServer 按配置拉起一个 MCP 服务器进程并完成握手。
//
// 进程放进独立进程组，Close 时整组结束：很多 MCP 服务器是 npx/uvx 起起来的包装脚本，
// 只杀父进程会留下一堆孤儿占着端口和内存。
func StartServer(ctx context.Context, cfg ServerConfig) (*Client, error) {
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	cmd := exec.CommandContext(runCtx, cfg.Command, cfg.Args...)
	cmd.Env = append(os.Environ(), envSlice(cfg.Env)...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("接入 %s 的 stdin 失败: %w", cfg.Name, err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("接入 %s 的 stdout 失败: %w", cfg.Name, err)
	}

	// stderr 只留最后若干字节：它是排查"服务器为什么没起来"的第一手材料，
	// 但不能让一条话痨服务器把内存吃光。
	stderr := &boundedBuf{limit: 8 * 1024}
	cmd.Stderr = stderr

	prepareProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("启动 MCP 服务器 %s 失败: %w", cfg.Name, err)
	}

	transport := &procTransport{
		cmd:    cmd,
		stdin:  stdin,
		stdout: stdout,
		cancel: cancel,
		stderr: stderr,
	}

	client, err := NewClient(ctx, cfg.Name, transport)
	if err != nil {
		transport.Close()
		return nil, withStderr(err, stderr)
	}
	return client, nil
}

// procTransport 把子进程的 stdin/stdout 包成 io.ReadWriteCloser。
type procTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	cancel context.CancelFunc
	stderr *boundedBuf

	closeOnce sync.Once
	waitErr   error
}

func (t *procTransport) Read(p []byte) (int, error)  { return t.stdout.Read(p) }
func (t *procTransport) Write(p []byte) (int, error) { return t.stdin.Write(p) }

// Close 结束子进程并回收。
//
// 顺序有讲究：先关 stdin（让干净的服务器自己退出），再用 cancel 触发整组 SIGKILL 兜底，
// 最后 Wait 回收，避免留下僵尸进程。
func (t *procTransport) Close() error {
	t.closeOnce.Do(func() {
		t.stdin.Close()
		t.cancel()
		t.waitErr = t.cmd.Wait()
	})
	return t.waitErr
}

func envSlice(env map[string]string) []string {
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	return out
}

// withStderr 把服务器打给 stderr 的内容拼进错误信息：
// "握手失败" 本身对用户没有可操作性，"npx: command not found" 才有。
func withStderr(err error, stderr *boundedBuf) error {
	text := strings.TrimSpace(stderr.String())
	if text == "" {
		return err
	}
	return fmt.Errorf("%w（服务器 stderr：%s）", err, text)
}

// boundedBuf 只保留最后 limit 字节的写入缓冲。
type boundedBuf struct {
	mu    sync.Mutex
	buf   bytes.Buffer
	limit int
}

func (b *boundedBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf.Write(p)
	if b.buf.Len() > b.limit {
		kept := b.buf.Bytes()[b.buf.Len()-b.limit:]
		nb := bytes.Buffer{}
		nb.Write(kept)
		b.buf = nb
	}
	return len(p), nil
}

func (b *boundedBuf) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
