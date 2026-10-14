//go:build unix

package mcp

import (
	"os/exec"
	"syscall"
	"time"
)

// prepareProcessGroup 让服务器进程独占一个进程组。
//
// MCP 服务器常见形态是 npx/uvx 这类包装脚本，它再派生真正的服务进程。
// 不做隔离的话关掉父进程会留下孤儿，会话结束了还占着内存和文件描述符。
func prepareProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 3 * time.Second
}
