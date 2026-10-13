//go:build unix

package tools

import (
	"os/exec"
	"syscall"
	"time"
)

// useProcessGroup 让命令独占一个进程组，超时或中断时整组杀掉。
//
// 不做这件事的话，kill 只会干掉 bash 本身，它派生的编译、测试进程会继续
// 占着 CPU，并且管道不会关闭，Run 也就不会返回。
func useProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 3 * time.Second // 超时后最多再等 3 秒收尾
}
