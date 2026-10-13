//go:build !unix

package tools

import "os/exec"

// useProcessGroup 在非 Unix 平台上不做进程组隔离，依赖 Go 自身的 kill 行为。
func useProcessGroup(cmd *exec.Cmd) {}
