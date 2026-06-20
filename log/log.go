// Package log 提供带文件:行号定位的执行日志系统。
//
// 通过 runtime.Caller() 自动获取调用者的源文件和行号，
// 输出格式为: [时间] [级别] 文件:行号 消息
//
// 支持四个日志级别: DEBUG（灰色）、INFO（白色）、WARN（黄色）、ERROR（红色）。
//
// 用法:
//
//	log.Info("操作完成, 耗时 %v", elapsed)
//	log.Debug("请求体: %+v", req)
//	log.Warn("工具返回空结果")
//	log.Error("API 调用失败: %v", err)
//
// 输出示例:
//
//	[15:04:05.123] [INFO ] agent/agent.go:127  开始 ReAct 循环第 3 轮
//	[15:04:05.456] [DEBUG] agent/agent.go:194  调用 tool.Execute(read_file)
//	[15:04:05.789] [WARN ] tools/run_command.go:88  命令执行超时(30s)
package log

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// ============================================================================
// 日志级别
// ============================================================================

// Level 日志级别。
type Level int

const (
	DEBUG Level = iota
	INFO
	WARN
	ERROR
)

var levelNames = map[Level]string{
	DEBUG: "DEBUG",
	INFO:  "INFO",
	WARN:  "WARN",
	ERROR: "ERROR",
}

// String 实现 fmt.Stringer 接口。
func (l Level) String() string {
	if name, ok := levelNames[l]; ok {
		return name
	}
	return "UNKNOWN"
}

// ANSI 颜色码
const (
	colorReset  = "\033[0m"
	colorGray   = "\033[90m"
	colorWhite  = "\033[97m"
	colorYellow = "\033[93m"
	colorRed    = "\033[91m"
)

var levelColors = map[Level]string{
	DEBUG: colorGray,
	INFO:  colorWhite,
	WARN:  colorYellow,
	ERROR: colorRed,
}

// ============================================================================
// 全局配置
// ============================================================================

var (
	currentLevel = INFO
	mu           sync.Mutex
)

// SetLevel 设置全局日志级别。只有 >= 当前级别的日志才会输出。
func SetLevel(level Level) {
	mu.Lock()
	defer mu.Unlock()
	currentLevel = level
}

// GetLevel 返回当前日志级别。
func GetLevel() Level {
	mu.Lock()
	defer mu.Unlock()
	return currentLevel
}

// ParseLevel 从字符串解析日志级别，不区分大小写。
// 支持: "debug", "info", "warn", "error"
func ParseLevel(s string) Level {
	switch strings.ToLower(s) {
	case "debug":
		return DEBUG
	case "info":
		return INFO
	case "warn", "warning":
		return WARN
	case "error":
		return ERROR
	default:
		return INFO
	}
}

// ============================================================================
// 公共输出函数
// ============================================================================

// Debug 输出 DEBUG 级别日志。仅在 LOG_LEVEL=debug 时可见。
func Debug(format string, args ...any) {
	log(DEBUG, format, args...)
}

// Info 输出 INFO 级别日志。标记关键执行步骤。
func Info(format string, args ...any) {
	log(INFO, format, args...)
}

// Warn 输出 WARN 级别日志。标记可恢复的问题或异常。
func Warn(format string, args ...any) {
	log(WARN, format, args...)
}

// Error 输出 ERROR 级别日志。标记错误情况。
func Error(format string, args ...any) {
	log(ERROR, format, args...)
}

// ============================================================================
// 内部实现
// ============================================================================

// log 核心日志输出函数。使用 runtime.Caller(2) 跳过本地调用栈。
func log(level Level, format string, args ...any) {
	mu.Lock()
	lvl := currentLevel
	mu.Unlock()

	if level < lvl {
		return
	}

	// 获取调用者文件:行号
	// 调用栈: log() → Debug/Info/Warn/Error() → 实际调用者
	// 所以 skip = 2
	_, file, line, ok := runtime.Caller(2)
	if !ok {
		file = "???"
		line = 0
	}

	// 截断文件路径，只保留最后 3 级（如 agent/agent.go 或 llm/provider_deepseek.go）
	file = trimPath(file, 3)

	// 格式化消息
	msg := fmt.Sprintf(format, args...)

	// 时间戳（仅时:分:秒.毫秒）
	now := time.Now()
	timestamp := now.Format("15:04:05") + fmt.Sprintf(".%03d", now.Nanosecond()/1e6)

	// 获取颜色
	color := levelColors[level]
	name := levelNames[level]

	// 输出格式: [时间] [级别] 文件:行号 消息
	fmt.Fprintf(os.Stderr, "%s[%s] [%-5s] %s:%-4d  %s%s\n",
		color, timestamp, name, file, line, msg, colorReset)
}

// trimPath 截断文件路径，只保留最后 n 级。
// 例如: /Users/junjunyi/src-code/go-agent/agent/agent.go (n=3) → agent/agent.go
func trimPath(path string, keepLevels int) string {
	// 标准化路径分隔符
	path = filepath.ToSlash(path)
	parts := strings.Split(path, "/")
	if len(parts) > keepLevels {
		parts = parts[len(parts)-keepLevels:]
	}
	return strings.Join(parts, "/")
}
