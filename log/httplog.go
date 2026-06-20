// Package log — HTTP 请求/响应文件日志器。
//
// HTTPLogger 将每次 LLM API 的完整请求体和响应体写入格式化日志文件，
// 方便调试和审计。日志文件存储在 ./logs/ 目录中，JSON 内容自动美化。
//
// 输出格式:
//
//	═══════════════════════════════════════════════════════════
//	🔼 REQUEST  [18:30:01.123]  Provider: deepseek  Round: 1
//	═══════════════════════════════════════════════════════════
//	POST https://api.deepseek.com/v1/chat/completions
//
//	{pretty-printed JSON 请求体}
//
//	───────────────────────────────────────────────────────────
//	🔽 RESPONSE  [18:30:03.456]  Duration: 2.3s
//	───────────────────────────────────────────────────────────
//	{pretty-printed JSON 响应体}
package log

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ============================================================================
// HTTPLogger
// ============================================================================

// HTTPLogger 将 HTTP 请求和响应的完整内容写入格式化日志文件。
type HTTPLogger struct {
	mu       sync.Mutex
	file     *os.File
	filePath string
	enabled  bool
}

// NewHTTPLogger 创建一个 HTTPLogger，日志写入 logDir 目录。
// 文件名格式: llm-http-20060102-150405.log
func NewHTTPLogger(logDir string) (*HTTPLogger, error) {
	if logDir == "" {
		logDir = "logs"
	}

	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, fmt.Errorf("创建日志目录失败: %w", err)
	}

	filename := filepath.Join(logDir, "llm-http-"+time.Now().Format("20060102-150405")+".log")
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("创建日志文件失败: %w", err)
	}

	hl := &HTTPLogger{
		file:     f,
		filePath: filename,
		enabled:  true,
	}

	// 写入文件头
	hl.writeHeader()

	fmt.Fprintf(os.Stderr, "📝 HTTP 日志文件: %s\n\n", filename)
	return hl, nil
}

// FilePath 返回日志文件路径。
func (hl *HTTPLogger) FilePath() string {
	return hl.filePath
}

// Enabled 返回是否已启用。
func (hl *HTTPLogger) Enabled() bool {
	return hl != nil && hl.enabled
}

// Close 关闭日志文件。
func (hl *HTTPLogger) Close() error {
	if hl == nil || hl.file == nil {
		return nil
	}
	hl.mu.Lock()
	defer hl.mu.Unlock()

	hl.writeSeparator("SESSION END")
	return hl.file.Close()
}

// ============================================================================
// 写入方法
// ============================================================================

// LogRequest 记录一次 API 请求。
func (hl *HTTPLogger) LogRequest(providerName, method, url string, round int, bodyJSON []byte) {
	if !hl.Enabled() {
		return
	}
	hl.mu.Lock()
	defer hl.mu.Unlock()

	hl.writeSeparator(fmt.Sprintf("🔼 REQUEST  [%s]  Provider: %s  Round: %d",
		time.Now().Format("15:04:05.000"), providerName, round))

	fmt.Fprintf(hl.file, "%s %s\n\n", method, url)
	hl.writePrettyJSON(bodyJSON)
	fmt.Fprintln(hl.file)
}

// LogResponse 记录一次 API 响应。
func (hl *HTTPLogger) LogResponse(providerName string, round int, elapsed time.Duration, bodyJSON []byte) {
	if !hl.Enabled() {
		return
	}
	hl.mu.Lock()
	defer hl.mu.Unlock()

	hl.writeSeparator(fmt.Sprintf("🔽 RESPONSE  [%s]  Duration: %.1fs  Round: %d",
		time.Now().Format("15:04:05.000"), elapsed.Seconds(), round))

	hl.writePrettyJSON(bodyJSON)
	fmt.Fprintln(hl.file)
	fmt.Fprintln(hl.file) // 额外空行分隔
}

// LogError 记录 API 错误。
func (hl *HTTPLogger) LogError(providerName string, round int, elapsed time.Duration, errMsg string, body []byte) {
	if !hl.Enabled() {
		return
	}
	hl.mu.Lock()
	defer hl.mu.Unlock()

	hl.writeSeparator(fmt.Sprintf("❌ ERROR  [%s]  Duration: %.1fs  Round: %d",
		time.Now().Format("15:04:05.000"), elapsed.Seconds(), round))

	fmt.Fprintf(hl.file, "Provider: %s\n", providerName)
	fmt.Fprintf(hl.file, "Error: %s\n\n", errMsg)
	if len(body) > 0 {
		fmt.Fprintf(hl.file, "Response body:\n%s\n\n", truncateBytes(body, 4000))
	}
}

// ============================================================================
// 内部方法
// ============================================================================

// writeSeparator 写入分隔行。
func (hl *HTTPLogger) writeSeparator(title string) {
	line := strings.Repeat("═", 68)
	fmt.Fprintf(hl.file, "%s\n", line)
	fmt.Fprintf(hl.file, "%s\n", title)
	fmt.Fprintf(hl.file, "%s\n\n", line)
}

// writeHeader 写入文件头。
func (hl *HTTPLogger) writeHeader() {
	fmt.Fprintf(hl.file, "╔══════════════════════════════════════════════════════════════╗\n")
	fmt.Fprintf(hl.file, "║  Go Agent — LLM HTTP 请求/响应日志                          ║\n")
	fmt.Fprintf(hl.file, "║  创建时间: %s                                  ║\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(hl.file, "╚══════════════════════════════════════════════════════════════╝\n\n")
}

// writePrettyJSON 美化 JSON 并写入文件。
func (hl *HTTPLogger) writePrettyJSON(raw []byte) {
	if len(raw) == 0 {
		fmt.Fprintln(hl.file, "(empty body)")
		return
	}

	// 尝试美化 JSON
	var obj any
	if err := json.Unmarshal(raw, &obj); err != nil {
		// 不是合法 JSON，直接写入原始内容（截断）
		fmt.Fprintln(hl.file, truncateBytes(raw, 8000))
		return
	}

	pretty, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		fmt.Fprintln(hl.file, truncateBytes(raw, 8000))
		return
	}

	// 截断过长的 JSON
	output := string(pretty)
	if len(output) > 32000 {
		output = output[:32000] + fmt.Sprintf("\n\n... (内容截断，完整大小 %d 字节)", len(pretty))
	}
	fmt.Fprintln(hl.file, output)
}

// truncateBytes 截断字节切片到指定长度。
func truncateBytes(data []byte, maxLen int) string {
	if len(data) <= maxLen {
		return string(data)
	}
	return string(data[:maxLen]) + fmt.Sprintf("\n... (截断，原始大小 %d 字节)", len(data))
}
