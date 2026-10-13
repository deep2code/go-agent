package llm

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"

	"go-agent/log"
)

// ============================================================================
// Error — 带重试分类的 API 错误
// ============================================================================

// Error 表示一次 LLM API 调用失败。
// 除了给用户看的文案，还携带两个供上层决策的分类：
// 是否值得重试（限流、过载、网络抖动），是否属于上下文超限（需要压缩历史后重试）。
type Error struct {
	Status      int           // HTTP 状态码；网络层错误为 0
	Retryable   bool          // 是否可重试
	OverContext bool          // 是否因上下文过长被拒
	Message     string        // 面向用户的中文文案
	Cause       error         // 底层错误
	RetryAfter  time.Duration // 服务端建议的等待时间
}

func (e *Error) Error() string { return e.Message }

func (e *Error) Unwrap() error { return e.Cause }

// IsRetryable 判断错误是否值得重试。
func IsRetryable(err error) bool {
	var e *Error
	if errors.As(err, &e) {
		return e.Retryable
	}
	return false
}

// IsOverContext 判断错误是否由上下文超出模型窗口引起。
func IsOverContext(err error) bool {
	var e *Error
	if errors.As(err, &e) {
		return e.OverContext
	}
	return false
}

// retryableStatus 判断 HTTP 状态码是否可重试。
// 529 是 Anthropic 的过载码，不在 net/http 常量里。
const statusOverloaded = 529

func retryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout, statusOverloaded:
		return true
	default:
		return false
	}
}

// overContextBody 从错误响应体判断是否为上下文超限错误。
// 各家文案不同，只按关键词匹配：Anthropic 的 "prompt is too long"、
// OpenAI 兼容端的 "context_length_exceeded"。
func overContextBody(body []byte) bool {
	s := strings.ToLower(string(body))
	for _, kw := range []string{"too long", "context_length", "context length", "maximum context", "reduce the length"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

// retryAfter 解析 Retry-After 头，仅支持秒数形式。
func retryAfter(h http.Header) time.Duration {
	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && secs > 0 {
		if secs > 60 {
			secs = 60
		}
		return time.Duration(secs) * time.Second
	}
	return 0
}

// ============================================================================
// Retry — 指数退避 + 抖动
// ============================================================================

// Retry 重试策略。
type Retry struct {
	MaxAttempts int           // 含首次在内的总尝试次数
	BaseDelay   time.Duration // 首次退避时长
	MaxDelay    time.Duration // 单次退避上限
}

// DefaultRetry 返回默认重试策略。
func DefaultRetry() Retry {
	return Retry{MaxAttempts: 4, BaseDelay: time.Second, MaxDelay: 20 * time.Second}
}

// delay 计算第 attempt 次失败后的等待时长：指数增长 + ±25% 抖动，
// 服务端给了 Retry-After 时以它为准。
func (r Retry) delay(attempt int, hint time.Duration) time.Duration {
	if hint > 0 {
		return hint
	}
	if r.MaxAttempts <= 1 || r.BaseDelay <= 0 {
		return 0
	}
	d := r.BaseDelay
	for i := 1; i < attempt; i++ {
		d *= 2
		if d > r.MaxDelay {
			d = r.MaxDelay
			break
		}
	}
	// 抖动到 [0.5d, 1.5d)，避免多实例同时重试
	half := d / 2
	return half + time.Duration(rand.Int63n(int64(half)+1))
}

// do 执行 fn，遇到可重试错误时退避重试。
//
// started 用于报告本轮是否已向用户输出过文本增量：一旦输出过就不再重试，
// 否则重试会在终端上留下重复内容。
func (r Retry) do(ctx context.Context, provider string, round int, started func() bool, fn func() error) error {
	var lastErr error
	for attempt := 1; attempt <= r.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		lastErr = fn()
		if lastErr == nil {
			return nil
		}
		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			return lastErr
		}
		if !IsRetryable(lastErr) || attempt == r.MaxAttempts || (started != nil && started()) {
			return lastErr
		}

		wait := r.delay(attempt, retryAfterHint(lastErr))
		log.Warn("%s 第 %d 轮调用失败（第 %d/%d 次尝试），%s 后重试: %v",
			provider, round, attempt, r.MaxAttempts, wait.Round(time.Millisecond), lastErr)

		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("重试 %d 次后仍失败: %w", r.MaxAttempts, lastErr)
}

// retryAfterHint 取出错误里携带的服务端等待建议。
func retryAfterHint(err error) time.Duration {
	var e *Error
	if errors.As(err, &e) {
		return e.RetryAfter
	}
	return 0
}

// ============================================================================
// 构造 Provider 共用的错误
// ============================================================================

// apiError 依据状态码与响应体构造带分类的 API 错误，msg 为面向用户的文案。
func apiError(status int, body []byte, msg string, hint time.Duration) *Error {
	return &Error{
		Status:      status,
		Retryable:   retryableStatus(status),
		OverContext: status == http.StatusBadRequest && overContextBody(body),
		Message:     msg,
		RetryAfter:  hint,
	}
}

// transportError 包装连接层失败（DNS、TLS、连接被重置等），一律视为可重试。
func transportError(err error) *Error {
	return &Error{
		Retryable: true,
		Message:   fmt.Sprintf("网络请求失败: %v", err),
		Cause:     err,
	}
}

// streamError 包装流式读取中途失败。若尚未吐出任何文本，可安全重试。
func streamError(err error) *Error {
	return &Error{
		Retryable: true,
		Message:   fmt.Sprintf("读取响应流失败: %v", err),
		Cause:     err,
	}
}
