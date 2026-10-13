package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"

	"go-agent/log"
)

// 重试路径会打 WARN 日志，测试里只保留 ERROR，免得输出淹没断言。
func TestMain(m *testing.M) {
	log.SetLevel(log.ERROR)
	os.Exit(m.Run())
}

func TestRetryableStatus(t *testing.T) {
	yes := []int{408, 409, 429, 500, 502, 503, 504, statusOverloaded}
	no := []int{200, 400, 401, 403, 404, 413, 422}
	for _, s := range yes {
		if !retryableStatus(s) {
			t.Errorf("HTTP %d 应可重试", s)
		}
	}
	for _, s := range no {
		if retryableStatus(s) {
			t.Errorf("HTTP %d 不该重试——重试只会更快烧完配额", s)
		}
	}
}

func TestOverContextBody(t *testing.T) {
	yes := []string{
		`{"error":{"message":"prompt is too long: 213000 tokens > 200000 maximum"}}`,
		`{"error":{"message":"This model's maximum context length is 8192 tokens"}}`,
		`{"error":{"type":"invalid_request_error","code":"context_length_exceeded"}}`,
		`{"message":"Please reduce the length of the messages"}`,
	}
	for _, b := range yes {
		if !overContextBody([]byte(b)) {
			t.Errorf("应识别为上下文超限: %s", b)
		}
	}
	no := []string{
		`{"error":{"message":"invalid api key"}}`,
		`{"error":{"message":"rate limit exceeded"}}`,
	}
	for _, b := range no {
		if overContextBody([]byte(b)) {
			t.Errorf("不该判为上下文超限: %s", b)
		}
	}
}

func TestAPIErrorClassification(t *testing.T) {
	e := apiError(400, []byte("prompt is too long"), "参数错误", 0)
	if !IsOverContext(e) {
		t.Error("400 + too long 应标记为上下文超限")
	}
	if IsRetryable(e) {
		t.Error("上下文超限原样重试没有意义，应交给上层压缩历史")
	}

	if !IsRetryable(apiError(429, []byte("slow down"), "限流", 0)) {
		t.Error("429 应可重试")
	}
	if !IsRetryable(apiError(529, []byte("overloaded_error"), "过载", 0)) {
		t.Error("529 应可重试")
	}
	if IsRetryable(apiError(401, []byte("bad key"), "认证失败", 0)) {
		t.Error("认证失败重试无用")
	}

	// 非 400 的 413 即使文案像超限也不标，避免误触发历史压缩
	if IsOverContext(apiError(500, []byte("prompt is too long"), "服务器错误", 0)) {
		t.Error("500 不应标记为上下文超限")
	}

	var cause = errors.New("connection reset")
	if !IsRetryable(transportError(cause)) {
		t.Error("连接层失败应可重试")
	}
	if !errors.Is(transportError(cause), cause) {
		t.Error("transportError 应保留底层错误")
	}
}

func TestRetryAfter(t *testing.T) {
	cases := []struct {
		header string
		want   time.Duration
	}{
		{"5", 5 * time.Second},
		{" 12 ", 12 * time.Second},
		{"120", 60 * time.Second}, // 上限封顶，别让一个头把会话钉死
		{"", 0},
		{"abc", 0},
		{"-3", 0},
		{Wed, 0},
	}
	for _, c := range cases {
		h := http.Header{}
		if c.header != "" {
			h.Set("Retry-After", c.header)
		}
		if got := retryAfter(h); got != c.want {
			t.Errorf("Retry-After %q → %v，期望 %v", c.header, got, c.want)
		}
	}
}

const Wed = "Wed, 21 Oct 2015 07:28:00 GMT"

func TestRetryDelay(t *testing.T) {
	r := Retry{MaxAttempts: 4, BaseDelay: time.Second, MaxDelay: 20 * time.Second}

	// 抖动区间 [0.5d, 1.5d)
	for _, c := range []struct {
		attempt  int
		min, max time.Duration
	}{
		{1, 500 * time.Millisecond, 1500 * time.Millisecond},
		{2, time.Second, 3 * time.Second},
		{3, 2 * time.Second, 6 * time.Second},
		{4, 4 * time.Second, 12 * time.Second},
	} {
		for i := 0; i < 200; i++ {
			d := r.delay(c.attempt, 0)
			if d < c.min || d >= c.max {
				t.Fatalf("第 %d 次退避 %v 超出 [%v, %v)", c.attempt, d, c.min, c.max)
			}
		}
	}

	// 封顶
	if d := r.delay(50, 0); d > 30*time.Second {
		t.Errorf("退避未封顶: %v", d)
	}

	// 服务端给的等待时间优先
	if d := r.delay(1, 7*time.Second); d != 7*time.Second {
		t.Errorf("应采纳 Retry-After，实际 %v", d)
	}

	// 关闭重试时不等待
	if d := (Retry{MaxAttempts: 1, BaseDelay: time.Second}).delay(1, 0); d != 0 {
		t.Errorf("MaxAttempts=1 不该有退避: %v", d)
	}
}

func TestRetryDo(t *testing.T) {
	ctx := context.Background()
	fast := Retry{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond}
	retryable := apiError(429, []byte("slow down"), "限流", 0)
	fatal := apiError(401, []byte("bad key"), "认证失败", 0)

	t.Run("两次失败后成功", func(t *testing.T) {
		calls := 0
		err := fast.do(ctx, "test", 1, nil, func() error {
			calls++
			if calls < 3 {
				return retryable
			}
			return nil
		})
		if err != nil || calls != 3 {
			t.Errorf("err=%v calls=%d，期望重试到成功", err, calls)
		}
	})

	t.Run("用尽次数后返回原错误", func(t *testing.T) {
		calls := 0
		err := fast.do(ctx, "test", 1, nil, func() error {
			calls++
			return retryable
		})
		if calls != 3 {
			t.Errorf("应尝试 3 次，实际 %d", calls)
		}
		if !IsRetryable(err) && !errors.Is(err, retryable) {
			t.Errorf("最终错误应仍能看出原因: %v", err)
		}
	})

	t.Run("不可重试立即返回", func(t *testing.T) {
		calls := 0
		err := fast.do(ctx, "test", 1, nil, func() error {
			calls++
			return fatal
		})
		if calls != 1 || !errors.Is(err, fatal) {
			t.Errorf("认证错误不该重试: calls=%d err=%v", calls, err)
		}
	})

	t.Run("已经吐出文字就不再重试", func(t *testing.T) {
		calls := 0
		err := fast.do(ctx, "test", 1, func() bool { return true }, func() error {
			calls++
			return retryable
		})
		if calls != 1 {
			t.Errorf("流已经开头发内容，重试会让用户看到重复文本: calls=%d", calls)
		}
		if err == nil {
			t.Error("应把原始失败返回给上层")
		}
	})

	t.Run("用户中断不再重试", func(t *testing.T) {
		canceled, cancel := context.WithCancel(ctx)
		cancel()
		calls := 0
		err := fast.do(canceled, "test", 1, nil, func() error {
			calls++
			return retryable
		})
		if calls != 0 {
			t.Errorf("ctx 已取消却仍发起了调用: calls=%d", calls)
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("应返回取消错误: %v", err)
		}
	})

	t.Run("中断期间停止等待", func(t *testing.T) {
		timeoutCtx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
		defer cancel()
		slow := Retry{MaxAttempts: 5, BaseDelay: time.Hour, MaxDelay: time.Hour}
		start := time.Now()
		err := slow.do(timeoutCtx, "test", 1, nil, func() error { return retryable })
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("退避等待应被 ctx 打断: %v", err)
		}
		if elapsed := time.Since(start); elapsed > time.Second {
			t.Errorf("等待没有被打断，耗时 %v", elapsed)
		}
	})
}

func TestIsRetryableOnPlainError(t *testing.T) {
	if IsRetryable(errors.New("boom")) || IsOverContext(errors.New("boom")) {
		t.Error("普通错误不应被分类为重试或超限")
	}
	// 逐层包装后仍可识别
	wrapped := fmt.Errorf("调用 %s 失败: %w", "anthropic", apiError(503, nil, "服务不可用", 0))
	if !IsRetryable(wrapped) {
		t.Error("包装后的 *Error 应仍被识别")
	}
	over := fmt.Errorf("发送失败: %w", apiError(400, []byte("prompt is too long"), "参数错误", 0))
	if !IsOverContext(over) {
		t.Error("包装后的超限标记应仍被识别")
	}
}
