package tools

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go-agent/llm"
)

// 这些测试跑的是真工具、真文件系统（t.TempDir），只把审批门换成
// 无人值守的 ModeAuto / io.Discard，验证的是"门有没有真的挡住"，
// 而不是提示词里那句"请勿"。

func autoGate() *Gate { return &Gate{Mode: ModeAuto, Out: io.Discard} }

func newTestRegistry(t *testing.T) (*Registry, string) {
	t.Helper()
	dir := t.TempDir()
	g, st := autoGate(), NewFileState()
	r := NewRegistry()
	r.Register(NewReadFileTool(dir, st))
	r.Register(NewListDirectoryTool(dir))
	r.Register(NewSearchCodeTool(dir))
	r.Register(NewWriteFileTool(dir, g, st))
	r.Register(NewEditFileTool(dir, g, st))
	r.Register(NewMoveFileTool(dir, g, st))
	r.Register(NewDeleteFileTool(dir, g, st))
	r.Register(NewRunCommandTool(dir, g, 0, io.Discard))
	r.Register(NewPlanTool(io.Discard))
	return r, dir
}

// 工具定义位于请求前缀，顺序必须稳定，否则每次都要重算 prompt cache。
func TestRegistryDefinitionsAreSorted(t *testing.T) {
	r, _ := newTestRegistry(t)

	var first []string
	for i := 0; i < 50; i++ {
		var names []string
		for _, d := range r.Definitions() {
			names = append(names, d.Name)
		}
		joined := strings.Join(names, ",")
		if i == 0 {
			first = names
			continue
		}
		if joined != strings.Join(first, ",") {
			t.Fatalf("第 %d 次 Definitions 顺序变了: %s", i, joined)
		}
	}

	for i := 1; i < len(first); i++ {
		if first[i-1] > first[i] {
			t.Errorf("工具名未按字母序排列: %s 在 %s 之前", first[i-1], first[i])
		}
	}
	if len(first) != 9 {
		t.Errorf("应注册 9 个工具，实际 %d: %v", len(first), first)
	}
}

// 名字对不上会让模型调不到工具；重复注册会静默覆盖。两者都该在启动时炸掉。
type mismatchTool struct{ name string }

func (m mismatchTool) Name() string { return m.name }
func (m mismatchTool) Definition() llm.ToolDef {
	return llm.ToolDef{Name: "other"}
}
func (m mismatchTool) Execute(context.Context, map[string]any) (string, error) {
	return "", nil
}
func (m mismatchTool) Mutating() bool { return false }

func TestRegistryRegisterGuards(t *testing.T) {
	mustPanic := func(t *testing.T, want string, fn func()) {
		t.Helper()
		defer func() {
			rec := recover()
			if rec == nil {
				t.Fatalf("期望 panic 包含 %q，实际没有 panic", want)
			}
			if msg := fmt.Sprintf("%v", rec); !strings.Contains(msg, want) {
				t.Fatalf("panic = %q，期望包含 %q", msg, want)
			}
		}()
		fn()
	}

	mustPanic(t, "不一致", func() {
		NewRegistry().Register(mismatchTool{name: "mine"})
	})
	mustPanic(t, "已注册", func() {
		r, dir := NewRegistry(), t.TempDir()
		r.Register(NewReadFileTool(dir, NewFileState()))
		r.Register(NewReadFileTool(dir, NewFileState()))
	})
}

func TestWriteRequiresPriorRead(t *testing.T) {
	dir := t.TempDir()
	g, st := autoGate(), NewFileState()
	w := NewWriteFileTool(dir, g, st)
	r := NewReadFileTool(dir, st)

	// 磁盘上已有文件，本会话没读过 → 直接覆盖必须被拒
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte("old-content-keep-me"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := w.Execute(context.Background(), map[string]any{"file_path": "a.go", "content": "new"})
	var stale *StaleFileError
	if !errors.As(err, &stale) {
		t.Fatalf("未读过的文件应拒绝覆盖，实际错误: %v", err)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "a.go")); string(data) != "old-content-keep-me" {
		t.Fatal("被拒的写入仍然改了文件")
	}

	if _, err := r.Execute(context.Background(), map[string]any{"file_path": "a.go"}); err != nil {
		t.Fatalf("read_file 失败: %v", err)
	}
	out, err := w.Execute(context.Background(), map[string]any{"file_path": "a.go", "content": "new"})
	if err != nil {
		t.Fatalf("读过之后应允许覆盖: %v", err)
	}
	if !strings.Contains(out, "已覆盖") {
		t.Errorf("返回值应说明是覆盖: %q", out)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "a.go")); string(data) != "new" {
		t.Errorf("内容未更新: %q", data)
	}
	bak, err := os.ReadFile(filepath.Join(dir, "a.go.bak"))
	if err != nil || string(bak) != "old-content-keep-me" {
		t.Errorf("应留下 .bak 备份旧内容，读到 %q err %v", bak, err)
	}

	// 新文件不需要先读
	if _, err := w.Execute(context.Background(), map[string]any{"file_path": "b.go", "content": "package b"}); err != nil {
		t.Errorf("新建文件不该要求先读: %v", err)
	}
}

func TestEditFileMatching(t *testing.T) {
	dir := t.TempDir()
	g, st := autoGate(), NewFileState()
	e := NewEditFileTool(dir, g, st)
	r := NewReadFileTool(dir, st)

	content := "line one\nsame\nline three\nsame\n"
	if err := os.WriteFile(filepath.Join(dir, "c.go"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Execute(context.Background(), map[string]any{"file_path": "c.go"}); err != nil {
		t.Fatal(err)
	}

	// 歧义匹配：不指定哪一处就不改
	if _, err := e.Execute(context.Background(), map[string]any{
		"file_path": "c.go", "old_string": "same", "new_string": "other",
	}); err == nil || !strings.Contains(err.Error(), "2 次") {
		t.Errorf("出现两次的 old_string 应拒绝，实际: %v", err)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "c.go")); string(data) != content {
		t.Error("歧义匹配却改了文件")
	}

	// 带上下文的唯一匹配
	if _, err := e.Execute(context.Background(), map[string]any{
		"file_path": "c.go", "old_string": "line one\nsame", "new_string": "line 1\nsame",
	}); err != nil {
		t.Fatalf("唯一匹配应成功: %v", err)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "c.go")); !strings.HasPrefix(string(data), "line 1\nsame") {
		t.Errorf("替换结果不对: %q", data)
	}

	// replace_all
	out, err := e.Execute(context.Background(), map[string]any{
		"file_path": "c.go", "old_string": "same", "new_string": "both", "replace_all": "true",
	})
	if err != nil {
		t.Fatalf("replace_all 应成功: %v", err)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "c.go")); strings.Contains(string(data), "same") {
		t.Errorf("replace_all 没替换完: %q", data)
	}
	if !strings.Contains(out, "2 处") {
		t.Errorf("应报告替换了两处: %q", out)
	}

	// 找不到原文
	if _, err := e.Execute(context.Background(), map[string]any{
		"file_path": "c.go", "old_string": "不存在的内容", "new_string": "x",
	}); err == nil || !strings.Contains(err.Error(), "未找到") {
		t.Errorf("未找到时应报错并提示重新读取: %v", err)
	}

	// 没读过的文件不能 edit
	if err := os.WriteFile(filepath.Join(dir, "d.go"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Execute(context.Background(), map[string]any{
		"file_path": "d.go", "old_string": "x", "new_string": "y",
	}); err == nil {
		t.Error("未读过的文件应拒绝 edit")
	}
}

// 只读模式下，写/改/删/命令一律落不了地。
func TestReadOnlyGateBlocksEveryWrite(t *testing.T) {
	dir := t.TempDir()
	ro := &Gate{Mode: ModeReadOnly, Out: io.Discard}
	st := NewFileState()

	if err := os.WriteFile(filepath.Join(dir, "e.go"), []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	st.Mark(filepath.Join(dir, "e.go"))

	cases := []struct {
		name string
		run  func() error
	}{
		{"write_file", func() error {
			_, err := NewWriteFileTool(dir, ro, st).Execute(context.Background(),
				map[string]any{"file_path": "e.go", "content": "boom"})
			return err
		}},
		{"edit_file", func() error {
			_, err := NewEditFileTool(dir, ro, st).Execute(context.Background(),
				map[string]any{"file_path": "e.go", "old_string": "keep", "new_string": "boom"})
			return err
		}},
		{"delete_file", func() error {
			_, err := NewDeleteFileTool(dir, ro, st).Execute(context.Background(),
				map[string]any{"file_path": "e.go"})
			return err
		}},
		{"move_file", func() error {
			_, err := NewMoveFileTool(dir, ro, st).Execute(context.Background(),
				map[string]any{"from_path": "e.go", "to_path": "f.go"})
			return err
		}},
		{"run_command", func() error {
			_, err := NewRunCommandTool(dir, ro, 0, io.Discard).Execute(context.Background(),
				map[string]any{"command": "echo hi"})
			return err
		}},
	}

	for _, c := range cases {
		err := c.run()
		if !errors.Is(err, ErrDenied) {
			t.Errorf("%s 在只读模式下未被拒绝: %v", c.name, err)
		}
	}

	if data, _ := os.ReadFile(filepath.Join(dir, "e.go")); string(data) != "keep" {
		t.Error("只读模式下文件被改动")
	}
	if _, err := os.Stat(filepath.Join(dir, "e.go")); err != nil {
		t.Error("只读模式下文件被删除")
	}
}

func TestDeleteGuards(t *testing.T) {
	dir := t.TempDir()
	g, st := autoGate(), NewFileState()
	d := NewDeleteFileTool(dir, g, st)
	ctx := context.Background()

	if _, err := d.Execute(ctx, map[string]any{"file_path": "."}); err == nil {
		t.Error("应拒绝删除工作目录本身")
	}

	if err := os.MkdirAll(filepath.Join(dir, ".git", "objects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Execute(ctx, map[string]any{"file_path": ".git/objects"}); err == nil {
		t.Error("应拒绝删除 .git 内容")
	}
	if _, err := os.Stat(filepath.Join(dir, ".git", "objects")); err != nil {
		t.Error(".git/objects 被删掉了")
	}

	if err := os.MkdirAll(filepath.Join(dir, "tmpdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Execute(ctx, map[string]any{"file_path": "tmpdir"}); err == nil {
		t.Error("删目录应要求 recursive")
	}
	if _, err := os.Stat(filepath.Join(dir, "tmpdir")); err != nil {
		t.Error("未授权时目录被删了")
	}
	if _, err := d.Execute(ctx, map[string]any{"file_path": "tmpdir", "recursive": "true"}); err != nil {
		t.Errorf("recursive=true 应可删目录: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "tmpdir")); !os.IsNotExist(err) {
		t.Error("recursive 删除未生效")
	}
}

func TestMoveRefusesExistingTarget(t *testing.T) {
	dir := t.TempDir()
	m := NewMoveFileTool(dir, autoGate(), NewFileState())
	for _, n := range []string{"g.go", "h.go"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte(n), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := m.Execute(context.Background(), map[string]any{
		"from_path": "g.go", "to_path": "h.go",
	}); err == nil || !strings.Contains(err.Error(), "已存在") {
		t.Errorf("目标存在时应拒绝移动: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "g.go")); err != nil {
		t.Error("被拒的移动把源文件弄丢了")
	}
	if _, err := m.Execute(context.Background(), map[string]any{
		"from_path": "g.go", "to_path": "sub/g.go",
	}); err != nil {
		t.Errorf("移动到不存在的子目录应成功: %v", err)
	}
}

// 路径逃逸在工具层同样要拦住，不能只靠 resolveSafe 自己测。
func TestToolsRejectEscape(t *testing.T) {
	dir := t.TempDir()
	g, st := autoGate(), NewFileState()
	if _, err := NewWriteFileTool(dir, g, st).Execute(context.Background(),
		map[string]any{"file_path": "../evil.txt", "content": "x"}); err == nil {
		t.Error("write_file 允许了越界路径")
	}
	if _, err := NewReadFileTool(dir, st).Execute(context.Background(),
		map[string]any{"file_path": "/etc/passwd"}); err == nil {
		t.Error("read_file 允许了绝对路径")
	}
	if _, err := NewDeleteFileTool(dir, g, st).Execute(context.Background(),
		map[string]any{"file_path": "../evil.txt"}); err == nil {
		t.Error("delete_file 允许了越界路径")
	}
}
