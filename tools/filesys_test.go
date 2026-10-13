package tools

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// 只读工具不会改文件系统，但它们决定了模型看到什么：
// 分页、跳过二进制、跳过 .git，这些少了就会把上下文塞满或翻不进代码。

func seedTree(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"main.go":        "package main\n\nfunc main() {\n\trun()\n}\n",
		"pkg/svc.go":     "package pkg\n\nfunc run() {}\n",
		"README.md":      "# 标题\n\n这里提到 run 但不该被 *.go 匹配到\n",
		"note.txt":       "临时说明\n",
		".git/config":    "run 在版本控制目录里，不该被搜到\n",
		"node_modules/x": "run 在依赖目录里，不该被搜到\n",
		"big.bin":        "\x00\x01\x02 二进制含 run\n",
		"long_lines.go":  "package pkg\n// " + strings.Repeat("A", 900) + "\n",
		"many_lines.go":  strings.Repeat("x\n", 2500),
		"deep/a/b/c.go":  "package deep\nfunc deepFn() {}\n",
		"vendor/ignored": "run 在 vendor 里，不该被搜到\n",
	}
	for name, content := range files {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestSearchCode(t *testing.T) {
	dir := seedTree(t)
	s := NewSearchCodeTool(dir)

	out, err := s.Execute(context.Background(), map[string]any{"pattern": `func\s+run`})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "pkg/svc.go") || !strings.Contains(out, "func run() {}") {
		t.Errorf("没搜到已知内容: %q", out)
	}
	if !strings.Contains(out, ":3:") && !strings.Contains(out, ":5:") {
		t.Errorf("结果应带行号: %q", out)
	}

	// 全量搜 run：版本控制、依赖、二进制目录都不该被翻出来
	out, err = s.Execute(context.Background(), map[string]any{"pattern": "run"})
	if err != nil {
		t.Fatal(err)
	}
	for _, skip := range []string{".git/", "node_modules/", "vendor/", "big.bin"} {
		if strings.Contains(out, skip) {
			t.Errorf("%s 应被跳过，却出现在结果里:\n%s", skip, out)
		}
	}
	if !strings.Contains(out, "README.md") {
		t.Errorf("默认应搜所有文本文件:\n%s", out)
	}

	// file_pattern 限定后缀
	out, err = s.Execute(context.Background(), map[string]any{"pattern": "run", "file_pattern": "*.go"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "README.md") {
		t.Errorf("*.go 过滤没生效:\n%s", out)
	}

	// 指定子目录
	out, err = s.Execute(context.Background(), map[string]any{"pattern": "deepFn", "directory": "deep"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "deep/a/b/c.go") {
		t.Errorf("子目录搜索失败: %q", out)
	}

	// 无匹配不是错误
	out, err = s.Execute(context.Background(), map[string]any{"pattern": "根本不存在的东西"})
	if err != nil {
		t.Fatalf("无匹配应正常返回: %v", err)
	}
	if !strings.Contains(out, "未找到") {
		t.Errorf("应明确告诉模型没找到: %q", out)
	}

	// 坏正则要报错而不是静默返回空
	if _, err := s.Execute(context.Background(), map[string]any{"pattern": "(["}); err == nil {
		t.Error("非法正则应报错")
	}
	if _, err := s.Execute(context.Background(), map[string]any{}); err == nil {
		t.Error("缺少 pattern 应报错")
	}
	if _, err := s.Execute(context.Background(), map[string]any{"pattern": "run", "directory": ".."}); err == nil {
		t.Error("越界目录应报错")
	}
}

func TestListDirectory(t *testing.T) {
	dir := seedTree(t)
	l := NewListDirectoryTool(dir)

	out, err := l.Execute(context.Background(), map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "pkg") || !strings.Contains(out, "main.go") {
		t.Errorf("目录内容不全: %q", out)
	}
	// 目录在前、文件在后
	if strings.Index(out, "📁 deep") > strings.Index(out, "📄 main.go") {
		t.Error("应先列目录再列文件")
	}

	out, err = l.Execute(context.Background(), map[string]any{"directory": "pkg"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "svc.go") {
		t.Errorf("子目录列出失败: %q", out)
	}

	if _, err := l.Execute(context.Background(), map[string]any{"directory": "../"}); err == nil {
		t.Error("越界目录应报错")
	}
	if _, err := l.Execute(context.Background(), map[string]any{"directory": "nope"}); err == nil {
		t.Error("不存在的目录应报错")
	}
}

func TestReadFilePaging(t *testing.T) {
	dir := seedTree(t)
	r := NewReadFileTool(dir, NewFileState())

	out, err := r.Execute(context.Background(), map[string]any{"file_path": "main.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "     1 | package main") {
		t.Errorf("应带行号: %q", out)
	}

	// 超过默认上限要提示续读位置
	out, err = r.Execute(context.Background(), map[string]any{"file_path": "many_lines.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "还有 500 行未显示") || !strings.Contains(out, "start_line=2001") {
		t.Errorf("截断后要给出续读参数，否则模型只能重读整文件: %q", out[strings.LastIndex(out, "\n"):])
	}

	// 分段读取
	out, err = r.Execute(context.Background(), map[string]any{"file_path": "many_lines.go", "start_line": "2001", "max_lines": "10"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "  2001 |") || strings.Contains(out, "  2012 |") {
		t.Errorf("start_line/max_lines 没生效: %q", out[:200])
	}

	// 超长单行（压缩产物）不能把上下文吃满
	out, err = r.Execute(context.Background(), map[string]any{"file_path": "long_lines.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "已截断") {
		t.Error("超长行应截断并说明")
	}
	if len(out) > 1500 {
		t.Errorf("单行截断失效，返回了 %d 字节", len(out))
	}

	// 越界与缺失
	if _, err := r.Execute(context.Background(), map[string]any{"file_path": "missing.go"}); err == nil {
		t.Error("文件不存在应报错")
	}
	if _, err := r.Execute(context.Background(), map[string]any{"file_path": "main.go", "start_line": "9999"}); err == nil {
		t.Error("起始行超出文件长度应报错")
	}
	if _, err := r.Execute(context.Background(), map[string]any{"file_path": "/etc/passwd"}); err == nil {
		t.Error("绝对路径应报错")
	}
}

// 读取即登记：read_file 要把文件记进 FileState，后续写入才放行。
func TestReadMarksFileState(t *testing.T) {
	dir := seedTree(t)
	st := NewFileState()
	r := NewReadFileTool(dir, st)

	abs := filepath.Join(dir, "main.go")
	if st.HasSeen(abs) {
		t.Fatal("前置条件：读之前不该已知")
	}
	if _, err := r.Execute(context.Background(), map[string]any{"file_path": "main.go"}); err != nil {
		t.Fatal(err)
	}
	if !st.HasSeen(abs) {
		t.Error("read_file 没有把文件登记进 FileState")
	}
	if !st.Unchanged(abs) {
		t.Error("刚读过且没改动，应判定为未变")
	}
	if err := os.WriteFile(abs, []byte("package main\n// 外面改了\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if st.Unchanged(abs) {
		t.Error("文件在读取后被外部改过，应判定为已变")
	}
}

func TestPlanTool(t *testing.T) {
	var buf strings.Builder
	p := NewPlanTool(&buf)

	out, err := p.Execute(context.Background(), map[string]any{"plan": strings.Join([]string{
		"- [done] 读取现有实现",
		"- [doing] 补上重试",
		"- [todo] 写测试",
		"",
		"随手一行没有格式也算一步",
	}, "\n")})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "4 步") {
		t.Errorf("应统计 4 步: %q", out)
	}
	if !strings.Contains(out, "进度 1/4") {
		t.Errorf("进度统计不对: %q", out)
	}
	if !strings.Contains(buf.String(), "任务清单") {
		t.Error("清单应回显到终端")
	}

	steps := p.Steps()
	if len(steps) != 4 || steps[0] != "[done] 读取现有实现" || steps[3] != "[todo] 随手一行没有格式也算一步" {
		t.Errorf("Steps() = %v", steps)
	}

	// 每次调用提交完整清单，覆盖旧的
	if _, err := p.Execute(context.Background(), map[string]any{"plan": "- [x] 只剩一步"}); err != nil {
		t.Fatal(err)
	}
	if steps := p.Steps(); len(steps) != 1 || steps[0] != "[done] 只剩一步" {
		t.Errorf("新清单应覆盖旧的，实际 %v", steps)
	}

	if _, err := p.Execute(context.Background(), map[string]any{"plan": "   "}); err == nil {
		t.Error("空清单应报错")
	}
	if _, err := p.Execute(context.Background(), map[string]any{}); err == nil {
		t.Error("缺参数应报错")
	}
	if p.Mutating() {
		t.Error("update_plan 不该被当作改文件（否则每步都要用户确认）")
	}

	// 无回显目标时也不能 panic
	quiet := NewPlanTool(nil)
	if _, err := quiet.Execute(context.Background(), map[string]any{"plan": "- [todo] 一步"}); err != nil {
		t.Fatal(err)
	}
}

// 工具名与 Mutating 标记是审批门的依据，标错就等于放行了写操作。
func TestToolMutatingFlags(t *testing.T) {
	dir := t.TempDir()
	g := &Gate{Mode: ModeReadOnly, Out: io.Discard}
	st := NewFileState()

	readOnly := map[string]Tool{
		"read_file":      NewReadFileTool(dir, st),
		"list_directory": NewListDirectoryTool(dir),
		"search_code":    NewSearchCodeTool(dir),
		"update_plan":    NewPlanTool(io.Discard),
	}
	for name, tool := range readOnly {
		if tool.Mutating() {
			t.Errorf("%s 是只读工具，Mutating() 却报告 true", name)
		}
		if tool.Name() != name {
			t.Errorf("%s 的 Name() 返回 %q", name, tool.Name())
		}
	}

	writing := map[string]Tool{
		"write_file":  NewWriteFileTool(dir, g, st),
		"edit_file":   NewEditFileTool(dir, g, st),
		"move_file":   NewMoveFileTool(dir, g, st),
		"delete_file": NewDeleteFileTool(dir, g, st),
		"run_command": NewRunCommandTool(dir, g, 0, io.Discard),
	}
	for name, tool := range writing {
		if !tool.Mutating() {
			t.Errorf("%s 有副作用，Mutating() 却报告 false，审批门会漏过它", name)
		}
	}
}
