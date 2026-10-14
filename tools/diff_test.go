package tools

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUnifiedDiffNoChange(t *testing.T) {
	if got := UnifiedDiff("a.go", "x\n", "x\n"); !strings.Contains(got, "没有实际改动") {
		t.Errorf("内容一致时应说明没有改动，得到:\n%s", got)
	}
}

func TestUnifiedDiffCreation(t *testing.T) {
	got := UnifiedDiff("new.go", "", "package main\n\nfunc main() {}\n")
	for _, want := range []string{"(文件尚不存在)", "+package main", "+func main() {}", "@@ -1,0 +1,3 @@"} {
		if !strings.Contains(got, want) {
			t.Errorf("新建文件的 diff 缺少 %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "-package") {
		t.Errorf("新建不该出现删除行:\n%s", got)
	}
}

func TestUnifiedDiffModifyKeepsContext(t *testing.T) {
	old := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	new := strings.Replace(old, "l5", "L5", 1)
	got := UnifiedDiff("f.txt", old, new)

	for _, want := range []string{" l4", "-l5", "+L5", " l6", "@@ -2,7 +2,7 @@"} {
		if !strings.Contains(got, want) {
			t.Errorf("改动行 %q 未在 diff 中；完整输出:\n%s", want, got)
		}
	}
	// 前后各 3 行上下文：l1 太远不该出现，l10 同理。
	if strings.Contains(got, " l1\n") || strings.Contains(got, " l10\n") {
		t.Errorf("上下文超出 %d 行:\n%s", diffContext, got)
	}
	// 改一行应当是"先删后加"，反过来读起来像新增。
	if strings.Index(got, "-l5") > strings.Index(got, "+L5") {
		t.Errorf("删除行应在新增行之前:\n%s", got)
	}
}

func TestUnifiedDiffDeletion(t *testing.T) {
	got := UnifiedDiff("f.txt", "a\nb\nc\n", "a\nc\n")
	if !strings.Contains(got, "-b") || strings.Contains(got, "+b") {
		t.Errorf("删除行没判对:\n%s", got)
	}
}

func TestUnifiedDiffMultipleHunks(t *testing.T) {
	var oldLines, newLines []string
	for i := 1; i <= 60; i++ {
		oldLines = append(oldLines, fmt.Sprintf("line%d", i))
		newLines = append(newLines, fmt.Sprintf("line%d", i))
	}
	newLines[2] = "CHANGED3"
	newLines[50] = "CHANGED51"

	got := UnifiedDiff("f.txt", strings.Join(oldLines, "\n")+"\n", strings.Join(newLines, "\n")+"\n")
	if n := strings.Count(got, "@@ "); n != 2 {
		t.Errorf("相距很远的两处改动应合成 2 个 hunk，实际 %d 个:\n%s", n, got)
	}
	for _, want := range []string{"-line3", "+CHANGED3", "-line51", "+CHANGED51"} {
		if !strings.Contains(got, want) {
			t.Errorf("缺少 %q", want)
		}
	}
}

func TestUnifiedDiffFallsBackWhenTooLarge(t *testing.T) {
	var oldBuf, newBuf strings.Builder
	for i := 0; i < diffMaxRegion+500; i++ {
		fmt.Fprintf(&oldBuf, "old%d\n", i)
		fmt.Fprintf(&newBuf, "new%d\n", i)
	}
	got := UnifiedDiff("big.txt", oldBuf.String(), newBuf.String())
	if !strings.Contains(got, "改动过大") {
		t.Errorf("超大改动应退化为统计信息:\n%s", got)
	}
	// 退化输出也必须是有上限的，不能把整份文件刷进终端。
	if lines := strings.Count(got, "\n"); lines > diffMaxOutLines+20 {
		t.Errorf("退化输出仍有 %d 行，未受上限约束", lines)
	}
}

func TestUnifiedDiffBinary(t *testing.T) {
	if got := UnifiedDiff("x.bin", "a\x00b", "a\x00c"); !strings.Contains(got, "二进制") {
		t.Errorf("二进制内容不应展开:\n%s", got)
	}
}

func TestUnifiedDiffLineNumbers(t *testing.T) {
	// 改动在第 7 行：hunk 头部要报出旧文件里的真实行号，
	// 用户才能按行号跳过去核对。
	var lines []string
	for i := 1; i <= 10; i++ {
		lines = append(lines, fmt.Sprintf("x%d", i))
	}
	changed := append([]string{}, lines...)
	changed[6] = "Y7"
	got := UnifiedDiff("f.txt", strings.Join(lines, "\n")+"\n", strings.Join(changed, "\n")+"\n")
	if !strings.Contains(got, "@@ -4,7 +4,7 @@") {
		t.Errorf("hunk 起点行号不对:\n%s", got)
	}
	if !strings.Contains(got, "-x7") || !strings.Contains(got, "+Y7") {
		t.Errorf("改动行未出现:\n%s", got)
	}
}

// TestWriteApprovalShowsDiff 是这一段的落点：确认框里必须带真 diff。
// 只报"覆盖 foo.go（N 字节）"等于让人盲签——门还在，签字的人没有信息。
func TestWriteApprovalShowsDiff(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "foo.go")
	if err := os.WriteFile(path, []byte("package a\n\nfunc Old() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var details []string
	gate := &Gate{Mode: ModeAsk, Out: io.Discard, Ask: func(_, detail string) bool {
		details = append(details, detail)
		return true
	}}
	state := NewFileState()
	state.Mark(path)

	tool := NewWriteFileTool(dir, gate, state)
	if _, err := tool.Execute(t.Context(), map[string]any{
		"file_path": "foo.go",
		"content":   "package a\n\nfunc New() {}\n",
	}); err != nil {
		t.Fatalf("写入失败: %v", err)
	}
	if len(details) != 1 {
		t.Fatalf("确认次数 = %d", len(details))
	}
	for _, want := range []string{"覆盖", "-func Old() {}", "+func New() {}", "@@ "} {
		if !strings.Contains(details[0], want) {
			t.Errorf("确认详情缺少 %q:\n%s", want, details[0])
		}
	}
}

// auto / readonly 模式不会问人，就不该为了生成 diff 去读整个文件。
func TestDiffSkippedWhenNoOneToAsk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "foo.go")
	if err := os.WriteFile(path, []byte("a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := NewFileState()
	state.Mark(path)
	tool := NewWriteFileTool(dir, autoGate(), state)
	out, err := tool.Execute(t.Context(), map[string]any{"file_path": "foo.go", "content": "b\n"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "@@") || strings.Contains(out, "-a") {
		t.Errorf("非交互模式的返回里不该带 diff:\n%s", out)
	}
}

func TestEditApprovalShowsDiff(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bar.txt")
	if err := os.WriteFile(path, []byte("one\ntwo\nthree\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var details []string
	gate := &Gate{Mode: ModeAsk, Out: io.Discard, Ask: func(_, detail string) bool {
		details = append(details, detail)
		return false // 拒绝执行：只关心确认框里展示了什么
	}}
	state := NewFileState()
	state.Mark(path)
	tool := NewEditFileTool(dir, gate, state)
	if _, err := tool.Execute(t.Context(), map[string]any{
		"file_path": "bar.txt", "old_string": "two", "new_string": "TWO",
	}); err == nil {
		t.Fatal("用户拒绝后不应写入成功")
	}
	if len(details) != 1 || !strings.Contains(details[0], "-two\n+TWO") {
		t.Fatalf("edit_file 的确认详情没有 diff:\n%v", details)
	}
	// 拒绝就是拒绝：文件必须还是原样。
	data, _ := os.ReadFile(path)
	if string(data) != "one\ntwo\nthree\n" {
		t.Errorf("被拒绝的改动落盘了: %q", data)
	}
}
