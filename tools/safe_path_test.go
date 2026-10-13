package tools

import (
	"os"
	"path/filepath"
	"testing"
)

// 三种逃逸写法都要拦住：绝对路径、.. 穿越、软链接指向外部。
// 这里测的是纯路径解析，不碰审批门。

func TestResolveSafeRejectsOutside(t *testing.T) {
	workingDir := t.TempDir()

	cases := []struct {
		name string
		path string
	}{
		{"empty", ""},
		{"absolute", "/etc/passwd"},
		{"parent", ".."},
		{"parent file", "../secret.txt"},
		{"deep traversal", "a/../../etc/passwd"},
		{"mixed", "a/b/../../../etc/hosts"},
		{"null byte", "ok\x00.txt"},
	}
	for _, c := range cases {
		if _, err := resolveSafe(workingDir, c.path, false); err == nil {
			t.Errorf("%s: resolveSafe(%q) 应当拒绝，却成功了", c.name, c.path)
		}
	}
}

func TestResolveSafeAcceptsInside(t *testing.T) {
	workingDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(workingDir, "src", "inner"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workingDir, "src", "inner", "a.go"), []byte("package a"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, rel := range []string{".", "src", "src/inner/a.go", "src/inner/../inner/a.go"} {
		got, err := resolveSafe(workingDir, rel, false)
		if err != nil {
			t.Errorf("%q 应被接受: %v", rel, err)
			continue
		}
		base, _ := filepath.Abs(workingDir)
		if filepath.Clean(got) != filepath.Clean(filepath.Join(base, rel)) {
			t.Errorf("%q 解析成了 %q", rel, got)
		}
	}
}

// 工作目录内部的软链接指向外部目录：Clean 之后看还在目录内，
// 必须靠 EvalSymlinks 才拦得住。
func TestResolveSafeRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	workingDir := filepath.Join(root, "work")
	outside := filepath.Join(root, "outside")
	if err := os.MkdirAll(workingDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "passwd"), []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(workingDir, "link")); err != nil {
		t.Skipf("当前环境不支持软链接: %v", err)
	}

	if _, err := resolveSafe(workingDir, "link/passwd", false); err == nil {
		t.Fatal("软链接指向工作目录外，resolveSafe 却放行了")
	}
	// 写场景同样不能被绕过
	if _, err := resolveSafe(workingDir, "link/new.txt", true); err == nil {
		t.Fatal("allowMissing 下软链接逃逸仍然放行")
	}
}

// 写入允许目标不存在，但不允许父目录逃出去。
func TestResolveSafeMissingTargets(t *testing.T) {
	workingDir := t.TempDir()

	got, err := resolveSafe(workingDir, "new/deep/file.txt", true)
	if err != nil {
		t.Fatalf("新建嵌套文件应被接受: %v", err)
	}
	base, _ := filepath.Abs(workingDir)
	if want := filepath.Join(base, "new/deep/file.txt"); got != want {
		t.Errorf("got %q want %q", got, want)
	}

	if _, err := resolveSafe(workingDir, "new/../..", true); err == nil {
		t.Error("allowMissing 下穿越仍应被拒绝")
	}
	if _, err := resolveSafe(workingDir, "a.go", false); err == nil {
		t.Error("读取不存在的文件应报错")
	}
}

func TestIsUnder(t *testing.T) {
	cases := []struct {
		path, base string
		want       bool
	}{
		{"/w/a", "/w", true},
		{"/w/a/b", "/w", true},
		{"/w", "/w", false},
		{"/wx/a", "/w", false}, // 前缀相同但不是子目录
		{"/other/a", "/w", false},
	}
	for _, c := range cases {
		if got := isUnder(c.path, c.base); got != c.want {
			t.Errorf("isUnder(%q, %q) = %v want %v", c.path, c.base, got, c.want)
		}
	}
}
