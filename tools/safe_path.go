package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ============================================================================
// 路径安全
// ============================================================================

// resolveSafe 把模型给出的相对路径解析到工作目录内，并拒绝任何逃逸。
//
// 三重校验：
//  1. 拒绝绝对路径；
//  2. filepath.Clean 后仍必须落在工作目录内，拦住的士 `a/../../etc` 这类写法；
//  3. EvalSymlinks 后必须落在真实工作目录内，拦住工作目录里指向外部的软链接。
//
// allowMissing 为 true 时允许目标尚不存在（写入场景）：此时解析已存在的最近祖先，
// 保证软链接逃逸在创建文件之前就被拦下。
func resolveSafe(workingDir, relPath string, allowMissing bool) (string, error) {
	if relPath == "" {
		return "", fmt.Errorf("路径不能为空")
	}
	if filepath.IsAbs(relPath) {
		return "", fmt.Errorf("不允许使用绝对路径: %s", relPath)
	}
	if strings.ContainsRune(relPath, 0) {
		return "", fmt.Errorf("路径含非法字符")
	}

	absWorkingDir, err := filepath.Abs(workingDir)
	if err != nil {
		return "", fmt.Errorf("无法解析工作目录: %w", err)
	}
	// macOS 上 /var → /private/var，工作目录自身也要取真实路径再比较
	realWorkingDir, err := filepath.EvalSymlinks(absWorkingDir)
	if err != nil {
		return "", fmt.Errorf("工作目录不可用: %w", err)
	}

	candidate := filepath.Clean(filepath.Join(absWorkingDir, relPath))
	if candidate != absWorkingDir && !isUnder(candidate, absWorkingDir) {
		return "", fmt.Errorf("路径越界: %s 不在工作目录 %s 内", relPath, absWorkingDir)
	}

	// 目标或其最近已存在祖先的真实路径
	target := candidate
	if allowMissing {
		if _, err := os.Lstat(candidate); err != nil {
			target = nearestExistingAncestor(candidate)
			if target == "" {
				return "", fmt.Errorf("路径越界: %s 的父目录不存在", relPath)
			}
		}
	}

	real, err := filepath.EvalSymlinks(target)
	if err != nil {
		if allowMissing && os.IsNotExist(err) {
			// 新建文件：祖先已校验过，目标本身不存在是允许的
			return candidate, nil
		}
		return "", fmt.Errorf("无法解析路径 %s: %w", relPath, err)
	}
	if real != realWorkingDir && !isUnder(real, realWorkingDir) {
		return "", fmt.Errorf("路径越界: %s 通过软链接指向了工作目录之外", relPath)
	}

	return candidate, nil
}

// isUnder 判断 path 是否位于 base 之内（不含 base 自身）。
func isUnder(path, base string) bool {
	return strings.HasPrefix(path, base+string(filepath.Separator))
}

// nearestExistingAncestor 返回 path 之上第一个真实存在的目录；找不到返回空串。
func nearestExistingAncestor(path string) string {
	dir := filepath.Dir(path)
	for {
		if _, err := os.Stat(dir); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
