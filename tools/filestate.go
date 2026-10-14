package tools

import (
	"os"
	"path/filepath"
	"sync"
)

// ============================================================================
// FileState — 记录本次会话里已经读过的文件
// ============================================================================

// FileState 记录工具读过的文件及其大小。
//
// 存在的理由：模型可以凭猜测写一个 file_path 直接覆盖掉它从没读过的文件。
// 写入前要求"这个文件我读过"，能把这类误伤挡在一次工具调用之前。
type FileState struct {
	mu         sync.Mutex
	seen       map[string]int64  // 绝对路径 -> 读取时的文件大小
	rels       map[string]string // 绝对路径 -> 读取时用的相对路径（展示依据来源）
	order      []string          // 读取过的绝对路径，最早在前
	suspicious map[string]bool   // 相对路径 -> 输出里出现过伪装成指令的文字
}

// NewFileState 创建文件状态跟踪器。
func NewFileState() *FileState {
	return &FileState{
		seen:       make(map[string]int64),
		rels:       make(map[string]string),
		suspicious: make(map[string]bool),
	}
}

// Mark 记录一个已读取的文件。
func (f *FileState) Mark(absPath string) {
	f.mark(absPath, "")
}

// MarkRead 记录一次读取，同时保存用户看到的相对路径，供依据来源展示。
func (f *FileState) MarkRead(relPath, absPath string) {
	f.mark(absPath, relPath)
}

func (f *FileState) mark(absPath, relPath string) {
	if f == nil {
		return
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.seen[absPath]; !ok {
		f.order = append(f.order, absPath)
	} else {
		// 重读要把顺序挪到最新：依据看的是"最近看过什么"。
		for i, p := range f.order {
			if p == absPath {
				f.order = append(f.order[:i], f.order[i+1:]...)
				break
			}
		}
		f.order = append(f.order, absPath)
	}
	f.seen[absPath] = info.Size()
	if relPath != "" {
		f.rels[absPath] = relPath
	}
}

// MarkSuspicious 记下某个文件的输出里出现过伪装成指令的文字。
// relPath 为空（命令输出这类）时只影响日志，不落到某个文件上。
func (f *FileState) MarkSuspicious(relPath string) {
	if f == nil || relPath == "" {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.suspicious[relPath] = true
}

// ReadNote 一个读过的文件及其可疑标记。
type ReadNote struct {
	Path       string
	Suspicious bool
}

// RecentReads 返回最近读过的至多 n 个文件，最新的在后。
func (f *FileState) RecentReads(n int) []ReadNote {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	notes := make([]ReadNote, 0, n)
	for i := len(f.order) - 1; i >= 0 && len(notes) < n; i-- {
		abs := f.order[i]
		name := f.rels[abs]
		if name == "" {
			name = filepath.Base(abs)
		}
		notes = append(notes, ReadNote{Path: name, Suspicious: f.suspicious[name]})
	}
	// 倒着收集，正着读。
	for l, r := 0, len(notes)-1; l < r; l, r = l+1, r-1 {
		notes[l], notes[r] = notes[r], notes[l]
	}
	return notes
}

// Unchanged 判断文件自上次读取后是否仍是同一个大小。
func (f *FileState) Unchanged(absPath string) bool {
	if f == nil {
		return true
	}
	f.mu.Lock()
	size, ok := f.seen[absPath]
	f.mu.Unlock()
	if !ok {
		return false
	}
	info, err := os.Stat(absPath)
	return err == nil && info.Size() == size
}

// HasSeen 报告该路径是否在本会话中被读过。
func (f *FileState) HasSeen(absPath string) bool {
	if f == nil {
		return true
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	_, ok := f.seen[absPath]
	return ok
}

// CheckWritable 校验目标文件能否被安全地创建或覆盖。
//
// 返回 existed 表示文件已存在。已存在但从未读过、或读过后大小又变了，都会拒绝，
// 让模型先调用 read_file。
func (f *FileState) CheckWritable(relPath, absPath string) (existed bool, err error) {
	if _, statErr := os.Stat(absPath); statErr != nil {
		if os.IsNotExist(statErr) {
			return false, nil
		}
		return false, statErr
	}
	if f == nil || f.HasSeen(absPath) {
		return true, nil
	}
	return true, &StaleFileError{Path: relPath}
}

// StaleFileError 表示要修改一个本会话还没读过的文件。
type StaleFileError struct {
	Path string
}

func (e *StaleFileError) Error() string {
	return "文件 " + e.Path + " 尚未读取过，请先用 read_file 查看内容再修改（避免整文件覆盖丢内容）"
}
