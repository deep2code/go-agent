package tools

import (
	"os"
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
	mu   sync.Mutex
	seen map[string]int64 // 绝对路径 -> 读取时的文件大小
}

// NewFileState 创建文件状态跟踪器。
func NewFileState() *FileState {
	return &FileState{
		seen: make(map[string]int64),
	}
}

// Mark 记录一个已读取的文件。
func (f *FileState) Mark(absPath string) {
	if f == nil {
		return
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.seen[absPath] = info.Size()
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
