package tools

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ============================================================================
// RepoIndex — 会话内的文件元信息缓存
// ============================================================================

// RepoIndex 缓存"这个路径是文本还是二进制、多大、什么时候改的"。
//
// 只缓存元信息，不缓存内容：内容一旦留在缓存里，下一趟搜索就可能把过时代码交给模型。
// 存在的理由是搜索本身重复做苦活——大仓库每趟 search_code 都要打开每个候选文件读前 512 字节
// 判断是不是二进制。判定结论按 mtime+size 复核，文件一变自动作废。
//
// 搜索类工具可能并行执行，所以内部自带锁。
type RepoIndex struct {
	mu      sync.Mutex
	entries map[string]indexEntry
	probes  int // 真做过二进制判定的次数；测试用它证明"第二趟不再重做"
}

type indexEntry struct {
	size    int64
	modTime time.Time
	binary  bool
}

// NewRepoIndex 创建空索引。写类工具与只读工具共用同一个实例。
func NewRepoIndex() *RepoIndex {
	return &RepoIndex{entries: make(map[string]indexEntry)}
}

// IsTextFile 判断文件是否值得按文本读取，结论带缓存。
func (x *RepoIndex) IsTextFile(absPath string, info fs.FileInfo) bool {
	if x == nil {
		return !isBinary(absPath)
	}
	if e, ok := x.get(absPath); ok && e.size == info.Size() && e.modTime.Equal(info.ModTime()) {
		return !e.binary
	}
	// 判定要开文件读一段，不能持锁做：并发搜索时白堵住别的查询。
	binary := isBinary(absPath)
	x.put(absPath, indexEntry{size: info.Size(), modTime: info.ModTime(), binary: binary})
	return !binary
}

func (x *RepoIndex) get(absPath string) (indexEntry, bool) {
	x.mu.Lock()
	defer x.mu.Unlock()
	e, ok := x.entries[absPath]
	return e, ok
}

func (x *RepoIndex) put(absPath string, e indexEntry) {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.entries[absPath] = e
	x.probes++
}

// Probes 返回实际做过二进制判定的次数：每次数都意味着真的开了一次文件。
func (x *RepoIndex) Probes() int {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.probes
}

// Forget 丢掉一个路径的结论。写、移动、删除之后必须调用：
// 同名文件被换成二进制（或反过来）时，旧结论会替它说假话。
func (x *RepoIndex) Forget(absPath string) {
	if x == nil {
		return
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	delete(x.entries, absPath)
}

// isIgnoredDir 报告整棵子树是否不该扫。
//
// .go-agent 是智能体自己的目录（会话、备份、工具输出落在这里）：
// 搜它等于把回灌过的历史和别人的输出当成代码事实再读一遍。
func isIgnoredDir(name string) bool {
	switch name {
	case ".git", ".svn", ".hg", "vendor", "node_modules":
		return true
	}
	return name == ".go-agent" || strings.HasPrefix(name, ".go-agent")
}

// ============================================================================
// 仓库骨架：目录 → 文件数 / 体量 / 语言分布
// ============================================================================

// DirStat 一个目录直接持有的文件（不含子目录）。
type DirStat struct {
	Path    string // 相对工作目录，用斜杠分隔；"." 是根
	Depth   int
	Files   int
	Bytes   int64
	Exts    map[string]int
	Secrets int // 密钥类文件数，只计数不列名
}

// TopExts 返回该目录里出现最多的前 n 种扩展名，按数量降序、同数按名字升序。
func (d DirStat) TopExts(n int) []string {
	type kv struct {
		ext string
		c   int
	}
	all := make([]kv, 0, len(d.Exts))
	for e, c := range d.Exts {
		all = append(all, kv{e, c})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].c != all[j].c {
			return all[i].c > all[j].c
		}
		return all[i].ext < all[j].ext
	})
	if len(all) > n {
		all = all[:n]
	}
	out := make([]string, 0, len(all))
	for _, e := range all {
		out = append(out, fmt.Sprintf("%s:%d", e.ext, e.c))
	}
	return out
}

// Skeleton 一次遍历得到的仓库骨架。
type Skeleton struct {
	Dirs        []DirStat // 按路径字母序
	TotalFiles  int
	TotalBytes  int64
	IgnoredDirs int // 被跳过的目录数（vendor/node_modules/…）
}

// ScanSkeleton 走一遍目录树，只统计元信息，不读任何文件内容。
//
// maxDepth 限制展开层数：大仓库不封顶会刷出上千行，模型一样读不完。
func ScanSkeleton(absRoot, workingDir string, maxDepth int) Skeleton {
	skel := Skeleton{}
	byDir := make(map[string]*DirStat)

	// 层数按"距离本次起始目录几层"算，不是距离工作目录：从 agent/ 起步时 agent 自己就是第 0 层
	rootRel := slashRel(workingDir, absRoot)
	depthOf := func(rel string) int {
		d := pathDepth(rel) - pathDepth(rootRel)
		if d < 0 {
			return 0
		}
		return d
	}

	ensureDir := func(rel string) *DirStat {
		if d, ok := byDir[rel]; ok {
			return d
		}
		d := &DirStat{Path: rel, Depth: depthOf(rel), Exts: map[string]int{}}
		byDir[rel] = d
		return d
	}
	ensureDir(rootRel)

	_ = filepath.WalkDir(absRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil || path == absRoot {
			return nil // 读不动的路径跳过，不影响整体骨架
		}
		rel := slashRel(workingDir, path)

		if d.IsDir() {
			if isIgnoredDir(d.Name()) {
				skel.IgnoredDirs++
				return filepath.SkipDir
			}
			if depthOf(rel) > maxDepth {
				return filepath.SkipDir
			}
			ensureDir(rel)
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}
		st := ensureDir(filepath.ToSlash(filepath.Dir(rel)))
		skel.TotalFiles++
		skel.TotalBytes += info.Size()
		st.Files++
		st.Bytes += info.Size()

		if SensitivePath(rel) {
			// 只报数量：列名等于告诉模型去哪找凭据
			st.Secrets++
			return nil
		}
		ext := strings.ToLower(filepath.Ext(d.Name()))
		if ext == "" {
			ext = "(无扩展名)"
		}
		st.Exts[ext]++
		return nil
	})

	for _, st := range byDir {
		skel.Dirs = append(skel.Dirs, *st)
	}
	sort.Slice(skel.Dirs, func(i, j int) bool { return skel.Dirs[i].Path < skel.Dirs[j].Path })
	return skel
}

// slashRel 把工作目录内的路径换成相对工作目录、用斜杠分隔的展示路径。
func slashRel(workingDir, absPath string) string {
	rel, err := filepath.Rel(workingDir, absPath)
	if err != nil {
		return absPath
	}
	return filepath.ToSlash(rel)
}

// pathDepth 数路径有几段："." 是 0（根不占一段），"a/b" 是 2。
func pathDepth(rel string) int {
	if rel == "." || rel == "" {
		return 0
	}
	return strings.Count(rel, "/") + 1
}
