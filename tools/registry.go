// Package tools 提供代码智能体可用的工具集合。
//
// 每个工具都实现 Tool 接口，通过 Registry 注册后提供给 LLM 调用。
// 工具的设计遵循 Anthropic tool_use 规范。
package tools

import (
	"context"
	"fmt"
	"sort"

	"go-agent/llm"
)

// ============================================================================
// Tool 接口
// ============================================================================

// Tool 是代码智能体可调用的工具接口。
// 每个具体工具需实现 Definition（返回 JSON Schema 定义）和 Execute（执行业务逻辑）。
type Tool interface {
	// Name 返回工具名，须与 Definition() 中的 name 一致。
	Name() string

	// Definition 返回工具的 JSON Schema 定义，用于传给 LLM。
	Definition() llm.ToolDef

	// Execute 执行工具逻辑，接收 LLM 传入的参数，返回执行结果。
	// 如果返回 err，调用方应将错误信息作为 tool_result 的 is_error 返回给 LLM。
	Execute(ctx context.Context, input map[string]any) (string, error)

	// Mutating 报告该工具是否会改变文件系统或执行命令，供审批门判断。
	Mutating() bool
}

// ============================================================================
// Registry — 工具注册表
// ============================================================================

// Registry 管理所有可用工具的集合。
type Registry struct {
	tools map[string]Tool // name -> Tool
}

// NewRegistry 创建一个新的工具注册表。
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register 注册一个工具。如果工具名已存在则 panic（开发期尽早发现错误）。
func (r *Registry) Register(t Tool) {
	name := t.Name()
	if name == "" {
		panic("工具必须实现 Name()")
	}
	if def := t.Definition(); def.Name != name {
		panic(fmt.Sprintf("工具 %q 的 Definition().Name 与 Name() 不一致: %q", name, def.Name))
	}
	if _, ok := r.tools[name]; ok {
		panic(fmt.Sprintf("工具 %q 已注册", name))
	}
	r.tools[name] = t
}

// Get 按名称获取工具。
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Names 返回按字母序排列的工具名。
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Definitions 返回所有已注册工具的 JSON Schema 定义列表。
// 用于在每次 API 请求中携带给 LLM。
//
// 顺序固定为字母序：map 的遍历顺序是随机的，而工具定义处在请求前缀里，
// 顺序一变，服务端的 prompt cache 每次都命中不了。
func (r *Registry) Definitions() []llm.ToolDef {
	names := r.Names()
	defs := make([]llm.ToolDef, 0, len(names))
	for _, name := range names {
		defs = append(defs, r.tools[name].Definition())
	}
	return defs
}
