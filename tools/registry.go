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

// ParallelSafeTool 是可选接口：实现它且返回 true 的工具，允许和其他同类工具
// 在同一轮里并行执行。
//
// 缺省是串行——"没声明"不等于"并发安全"。判定标准只有两条，都要同时满足：
//  1. 不会改动文件系统（Mutating() 为 false），否则并行的两次改动会互相踩 diff 基线；
//  2. 自己不占终端、不派子循环、不产生额外花费。
//
// 典型反例是 task：它是只读的，但要跑完一整轮子 agent，并行等于同时开几个
// 无人看管的模型调用。
type ParallelSafeTool interface {
	Tool

	// ParallelSafe 报告该工具可以安全地并发执行。
	ParallelSafe() bool
}

// Parallelizable 判断一个工具能否进并发组。没实现 ParallelSafeTool 的一律串行。
func Parallelizable(t Tool) bool {
	ps, ok := t.(ParallelSafeTool)
	if !ok {
		return false
	}
	return !t.Mutating() && ps.ParallelSafe()
}

// BodyParamTool 是可选接口：实现它，声明"某个参数的值是一大块正文，
// 而这次调用成功之后，正文的每一个字节都已经在磁盘上了"。
//
// 缺省是"没有大正文参数"——没声明就没人动它，最坏结果只是历史里多留一份全文，
// 也就是慢和贵，不会变危险。
//
// 为什么只该写在成功落盘之后：写入类参数是模型自己生成的，换掉它等于把"这次改动写过什么"
// 的唯一记录从历史里拿走。只有在拿到成功结果之后，这份记录才真的有磁盘上的文件兜底。
// 还在等审批的调用不能换——diff 要从参数里生成。
//
// 只声明**能回读**的那一侧。edit_file 声明 new_string 而不声明 old_string：
// 成功后新内容在文件里，read_file 就能拿回来；old_string 已经被替换掉了，
// 历史是它唯一的去处，砍掉就再也找不回。
type BodyParamTool interface {
	Tool

	// BodyParams 返回参数名：其值在工具成功执行后可从磁盘回读，不必常驻历史。
	BodyParams() []string
}

// BodyParamNames 返回工具声明的大正文参数名；没声明就是空。
func BodyParamNames(t Tool) []string {
	bp, ok := t.(BodyParamTool)
	if !ok {
		return nil
	}
	return bp.BodyParams()
}

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

// Has 报告该名字是否已被占用。
//
// Register 重名会 panic（那是给本地代码的开发期门）；注册外部来源的工具前先用它查一遍，
// 外部工具名由别人决定，撞名是配置问题，不该把进程弄死。
func (r *Registry) Has(name string) bool {
	_, ok := r.tools[name]
	return ok
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
