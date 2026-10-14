package tools

import (
	"encoding/json"
	"io"
	"reflect"
	"testing"
	"time"
)

// 大正文参数的声明对着真实工具表校验，理由和并发白名单一样：
// 最容易出错的是"新加了写入工具却忘了声明"，而那只会让历史白白变大，不会报错。
func TestBodyParamsOnRealTools(t *testing.T) {
	dir := t.TempDir()
	gate := NewGate(ModeAuto)
	state := NewFileState()

	if got := BodyParamNames(NewWriteFileTool(dir, gate, state)); !reflect.DeepEqual(got, []string{"content"}) {
		t.Errorf("write_file 的大正文参数 = %v，期望只有 content", got)
	}
	// old_string 不在名单里是刻意的：替换成功后原文已经不在磁盘上了，
	// 历史是它唯一的去处，砍掉等于让模型再也看不到自己改掉了什么。
	if got := BodyParamNames(NewEditFileTool(dir, gate, state)); !reflect.DeepEqual(got, []string{"new_string"}) {
		t.Errorf("edit_file 的大正文参数 = %v，期望只有 new_string", got)
	}

	// 其余工具一律没有大正文参数：命令、读取结果、子 agent 报告都不是"落盘后可回读"的东西。
	for _, tool := range []Tool{
		NewReadFileTool(dir, state),
		NewSearchCodeTool(dir),
		NewListDirectoryTool(dir),
		NewRepoOverviewTool(dir),
		NewRunCommandTool(dir, gate, time.Second, io.Discard),
		NewPlanTool(io.Discard),
		NewTaskTool(nil, NewGate(ModeReadOnly), io.Discard),
	} {
		if names := BodyParamNames(tool); len(names) != 0 {
			t.Errorf("%s 不该声明大正文参数，实际声明了 %v", tool.Name(), names)
		}
	}
}

// 声明写错名字不会报错，只会静默失效——参数名必须真的在该工具的 schema 里。
func TestBodyParamsNameRealSchemaKeys(t *testing.T) {
	dir := t.TempDir()
	gate := NewGate(ModeAuto)
	state := NewFileState()

	for _, tool := range []Tool{
		NewWriteFileTool(dir, gate, state),
		NewEditFileTool(dir, gate, state),
	} {
		props := tool.Definition().InputSchema.Properties
		for _, name := range BodyParamNames(tool) {
			if _, ok := props[name]; !ok {
				t.Errorf("%s 声明了 schema 里不存在的参数 %q，瘦身会永远轮不到它", tool.Name(), name)
			}
		}
	}
}

// 没实现接口的工具就是没有大正文参数——缺省从严，不是缺省可省。
func TestBodyParamsDefaultsToNone(t *testing.T) {
	if names := BodyParamNames(noDeclare{}); len(names) != 0 {
		t.Errorf("未声明的工具返回了 %v", names)
	}
	if names := BodyParamNames(declared{}); len(names) != 0 {
		t.Errorf("只声明并行的工具不该带出大正文参数，实际 %v", names)
	}
}

// 远端工具永远不该被瘦身：正文没落盘，砍掉就是真丢数据。
func TestBodyParamsForRemoteTools(t *testing.T) {
	srv := &fakeMCPServer{name: "srv"}
	remote := NewMCPTool(srv, NewGate(ModeAuto), time.Second,
		"mcp_srv_write", "write", false, "远端写入工具", json.RawMessage(`{"type":"object"}`))
	if names := BodyParamNames(remote); len(names) != 0 {
		t.Errorf("远端工具声明了大正文参数 %v", names)
	}
}
