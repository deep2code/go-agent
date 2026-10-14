package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// 外部来源（MCP）的工具定义带的是原始 JSON Schema。
// 这里钉住两件事：enum / items / required 这些结构必须原样到达服务端，
// 以及非法 schema 要在发出请求前就失败，而不是把一堆垃圾发给模型厂商。

const remoteSchema = `{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"],` +
	`"description":"取值"},"tags":{"type":"array","items":{"type":"string"}}},` +
	`"required":["mode"],"additionalProperties":false}`

func remoteTool() ToolDef {
	return ToolDef{
		Name:        "mcp_fs_export",
		Description: "导出一份文件",
		RawSchema:   json.RawMessage(remoteSchema),
	}
}

func TestToolDefMarshalKeepsRawSchema(t *testing.T) {
	data, err := json.Marshal(remoteTool())
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		InputSchema json.RawMessage `json:"input_schema"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Name != "mcp_fs_export" {
		t.Errorf("name = %q", got.Name)
	}
	if string(got.InputSchema) != remoteSchema {
		t.Errorf("input_schema 应逐字节等于远端给的 schema：\n得到 %s\n期望 %s", got.InputSchema, remoteSchema)
	}

	// 常见结构必须在字段名和层级上都保持原样，模型才可能按远端要求填参数
	for _, want := range []string{`"enum":["fast","safe"]`, `"items":{"type":"string"}`,
		`"required":["mode"]`, `"additionalProperties":false`} {
		if !strings.Contains(string(data), want) {
			t.Errorf("序列化后丢了 %s，实际请求体：%s", want, data)
		}
	}
}

func TestToolDefMarshalWithoutRawSchema(t *testing.T) {
	local := ToolDef{
		Name:        "read_file",
		Description: "读取文件",
		InputSchema: InputSchema{
			Type:       "object",
			Properties: map[string]Property{"file_path": {Type: "string", Description: "相对路径"}},
			Required:   []string{"file_path"},
		},
	}
	data, err := json.Marshal(local)
	if err != nil {
		t.Fatal(err)
	}
	// 结构体序列化不能因为自定义 MarshalJSON 而把 input_schema 变成字符串或丢掉
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	schema, ok := got["input_schema"].(map[string]any)
	if !ok {
		t.Fatalf("input_schema 形状不对: %s", data)
	}
	if schema["type"] != "object" {
		t.Errorf("input_schema.type = %v", schema["type"])
	}
	if _, ok := schema["properties"].(map[string]any)["file_path"]; !ok {
		t.Errorf("properties.file_path 没了: %s", data)
	}
	if _, has := got["cache_control"]; has {
		t.Error("未设断点时不该带 cache_control")
	}
}

func TestToolDefMarshalKeepsCacheBreakpoint(t *testing.T) {
	tool := remoteTool()
	tool.CacheControl = ephemeralCache()
	data, err := json.Marshal(tool)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"cache_control":{"type":"ephemeral"}`) {
		t.Errorf("透传 raw schema 时断点丢了: %s", data)
	}
}

func TestToolDefMarshalRejectsBrokenRawSchema(t *testing.T) {
	for _, bad := range []string{`{"type":"object",}`, `not json`, `{"type":"object"`} {
		tool := remoteTool()
		tool.RawSchema = json.RawMessage(bad)
		if _, err := json.Marshal(tool); err == nil {
			t.Errorf("非法 schema %q 竟然序列化成功了，垃圾会被直接发给厂商", bad)
		}
	}
}

// TestAnthropicWireKeepsRemoteSchema：加缓存断点时 Tools 会被复制一份，
// 复制必须把 raw schema 一起带过去，否则第二起轮次的请求就没有参数说明了。
func TestAnthropicWireKeepsRemoteSchema(t *testing.T) {
	long := strings.Repeat("说明", 4000) // 撑过 minCachePrefixChars，走复制那条分支
	wire := buildAnthropicRequest(ChatRequest{
		Model:  "claude-x",
		System: long,
		Tools:  []ToolDef{localTool(), remoteTool()},
	})
	data, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"enum":["fast","safe"]`) {
		t.Errorf("远端 schema 在带断点的请求里丢了: %s", data)
	}
	if !strings.Contains(string(data), `"cache_control":{"type":"ephemeral"}`) {
		t.Error("断点应打在最后一个工具上")
	}
	// 调用方传进来的那份不该被改写
	if has := wire.Tools[len(wire.Tools)-1].CacheControl; has == nil {
		t.Error("断点没打在最后一个工具上")
	}
}

func localTool() ToolDef {
	return ToolDef{
		Name:        "read_file",
		Description: "读取文件",
		InputSchema: InputSchema{Type: "object", Properties: map[string]Property{"file_path": {Type: "string"}}},
	}
}

// TestOpenAIWireKeepsRemoteSchema：OpenAI 兼容端点用的是 parameters 字段，
// 那边没有 input_schema 这个名字，映射写错就是静默丢参数说明。
func TestOpenAIWireKeepsRemoteSchema(t *testing.T) {
	p := &DeepSeekProvider{}
	oa := p.toOpenAIRequest(ChatRequest{Model: "deepseek-chat", Tools: []ToolDef{localTool(), remoteTool()}})
	data, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}

	var got struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name       string         `json:"name"`
				Parameters map[string]any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Tools) != 2 {
		t.Fatalf("工具数量 = %d", len(got.Tools))
	}
	local, remote := got.Tools[0], got.Tools[1]
	if local.Function.Parameters["type"] != "object" {
		t.Errorf("本地工具的 parameters 不对: %v", local.Function.Parameters)
	}
	props, ok := remote.Function.Parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("远端工具的 parameters 没有 properties，实际请求体: %s", data)
	}
	mode, ok := props["mode"].(map[string]any)
	if !ok {
		t.Fatalf("properties.mode 丢了: %v", props)
	}
	enum, ok := mode["enum"].([]any)
	if !ok || len(enum) != 2 || enum[1] != "safe" {
		t.Errorf("enum 没原样传到: %v", mode["enum"])
	}
	if remote.Function.Parameters["additionalProperties"] != false {
		t.Errorf("additionalProperties 没传到: %v", remote.Function.Parameters)
	}
}
