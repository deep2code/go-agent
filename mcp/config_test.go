package mcp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mcp.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfigMissingFileIsNotAnError(t *testing.T) {
	servers, err := LoadConfig(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("没有配置不该报错: %v", err)
	}
	if servers != nil {
		t.Errorf("期望空列表，得到 %+v", servers)
	}
}

func TestLoadConfigBasics(t *testing.T) {
	t.Setenv("CTX7_KEY", "secret-from-env")
	path := writeConfig(t, `{
	  "mcpServers": {
	    "context7": {
	      "command": "npx",
	      "args": ["-y", "@upstash/context7-mcp", "--key", "${CTX7_KEY}"],
	      "env": {"API_KEY": "${CTX7_KEY}"},
	      "timeout": "90s"
	    },
	    "zzz-local": {"command": "uv", "args": ["run", "srv"]},
	    "off-one": {"command": "echo", "disabled": true}
	  }
	}`)

	servers, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(servers) != 3 {
		t.Fatalf("服务器数量 = %d", len(servers))
	}
	// 名字排序保证同一份配置每次启动的顺序一致
	if servers[0].Name != "context7" || servers[1].Name != "off-one" || servers[2].Name != "zzz-local" {
		t.Errorf("顺序不稳定: %+v", servers)
	}

	c7 := servers[0]
	if c7.Timeout != 90*time.Second {
		t.Errorf("timeout = %v", c7.Timeout)
	}
	if c7.Args[3] != "secret-from-env" {
		t.Errorf("args 里的 ${VAR} 没展开: %v", c7.Args)
	}
	if c7.Env["API_KEY"] != "secret-from-env" {
		t.Errorf("env 里的 ${VAR} 没展开: %+v", c7.Env)
	}
	if servers[1].Disabled != true {
		t.Error("disabled 标记丢了")
	}
	if servers[2].Timeout != DefaultTimeout {
		t.Errorf("没写 timeout 时应用默认值，得到 %v", servers[2].Timeout)
	}
}

func TestLoadConfigRejectsBadShapes(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"服务器名非法", `{"mcpServers":{"a b":{"command":"x"}}}`, "非法"},
		{"服务器名带点", `{"mcpServers":{"a.b":{"command":"x"}}}`, "非法"},
		{"缺少 command", `{"mcpServers":{"ok":{}}}`, "缺少 command"},
		{"command 为空", `{"mcpServers":{"ok":{"command":"  "}}}`, "缺少 command"},
		{"timeout 无法解析", `{"mcpServers":{"ok":{"command":"x","timeout":"很快"}}}`, "timeout"},
		{"timeout 为负", `{"mcpServers":{"ok":{"command":"x","timeout":"-5s"}}}`, "timeout"},
		{"拼错的字段", `{"mcpServers":{"ok":{"command":"x","commnad":"y"}}}`, "unknown"},
		{"顶层不是对象", `[]`, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := LoadConfig(writeConfig(t, c.body))
			if err == nil {
				t.Fatalf("%s 应当报错", c.body)
			}
			if c.want != "" && !strings.Contains(err.Error(), c.want) {
				t.Errorf("错误信息应包含 %q，得到 %v", c.want, err)
			}
		})
	}
}

func TestLoadConfigRecordsMissingEnv(t *testing.T) {
	t.Setenv("SET_ONE", "on")
	path := writeConfig(t, `{"mcpServers":{"ok":{"command":"bin","env":{"A":"${SET_ONE}","B":"${NOT_SET_KEY}"}}}}`)

	servers, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(servers) != 1 {
		t.Fatalf("数量 %d", len(servers))
	}
	if servers[0].Env["A"] != "on" {
		t.Errorf("已设置的变量没展开: %+v", servers[0].Env)
	}
	// 未设置的展开为空串，但必须记下来，否则远端服务器拿着空 key 静默跑
	if servers[0].Env["B"] != "" {
		t.Errorf("未设置变量应展开为空: %q", servers[0].Env["B"])
	}
	if strings.Join(servers[0].MissingEnv, ",") != "NOT_SET_KEY" {
		t.Errorf("MissingEnv = %v", servers[0].MissingEnv)
	}
}

func TestLoadConfigCommandExpandsToEmpty(t *testing.T) {
	path := writeConfig(t, `{"mcpServers":{"ok":{"command":"${NO_SUCH_BIN_PATH}"}}}`)
	if _, err := LoadConfig(path); err == nil ||
		!strings.Contains(err.Error(), "展开后为空") {
		t.Fatalf("command 展开成空串时必须报错: %v", err)
	}
}

func TestConfigPath(t *testing.T) {
	if got := ConfigPath("/etc/mcp.json", "/work"); got != "/etc/mcp.json" {
		t.Errorf("绝对路径不该被改写: %s", got)
	}
	if got := ConfigPath(DefaultConfigPath, "/work"); got != "/work/.go-agent/mcp.json" {
		t.Errorf("相对路径应按工作目录展开: %s", got)
	}
}
