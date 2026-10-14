package mcp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// DefaultConfigPath 是默认读取的配置文件位置（相对工作目录）。
const DefaultConfigPath = ".go-agent/mcp.json"

// DefaultTimeout 单次远端调用的默认超时。
const DefaultTimeout = 60 * time.Second

// ServerConfig 一个 MCP 服务器的启动配置。
type ServerConfig struct {
	Name     string
	Command  string
	Args     []string
	Env      map[string]string
	Timeout  time.Duration
	Disabled bool

	// MissingEnv 是配置里引用了但当前没有设置的环境变量名。
	// 空值静默展开会让远端服务器拿着空 key 跑下去，所以这里记一笔给启动时提示。
	MissingEnv []string
}

// fileServer 是 mcp.json 里的形状。
type fileServer struct {
	Command  string            `json:"command"`
	Args     []string          `json:"args"`
	Env      map[string]string `json:"env"`
	Timeout  string            `json:"timeout"`
	Disabled bool              `json:"disabled"`
}

type fileConfig struct {
	MCPServers map[string]fileServer `json:"mcpServers"`
}

// 服务器名要参与拼工具名，只允许字母数字和下划线/连字符。
var serverNameRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,32}$`)

// LoadConfig 读取配置文件里的服务器列表。
//
// 文件不存在不算错误——没有配置就是不开外部工具。
// env 和 args 里的 ${VAR} 从进程环境展开：真实凭据留在环境变量里，
// 不写进这份常常被顺手提交的文件。
func LoadConfig(path string) ([]ServerConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("读取 MCP 配置失败: %w", err)
	}

	var fc fileConfig
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&fc); err != nil {
		return nil, fmt.Errorf("解析 %s 失败: %w", path, err)
	}

	names := make([]string, 0, len(fc.MCPServers))
	for name := range fc.MCPServers {
		names = append(names, name)
	}
	// 稳定顺序：注册顺序决定日志与工具列表的可复现性
	for i := 1; i < len(names); i++ {
		for j := i; j > 0 && names[j-1] > names[j]; j-- {
			names[j-1], names[j] = names[j], names[j-1]
		}
	}

	out := make([]ServerConfig, 0, len(names))
	for _, name := range names {
		fs := fc.MCPServers[name]
		if !serverNameRe.MatchString(name) {
			return nil, fmt.Errorf("MCP 服务器名 %q 非法：只允许字母、数字、下划线和连字符，长度 1-32", name)
		}
		if strings.TrimSpace(fs.Command) == "" {
			return nil, fmt.Errorf("MCP 服务器 %s 缺少 command", name)
		}
		timeout := DefaultTimeout
		if strings.TrimSpace(fs.Timeout) != "" {
			d, err := time.ParseDuration(fs.Timeout)
			if err != nil || d <= 0 {
				return nil, fmt.Errorf("MCP 服务器 %s 的 timeout 无法解析: %q", name, fs.Timeout)
			}
			timeout = d
		}
		var missing []string
		env := make(map[string]string, len(fs.Env))
		for k, v := range fs.Env {
			env[k] = expand(v, &missing)
		}
		args := make([]string, len(fs.Args))
		for i, a := range fs.Args {
			args[i] = expand(a, &missing)
		}
		command := expand(fs.Command, &missing)
		// command 展开后为空意味着进程根本起不来，直接报错而不是留给启动
		if strings.TrimSpace(command) == "" {
			return nil, fmt.Errorf("MCP 服务器 %s 的 command 展开后为空（检查 ${VAR} 是否已设置）", name)
		}
		out = append(out, ServerConfig{
			Name:       name,
			Command:    command,
			Args:       args,
			Env:        env,
			Timeout:    timeout,
			Disabled:   fs.Disabled,
			MissingEnv: missing,
		})
	}
	return out, nil
}

var envRefRe = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// expand 把 ${VAR} 换成环境变量值；未设置的展开为空串并记进 missing。
func expand(s string, missing *[]string) string {
	return envRefRe.ReplaceAllStringFunc(s, func(m string) string {
		key := envRefRe.FindStringSubmatch(m)[1]
		v, ok := os.LookupEnv(key)
		if !ok {
			*missing = append(*missing, key)
			return ""
		}
		return v
	})
}

// ConfigPath 解析实际使用的配置路径：绝对路径原样返回，
// 相对路径按工作目录展开。
func ConfigPath(path, workingDir string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(workingDir, path)
}
