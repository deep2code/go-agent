package tools

import (
	"fmt"
	"io"
	"regexp"
	"sync"
)

// ============================================================================
// Gate — 有副作用操作的审批门
// ============================================================================

// GateMode 审批模式。
type GateMode string

const (
	// ModeAsk 每次改动作都问用户（交互模式默认）。
	ModeAsk GateMode = "ask"
	// ModeAuto 放行改动作，用于 --yes 和非交互模式下的显式授权。
	ModeAuto GateMode = "auto"
	// ModeReadOnly 拒绝一切改动作，用于 -p 未授权时的默认。
	ModeReadOnly GateMode = "readonly"
)

// Gate 决定写文件、删除、执行命令这类操作能否发生。
//
// 系统提示词里的"请勿执行 rm -rf"只是请求，不是边界；模型被诱导或判断失误时
// 一样会执行。真正的边界在这里。
type Gate struct {
	Mode GateMode

	// Ask 向用户确认，返回是否放行。为 nil 时按拒绝处理。
	Ask func(title, detail string) bool

	// Out 用于把拒绝原因回显给用户，可为 nil。
	Out io.Writer

	// askMu 把"提问—等回答"串起来。
	// 主 agent 和可写子 agent 共用一道门，两边都可能在问话；
	// 交错打印会让用户看到半句问题就输入了 y，那等于把确认变成抽签。
	askMu sync.Mutex
}

// NewGate 创建审批门。
func NewGate(mode GateMode) *Gate {
	return &Gate{Mode: mode}
}

// ErrDenied 表示操作被审批门拒绝。
var ErrDenied = fmt.Errorf("操作未获授权，已取消")

// Approve 请求执行一个有副作用的操作。
//
// 返回 nil 表示放行；返回 ErrDenied 表示被拒绝，调用方应把原因作为
// tool_result 的 is_error 交还给模型，让它换一条路子。
func (g *Gate) Approve(toolName, detail string) error {
	switch g.Mode {
	case ModeAuto:
		return nil

	case ModeReadOnly:
		g.note("🚫 %s 需要授权，但当前为只读模式（加 --yes 或改用交互模式）", toolName)
		return fmt.Errorf("%w: 只读模式禁止 %s", ErrDenied, toolName)

	case ModeAsk:
		if g.Ask == nil {
			g.note("🚫 %s 需要确认，但没有可用的确认渠道", toolName)
			return fmt.Errorf("%w: 无法向用户确认 %s", ErrDenied, toolName)
		}
		// 一次只问一个问题：子 agent 的写入和主 agent 的写入可能同时找上门，
		// 把整段"打印详情—读输入"锁住，用户看到的才是一道完整的问句。
		g.askMu.Lock()
		defer g.askMu.Unlock()
		if g.Ask(toolName, detail) {
			return nil
		}
		g.note("🚫 用户拒绝执行 %s", toolName)
		return fmt.Errorf("%w: 用户拒绝了 %s", ErrDenied, toolName)

	default:
		return fmt.Errorf("%w: 未知审批模式 %s", ErrDenied, g.Mode)
	}
}

// note 把审批门的判断回显到终端。
func (g *Gate) note(format string, args ...any) {
	if g.Out == nil {
		return
	}
	fmt.Fprintf(g.Out, "  "+format+"\n", args...)
}

// ============================================================================
// 命令风险分级
// ============================================================================

// forbiddenPatterns 无论何种模式都拒绝的命令特征。
// 这些操作对机器或仓库是不可逆的，必须由人在终端里手动执行。
var forbiddenPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\$HOME|\$\{HOME\})`),
	regexp.MustCompile(`\bmkfs(\.|\s|$)`),
	regexp.MustCompile(`\bdd\b[^|;]*\bof=/dev/`),
	regexp.MustCompile(`>\s*/dev/(sd|nvme|disk|rdisk)`),
	regexp.MustCompile(`:\(\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;`), // fork bomb
	regexp.MustCompile(`\b(chmod|chown)\s+(-[a-zA-Z]+\s+)*(777|000)\s+/(\s|$)`),
	regexp.MustCompile(`\b(shutdown|reboot|poweroff|halt)\b`),
	regexp.MustCompile(`\bgit\s+push\s+[^|;]*--force(-with-lease)?\s+.*(main|master)\b`),
}

// riskyPatterns 需要显式授权的命令特征（ModeAuto 下视为已授权）。
var riskyPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\bsudo\b`),
	regexp.MustCompile(`\bgit\s+push\b`),
	regexp.MustCompile(`\bgit\s+(reset\s+--hard|clean)\b`),
	regexp.MustCompile(`\b(curl|wget)\b[^|;]*\|\s*(bash|sh|zsh)\b`),
	regexp.MustCompile(`\brm\s+(-[a-zA-Z]+\s+)*[^\s]*\*`),
	regexp.MustCompile(`\bxargs\s+.*\brm\b`),
	regexp.MustCompile(`\bkill(all|proc)?\b`),
}

// CommandRisk 命令的风险等级。
type CommandRisk int

const (
	RiskSafe CommandRisk = iota
	RiskRisky
	RiskForbidden
)

// classifyCommand 给命令定级。
func classifyCommand(command string) (CommandRisk, string) {
	for _, re := range forbiddenPatterns {
		if re.MatchString(command) {
			return RiskForbidden, re.String()
		}
	}
	for _, re := range riskyPatterns {
		if re.MatchString(command) {
			return RiskRisky, re.String()
		}
	}
	return RiskSafe, ""
}

// CheckCommand 判断一条 shell 命令能否执行。
func (g *Gate) CheckCommand(command string) error {
	risk, pattern := classifyCommand(command)
	if risk == RiskForbidden {
		g.note("🚫 命令命中禁止名单（%s），需要人工在终端执行", pattern)
		return fmt.Errorf("%w: 该命令属于不可逆的破坏性操作，智能体无权执行", ErrDenied)
	}

	preview := command
	if len(preview) > 160 {
		preview = preview[:160] + "..."
	}
	if risk == RiskRisky && g.Mode == ModeAsk {
		g.note("⚠️  高风险命令（匹配 /%s/）", pattern)
	}
	return g.Approve("run_command", "$ "+preview)
}

// DescribeWrite 生成写入类操作的确认摘要。
func DescribeWrite(path string, bytes int, existed bool) string {
	verb := "创建"
	if existed {
		verb = "覆盖"
	}
	return fmt.Sprintf("%s文件 %s（%d 字节）", verb, path, bytes)
}
