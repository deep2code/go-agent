# AGENTS.md

面向在这个仓库上工作的编码智能体和贡献者。改动前先读这里。

## 这是什么

一个 Go 写的命令行代码智能体（ReAct 循环 + 工具调用 + 多厂商 LLM）。
它自己会读文件、改文件、跑命令，所以**安全边界是产品本身的核心，不是附属品**。

## 常用命令

```bash
/usr/local/go/bin/go build ./...     # 编译全部
/usr/local/go/bin/go vet ./...       # 静态检查
/usr/local/go/bin/go test ./...      # 全部测试（4 个包都覆盖）
/usr/local/go/bin/gofmt -l .         # 格式检查，提交前应无输出
```

这台机器上 `go` 不在 PATH，用绝对路径 `/usr/local/go/bin/go`。
8 核 16G，别在空闲时并发跑重编译。

## 架构与依赖方向

```
llm     ← 最底层，不依赖任何内部包。Provider 接口 + Anthropic / OpenAI 兼容(SSE) 实现 + 重试/错误分类
tools   ← 依赖 llm。9 个工具、审批门、路径安全、文件状态跟踪、命令风险分级
agent   ← 依赖 llm + tools。ReAct 主循环、上下文预算与折叠、历史修复、会话持久化、成本估算
main    ← 装配上面三者成 CLI：解析选项 → 选厂商 → 注册工具 → REPL / 非交互两条路径
log     ← 分级日志 + HTTP 请求/响应落盘
```

依赖只能向下。新增能力时先判断它属于哪一层，不要制造反向依赖。

## 不可破坏的约定

1. **提示词不是安全边界。** 真正的约束在 `tools/permissions.go`（审批门 + 命令分级）和
   `tools/safe_path.go`（`EvalSymlinks` 后再判目录包含）。任何"让模型别这么做"的措辞都不算防护；
   要挡住就得改代码。系统提示词只是引导。

2. **审批门有三态**：`ask`（交互逐个确认）、`auto`（`--yes` 预放行）、`readonly`（`-p` 不带 `--yes` 的默认，
   拒绝一切改动）。`--yes` 也**不能**绕过禁止名单（`RiskForbidden`）——见 `TestCheckCommandForbiddenIgnoresMode`。
   放宽任何一态都要有明确理由和对应测试。

3. **写操作先读后改**：没被 `read_file` 读过的文件不允许 `write_file`/`edit_file` 覆盖，靠 `tools/filestate.go` 跟踪。

4. **工具一致性**：`Registry.Register` 会在 `Definition().Name != Name()` 或重名时 panic——这是有意的，别改成 return error。
   工具定义按字母序下发（`Definitions()` 里排序），因为工具处在请求前缀，顺序一变服务端 prompt cache 就命中不了。
   新增工具必须同时出现在 `agent/prompt.go` 的系统提示词里，`TestEveryToolIsDocumented` 会挡住漏写。

5. **历史残缺要修好再发**：中断、截断（`finish_reason=length`）会留下无配对的 `tool_use`/`tool_result`。
   `agent/context.go` 的 `RepairHistory` 负责补占位、删孤儿、合并同角色；`CompactHistory` 只在**超出预算时**才折叠，
   没超就原样返回（别让它无谓破坏历史）。

6. **子进程隔离**：`run_command` 用独立进程组，超时整组 `SIGKILL`（`run_unix.go`），`cmd.Stdin = nil` 不继承终端。

## 测试风格

- 每个包都有独立测试。网络用 `httptest` 起本地 SSE 服务，不碰真实 API。
- 文件操作用 `t.TempDir()`，命令测试只写 `/tmp`，绝不碰真实仓库或用户目录。
- 测试是**行为门**，不是摆设。历史上多个真实缺陷（`rm -rf /etc` 漏判、`git clean` 分级错、
  会话 ID 同秒撞车、被截断的 tool_use 复活）都是测试先红、再修产品代码修好的。
- 测试红的时候先查根因改产品代码，**不要**为了变绿去弱化断言或跳过。
- `main_test.go` 用 `isolateEnv` 清空所有厂商密钥与日志目录环境变量，别让它泄漏到真实 shell 或写出 `./logs`。

## 文档同步

改完代码，README.md 和本文件若与事实不符就当场更新（选项、命令、安全行为、依赖方向）。
过时的文档比没文档更糟。
