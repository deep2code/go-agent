# go-agent

一个用 Go 写的命令行代码智能体。接入 LLM 后，它能在你的项目目录里读代码、改文件、跑命令，
像一个会自己动手的编程助手。支持交互模式（一条命令接一条地聊）和非交互模式（脚本、CI 里一次跑完）。

## 能力概览

- **ReAct 循环**：模型决定调用哪个工具，工具结果再喂回模型，循环直到任务完成。
- **9 个工具**：读文件、搜索、列目录、写文件、精确编辑、重命名/移动、删除、执行命令、任务清单。
- **多厂商**：DeepSeek（默认）、Anthropic Claude，以及任意 OpenAI 兼容端点。
- **上下文治理**：估算 token 占用，接近上限时折叠旧的工具输出、丢弃最早的往返，修好被中断的历史。
- **写操作确认**：改文件、删文件、执行命令都要过人；破坏性命令直接拦下。
- **会话持久化**：对话落盘到 `.go-agent/sessions`，可随时恢复。
- **流式输出**：逐字打印回复，中断（Ctrl+C）保留现场。

## 安装

需要 Go 1.26 或更高版本。

```bash
git clone <本仓库>
cd go-agent
./build.sh          # 编译并安装到 $GOPATH/bin，之后可全局用 go-agent
./build.sh -l       # 只编译到 ./build/，不安装
./build.sh -c       # 清理编译产物
```

不想用脚本也可以直接 `go build -o go-agent .`。

## 配置密钥

至少设置一个厂商的密钥。默认为 DeepSeek：

```bash
export DEEPSEEK_API_KEY=sk-...      # DeepSeek（默认）
export ANTHROPIC_API_KEY=sk-ant-...  # Anthropic；没配 DeepSeek 时自动切到它
export OPENAI_API_KEY=sk-...         # 配合 -provider openai 使用
```

## 使用

### 交互模式

直接运行，进入 REPL。写操作逐个确认：

```bash
cd 你的项目
go-agent
```

```
> 这个项目的入口在哪？帮我梳理一下调用链
> 把 utils.go 里的 ParseConfig 重构成返回 error 而不是 panic
```

斜杠命令：

| 命令 | 作用 |
|------|------|
| `/help` | 查看帮助 |
| `/clear` | 清空对话，开新会话 |
| `/sessions` · `/resume <ID\|last>` | 列出并恢复历史会话 |
| `/model [名]` | 查看或切换模型 |
| `/cost` | 累计 token 与估算花费 |
| `/tools` | 列出可用工具 |
| `/plan` | 查看当前任务清单 |
| `/log [级别]` | 查看或切换日志级别 |
| `/reset-approval` | 恢复逐个确认 |
| `/exit` | 退出 |

执行中按 `Ctrl+C` 取消当前任务，对话保留，接着说"继续"即可。

### 非交互模式

跑一条任务就退出，适合脚本和 CI：

```bash
go-agent -p "统计这个目录里所有 TODO 注释"          # 默认只读，写操作会被拒
go-agent -p "给 retry.go 补上单元测试" --yes         # 授权它写文件、跑命令
go-agent -p "..." --json                             # 只输出一段 JSON 结果
```

`-p` 未加 `--yes` 时是只读的：任何写文件、删除、执行命令的请求都会被拒绝。
加上 `--yes` 才预先放行——但破坏性命令清单仍然生效。

`--json` 输出结构：

```json
{
  "reply": "……最终回复……",
  "stop_reason": "completed",
  "iterations": 4,
  "usage": { "input_tokens": 1200, "output_tokens": 300 },
  "cost_usd": 0.0012,
  "error": ""
}
```

## 命令行选项

```
-p <任务>            非交互执行一条任务后退出
-provider <名>       deepseek（默认）/ anthropic / openai
-model <ID>          覆盖默认模型
-yes                 预先放行所有写操作与命令（禁止清单仍然生效）
-json                只输出 JSON 结果（隐含 -quiet）
-quiet               不打印执行过程
-max-turns <n>       单次任务最大轮数（默认 30）
-cmd-timeout <dur>   单条命令默认超时（默认 2m）
-resume <ID|last>    启动即恢复某次会话
-no-session          不把对话写入 .go-agent/sessions
-log-level <级>      debug / info / warn / error
-version             打印版本后退出
```

环境变量：`LLM_PROVIDER`、`LLM_MODEL`、`LLM_BASE_URL`、`LOG_LEVEL`、`LLM_LOG_DIR`（HTTP 日志目录，默认 `./logs`）。

## 安全模型

提示词只是引导，不是边界。真正的约束在代码里：

- **路径限制**：所有文件操作都限定在工作目录内，用 `filepath.EvalSymlinks` 解析软链接后再判断，
  绝对路径和 `..` 逃逸会被拒绝。
- **先读后改**：没读过的文件不允许直接覆盖写入。
- **写操作确认**：写文件、移动、删除、执行命令都经审批门；三种模式——`ask`（交互逐个确认）、
  `auto`（`--yes` 预放行）、`readonly`（`-p` 默认，拒绝一切改动）。
- **破坏性命令拦截**：`rm -rf /`、`mkfs`、写裸盘、fork 炸弹、`git push --force` 到主干等
  在禁止清单里，任何模式下都会被拦下；高风险命令（如 `git reset --hard`）在确认时额外告警。
- **命令执行**：子进程放进独立进程组，超时后整组 kill，`stdin` 关闭不继承终端。

## 架构

```
main.go      CLI 装配：解析选项、选择厂商、注册工具、REPL 与非交互两条执行路径
agent/       ReAct 主循环、上下文预算与折叠、历史修复、会话持久化、成本估算
llm/         Provider 接口 + Anthropic / OpenAI 兼容(SSE)实现、重试与错误分类
tools/       9 个工具、审批门、路径安全、文件状态跟踪、命令风险分级
log/         分级日志 + HTTP 请求/响应落盘
```

- `llm` 不依赖其它内部包，是最底层的 API 适配。
- `tools` 依赖 `llm`（工具定义用同一套 schema 类型）。
- `agent` 依赖 `llm` 和 `tools`，负责把两者编排成循环。
- `main` 把三者装配成可运行的 CLI。

## 开发

```bash
go build ./...        # 编译
go vet ./...          # 静态检查
gofmt -l .            # 格式检查（提交前应无输出）
go test ./...         # 运行全部测试
```

`tools/` 覆盖路径穿越、审批门分级、命令超时与进程组清理；`llm/` 覆盖 SSE 解析、
重试分类与退避；`agent/` 覆盖 ReAct 分支（长度截断、残缺历史、中断、超上下文折叠、
最大轮数）与会话存取；`main` 覆盖选项到审批模式/模型/输出的装配。CI 在
`.github/workflows/ci.yml` 里跑 gofmt、vet、build、test。
