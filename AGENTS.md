# AGENTS.md

面向在这个仓库上工作的编码智能体和贡献者。改动前先读这里。

## 这是什么

一个 Go 写的命令行代码智能体（ReAct 循环 + 工具调用 + 多厂商 LLM）。
它自己会读文件、改文件、跑命令，所以**安全边界是产品本身的核心，不是附属品**。

## 常用命令

```bash
/usr/local/go/bin/go build ./...     # 编译全部
/usr/local/go/bin/go vet ./...       # 静态检查
/usr/local/go/bin/go test ./...      # 全部测试（5 个包都覆盖）
/usr/local/go/bin/gofmt -l .         # 格式检查，提交前应无输出
```

这台机器上 `go` 不在 PATH，用绝对路径 `/usr/local/go/bin/go`。
8 核 16G，别在空闲时并发跑重编译。

## 架构与依赖方向

```
log     ← 最底层，不依赖任何内部包。分级日志 + HTTP 请求/响应落盘
mcp     ← 同样不依赖内部包。MCP 客户端：JSON-RPC over stdio、工具发现/调用、服务器配置、子进程生命周期
llm     ← 依赖 log。Provider 接口 + Anthropic / OpenAI 兼容(SSE) 实现 + 重试/错误分类 + prompt caching
tools   ← 依赖 llm + mcp。11 个本地工具、外部工具适配、审批门（含 diff 展示）、路径安全、文件状态跟踪、命令风险分级、外部内容守卫、会话级文件元信息缓存、并发安全声明、大正文参数声明
agent   ← 依赖 llm + tools + log。ReAct 主循环、大输出两级外置治理（L1 入口 / L2 构建时指针）、写入参数瘦身与重复读取省略、上下文预算与折叠（含语义摘要、服务端用量校准）、历史修复、花费上限、会话持久化、成本估算
main    ← 装配上面几层成 CLI：解析选项 → 选厂商 → 注册工具 → 拉起外部服务器 → 装配子 agent（只读/受限可写两副面孔）→ REPL / 非交互两条路径
```

依赖只能向下（README「架构」有同视角的 mermaid 图，改依赖时两边一起改）。
新增能力时先判断它属于哪一层，不要制造反向依赖。

## 不可破坏的约定

1. **提示词不是安全边界。** 真正的约束在 `tools/permissions.go`（审批门 + 命令分级）、
   `tools/safe_path.go`（`EvalSymlinks` 后再判目录包含）和 `tools/content_guard.go`
   （见第 10 条）。任何"让模型别这么做"的措辞都不算防护；
   要挡住就得改代码。系统提示词只是引导。

2. **审批门有三态**：`ask`（交互逐个确认）、`auto`（`--yes` 预放行）、`readonly`（`-p` 不带 `--yes` 的默认，
   拒绝一切改动）。`--yes` 也**不能**绕过禁止名单（`RiskForbidden`）——见 `TestCheckCommandForbiddenIgnoresMode`。
   放宽任何一态都要有明确理由和对应测试。
   `ask` 模式下的写入类工具必须把**真实 diff** 放进确认详情（`tools/diff.go`，由 `UnifiedDiff` 生成）：
   只报"某文件将变成 N 字节"等于让人盲签。新增写入类工具时照此办理，且只在 `ModeAsk` 时生成——
   auto/readonly 没人会看到，不值得为它读一遍文件。diff 有条目数与输出行数上限，超限退化成统计信息。

3. **写操作先读后改**：没被 `read_file` 读过的文件不允许 `write_file`/`edit_file` 覆盖，靠 `tools/filestate.go` 跟踪。

4. **工具一致性**：`Registry.Register` 会在 `Definition().Name != Name()` 或重名时 panic——这是有意的，别改成 return error。
   工具定义按字母序下发（`Definitions()` 里排序），因为工具处在请求前缀，顺序一变服务端 prompt cache 就命中不了。
   新增工具必须同时出现在 `agent/prompt.go` 的系统提示词里，`TestEveryToolIsDocumented` 会挡住漏写。
   给**子 agent** 加工具则要同时改它那一副面孔的提示词（`agent.SubAgentPrompt` 只读面 /
   `agent.SubAgentWritePrompt` 可写面，公共部分在 `subAgentBody`）——两边的一致性由
   `TestSubAgentIsReadOnly` 对两副面孔分别双向校验：工具表里有的必须写进提示词，提示词里提到的必须真有其工具。
   另外，新增**只读**工具时要想清楚是否同时声明 `ParallelSafe()`（见第 13 条）；不声明只是变慢，不会变危险。
同理，新增**写入类**工具要想清楚是否声明 `BodyParams()`（见第 15 条）——不声明只是历史里多留一份全文，不会丢数据。

5. **历史残缺要修好再发**：中断、截断（`finish_reason=length`）会留下无配对的 `tool_use`/`tool_result`。
   `agent/context.go` 的 `RepairHistory` 负责补占位、删孤儿、合并同角色；`CompactHistory` 只在**超出预算时**才折叠，
   没超就原样返回（别让它无谓破坏历史）。折叠会丢弃整轮、往首条消息里插摘要，输出可能不再满足角色交替，
   所以 `Agent.compact` 折完必须再过一次 `RepairHistory` 才能发出去。
   第二级丢弃会调 `Summarizer` 把丢掉的往返压成摘要（`agent/summarize.go`）；
   摘要调用失败就退化成纯丢弃，不能因为摘要报错而让整个任务失败。
   `prepareForSend` 的顺序是**修复 → L2 指针替代（见第 14 条）+ 写入参数瘦身（见第 15 条）→
   估算 → 比较预算 → 折叠**：
   这两级排在估算之后就等于永远轮不到发言，因为"超没超"已经按瘦身之前的体积判完了。

6. **子进程隔离**：`run_command` 用独立进程组，超时整组 `SIGKILL`（`run_unix.go`），`cmd.Stdin = nil` 不继承终端。

7. **子 agent 有两副面孔，都封不了顶**：`tools.NewTaskTool` 依赖 `SubAgentRunner` 接口，实现在 `main.go`
   （`subAgentRunner` / `newSubAgent` / `subRegistry`），这样依赖方向不反转。
   缺省是**只读面**，注册表**只能**有 `read_file`/`search_code`/`list_directory`/`repo_overview`；
   模型传 `writable=true` 才切到**可写面**，可写面也**只**能多 `write_file` 与 `edit_file`。
   上限是这三样，一样都不能多：`run_command` 的破坏半径无法用 diff 表达，
   `delete_file`/`move_file` 会把"改了哪个文件"变成"哪个文件没了"（没有可审的 diff），
   `task` 则让子 agent 能递归派孙 agent，审批散落到没人看的那一层。
   `writable` **缺省必须为 false**——布尔参数缺省若站在有害的一侧，模型漏传一次就等于偷偷派了个能改文件的子 agent。
   可写子任务在**派发前**过一次 `Gate.Approve`（`readonly` 模式当场拒绝，省掉一整轮白烧的调用），
   之后每次写入仍各自过门：**父子共用同一个 `*Gate`**，那次"批准派发"只覆盖了"允许它开口问"。
   `Gate` 里用 `askMu` 把"提问—等回答"串起来，一次只问一个问题。
   共用 `RepoIndex`（各拿一份等于白判定），但**各持一份 `FileState`**——
   共享的话父级读一遍就替子级开了门，第 3 条的"先读后改"只剩父级那一次读。
   子 agent 的 `Quiet` 必须为 true（过程不刷屏），它的 token 用量必须 `AddUsage` 回主 agent（否则 `/cost` 低估）。
   两副面孔各有提示词（`SubAgentPrompt` / `SubAgentWritePrompt`，公共段在 `subAgentBody`），
   两边的一致性由 `TestSubAgentIsReadOnly` 对两副面孔双向校验；可写面的装配由 `TestWritableSubAgentSharesGateAndNotFileState` 钉住。

8. **预算走服务端口径。** `EstimateTokens` 是启发式，和真实 tokenizer 有系统性偏差。
   每轮用 `resp.Usage.PromptTokens()`（含缓存段，不能只看 `InputTokens`）校准 `estScale`，
   限幅 0.4–3、两轮折中；`prepareForSend` 必须先换算再和预算比，`CompactHistory` 那边再折算回本地口径。
   花费上限同理：`checkCostLimit` 只在**发起下一次调用之前**拦，且模型不在价目表里时无法换算金额，
   必须告警"上限不生效"而不是静默放行。

9. **prompt caching 靠前缀不变。** Anthropic provider 只在 `system` + `tools` 这一段前缀够长时
   打**一个**断点（位置在最后一个工具定义上；打两处会让 system 被写两遍、多付 1.25 倍）。
   因此：工具定义必须保持字母序（见第 4 条），`buildAnthropicRequest` 不得就地修改调用方传入的 `Tools`，
   短前缀（如折叠摘要那次调用）不该带断点。缓存计价系数在 `agent/cost.go`，改价目表时同步核对。

10. **外部内容按不可信处理。** `tools/content_guard.go` 是第 1 条的另一半：文件文字和命令输出
    能造成的后果在代码里掐掉，不靠提示词求。三件事各有边界，别混淆：
    密钥类路径（`SensitivePath`）**读写都拒**——写也要拒是因为 `ask` 的 diff 会把原文件明文摊进终端；
    回给模型的任何输出过 `GuardExternalContent`（疑似密钥隐去 + 伪装指令贴告警），否则 `cat .env`
    就绕过了路径拒绝；落盘内容过 `CheckSecretWrite`，`auto`/`readonly` 拒绝、`ask` 只告警。
    `InjectionSigns` 那条是**标注不是拦截**，误报代价只是多一行提示——不要把它改成拒绝执行，
    也不要反过来把"看到告警就别执行"写进提示词当边界。新增会回传外部文本的工具，必须过同一层：
    目录名也是外部文字（`list_directory` 走同一层，构造参数里没有 `FileState` 是有意的——
    目录不是文件，不该被记进写操作确认框点名的"依据文件"）。
    第 14 条把大输出外置落盘时，落的必须是**守卫之后**那份，否则缓存目录会存着明文密钥。

11. **外部工具不享有特权。** MCP 只是把工具换了个来源，约束不能换：`tools/mcp.go` 的适配器
    照样过 `Gate.Approve`、照样过第 10 条的 `GuardExternalContent`，远端自己写的工具描述里出现
    指令式文字时也要在工具定义上标注（那会进每次请求，那里没有审批门可站）。
    **只读与否缺省按"会改动"对待**——`readOnlyHint` 由服务器自己声明，信错了就是远端改本地文件没人过目。
    远端工具一律 `mcp_<服务器>_<工具>` 前缀且截断到 64 字符（截断必须可复现，否则恢复会话时
    历史里的 tool_use 找不到工具）；注册前查 `Registry.Has`，撞名跳过而**不要**依赖 `Register` 的 panic——
    外部工具名由别人决定，撞名是配置问题，不该把 CLI 弄死。远端 schema 走 `ToolDef.RawSchema` 原样透传
    （`enum`/`items` 不能被本地类型折损），不合法才退化成空对象。
    子 agent 拿不到外部工具（第 7 条），`mcp.Client.Instructions()` 也进不了系统提示词。
    起不来的服务器只警告不致命；但配置文件本身写错必须当场退出，静默会让用户以为工具在。

12. **仓库索引只缓存元信息。** `tools/repoindex.go` 缓存的是"这个路径是文本还是二进制、多大、
    什么时候改的"，**永远不缓存内容**——内容进了缓存就等于把过时代码交给模型。
    结论按 mtime+size 复核，所以写类工具（write/edit/move/delete）成功后**必须** `index.Forget(absPath)`，
    漏一次就会让搜索拿着旧判定说假话；新增写类工具时照此办理。
    主 agent 与子 agent 共用同一份索引实例（`main.go` 的 `a.index`），各拿一份等于白判定。
    `repo_overview` 只给统计不给内容，且密钥类文件**只计数不列名**——目录里出现 `.env` 这类路径名
    本身就是在给模型指路；它和别的工具一样过 `GuardExternalContent`（目录名也是外部文字）。
    `isIgnoredDir` 里包含 `.go-agent`：那是智能体自己落盘的地方，搜它等于把回灌过的历史当代码事实。

13. **并发缺省串行、按声明放行。** `tools.Parallelizable` 要同时满足两件事才为 true：
    实现 `ParallelSafeTool` 且返回 true，**并且** `Mutating()` 为 false。没实现那个可选接口就是串行——
    "没声明"不等于"并发安全"，新增工具时忘了写 `ParallelSafe()` 只会变慢，不会变危险。
    `agent/agent.go` 的 `runBlocks` 只把**连续**的、可并发的调用组成一组跑，遇到写类就断开重新走串行，
    这样写操作永远独自占终端（审批门要逐条打字确认，两个 diff 同时问等于盲签）。
    结果必须按原始 `tool_use` 顺序回填到 `slots`：`tool_result` 与 `tool_use` 的配对是第 5 条的前提，
    乱序会让 `RepairHistory` 判出孤儿、让交替结构失真，而这种错误模型自己发现不了。
    并行分支里用 `runOneQuiet`（不打印）——并发写终端会把行撕成看不懂的碎片；
    组跑完后按原序 `echoResult`，人看到的顺序和模型看到的顺序一致。
    中断时未启动的调用留空槽，交给 `RepairHistory` 补占位，**不要**在这里伪造成功结果。
    并发上限 `maxParallelTools = 4`，一台 8 核机器上再高也只是把 IO 等待换成调度开销。

14. **大输出外置：两级治理，指针必须指得到东西。** `agent/toolcache.go` 治的不是"省 token"，
    是重复计费：一次 `read_file`/`run_command` 吐回几十万字节，原样进历史就等于此后每一轮都再付一次。
    L1（`Govern`）在**每次工具输出进历史之前**触发；L2（`Pointerize`，在 `agent/context.go`）
    在**每次构建上下文**时把早期的大输出换成同一形式的指针。两级共用同一份缓存目录
    （`.go-agent/toolcache/<tool_use_id>.txt`，文件名取自 tool_use ID，所以恢复会话后老指针仍指得到东西）。
    **L1 必须排在第 10 条的内容守卫之后**：落盘的那份要是还没隐去密钥，缓存目录就成了仓库里最亮的靶子。
    几条不能破的规矩：
    **`pointerPhrase` 同时是幂等标记**——改它等于改协议，L2 会认不出已治理过的正文而再存一遍、再套一层；
    **L2 只换正文，不许动 `ToolUseID`、块顺序、消息角色**，否则破坏第 5 条的配对关系；
    **`CompactHistory` 折叠已治理过的正文时只能砍头尾预览，指针那一行必须留着**（`foldToPointer`），
    换成一句"已折叠"就等于把回读路径一起丢掉，那一轮再也取不回原文；
    **头尾之和必须明显小于该级的门槛**，否则切完"头 + 指针 + 尾"比原文还长，这一级形同不存在
    （L1 门槛 6000 对头尾 5500，L2 门槛 3000 对头尾 1100）；
    **落盘失败要给诚实的降级文本**（"全文未能落盘"+ 怎么缩小范围），而不是造一个读不到的路径；
    **缓存有容量闸门**（`cacheLimits`，超限按 mtime 从旧到新删且至少留一份），
    否则只是把上下文问题换成磁盘问题。
    父子 agent 共用同一个 `*ToolCache` 实例（`Config.Cache`，`main.go` 装配）：
    各建一份的话，主对话里生成的指针在子 agent 那边读不到。

15. **可回读的内容不重复发送。** 第 14 条治的是"一次调用的输出太大"，这一条治的是
    "同样的字节被每一轮再付一遍"，三处下手，各自的闸门都不能少：

    **① 写入参数瘦身（`Agent.SlimBodyParams`，在 `agent/context.go`）**：`write_file` 的 `content`
    是模型亲手写的几千字节，写完那一刻它就在磁盘上了，却仍留在 `tool_use` 里被重复发送。
    工具用可选接口 `tools.BodyParamTool`（`BodyParams()`）声明**哪个参数是大正文**，
    **缺省是没有**——没声明就一律不动，最坏结果只是贵，不会丢东西。四道门同时成立才换：
    声明过、配对结果存在且非 `is_error`、不属于最近 `l2KeepRecent` 次调用、值超过 `l2ParamFrom`。
    两条容易搞错的取舍：
    **`edit_file` 只声明 `new_string` 不声明 `old_string`**——成功后新内容在文件里可回读，
    旧原文已经被替换掉，历史是它唯一的去处；
    **没拿到结果的调用不能换**——还在等审批时 diff 正是从参数里生成的（第 2 条），
    写失败时参数是下一轮重试的唯一依据。
    瘦身**不落盘**：全文就在 `file_path` 那个文件里，再抄一份进 `.go-agent/toolcache`
    等于把同一内容落两次、多一个可被翻看的副本。
    `Input` 是引用类型，必须换成新 map 而不是就地改（会话快照和调用方可能持有同一张表）。

    **② 重复读取省略（`Agent.duplicateNotice`，在 `runOneQuiet` 里、L1 之前）**：同一任务里
    常为不同目的把同一个文件读上三四遍。判定只用"与历史里同名同参调用的结果**逐字相等**"这一条，
    而这一条同时兜住三种风险，所以不需要额外闸门：文件被改过→输出必然不同、
    那一份已被指针化或折叠→文本对不上、上次是错误结果→带 `is_error` 不参与比较。
    因此它是**自愈**的：那一份一旦被折叠掉，下一次同样的调用自然重新发送全文。
    两条限制：只给非改动类工具用（写类结果是"改动确实发生过"的凭据，逐字相同也要逐条留），
    且结果短于 `dupMinBytes` 时不省（省不下什么，却多让模型背一个"这次怎么没返回内容"的疑问）。
    `argKey` 里的参数值**不能截断**——两个只在截断点之后才不同的路径会被判成同一次调用，
    省掉的就是真数据。

    **③ 别把格式开销当正文**：`read_file` 的行号前缀用 `%d| ` 而不是定宽的 `%6d | `。
    定宽前缀每行固定 9 字节，一次 800 行的读取里光前缀就多花 3.3K 字节（同一份内容现在 3.9K）。
    行号是给模型定位用的，不是给终端对齐用的。

## 测试风格

- 每个包都有独立测试。网络用 `httptest` 起本地 SSE 服务，不碰真实 API。
- 文件操作用 `t.TempDir()`，命令测试只写 `/tmp`，绝不碰真实仓库或用户目录。
- 测试是**行为门**，不是摆设。历史上多个真实缺陷（`rm -rf /etc` 漏判、`git clean` 分级错、
  会话 ID 同秒撞车、被截断的 tool_use 复活）都是测试先红、再修产品代码修好的。
- 并发那一路有两个坑反复踩到，写用例时先自查：
  **计数器必须多个工具共用一份**（各自一份的话"峰值并发数"恒等于 1，断言永远真）；
  **顺序断言要看内容不看次数**（比如慢工具先发起、快工具先完成，才证明回填按的是原始顺序）。
  涉及共享状态的用例要单独跑 `-race`（`go test -race -run TestGate ./tools/`），
  全量 `-race` 在这台 8 核机器上太吃资源，别顺手带上。
- 测试红的时候先查根因改产品代码，**不要**为了变绿去弱化断言或跳过。
- `main_test.go` 用 `isolateEnv` 清空所有厂商密钥与日志目录环境变量，别让它泄漏到真实 shell 或写出 `./logs`。
  它同样要压掉 `GO_AGENT_MCP_CONFIG`：开发机上真实存在的 `.go-agent/mcp.json` 会让每个用例去拉子进程，
  工具数量断言便莫名其妙地红。
- 要测"真的起一个进程"那一路就用测试二进制自己充当假服务器（`mcp/stdio_test.go`、`main_test.go`
  里的 self-exec），不依赖机器上装没装 npx/python，也不碰网络；协议本身用 `net.Pipe()` 的内存假服务器覆盖。
  测试里绝不能用 `/usr/bin/env <包名>` 之类去拉外部包——CI 没有网络，会卡在握手。
- 摘要那一路会真的调 provider，测试里用 `scriptedProvider` 记录请求，断言"一次调用都没多"；
  走 `a.Run` 的端到端断言不稳定（摘要请求会从脚本队列里吃掉一个 step），要折叠就直接调 `a.compact`。
- 第 14 条那一路有三个只有写用例才能发现的坑：
  **某一级的门槛必须高于上一级的头尾之和**，否则那一级的切法盖住全文，测出来就是"一条都没换掉"，
  而产品代码看起来完全正常；
  **断言"切在 rune 边界"之前要先断言用例的切点确实落在汉字中间**（`utf8.RuneStart(full[l1HeadBytes])`），
  常量一改这种用例就会静默变成永远真；
  **测折叠形式时预算要卡在"只走第一级"的位置**，一旦走进丢弃整轮，消息下标全变，
  断言就落在别的消息上。
  校准那一路的用例要**把每条输出压到 `l2CacheFrom` 以下**，不然省下的量记在指针的账上，
  校准是否影响折叠根本测不出来（`TestCalibratedScaleTriggersCompaction` 曾因此变红）。
- 涉及缓存目录的用例都用 `t.TempDir()`，别让全文落到真实仓库的 `.go-agent/` 里。
- **省略类改动必须带反向门**："该省的省了"很好测，真正会写坏的是"不该省的也被省了"。
  第 15 条每一处都要配齐反面用例：失败的写入、还没拿到结果的调用、未声明的工具、
  内容已经变过的第二次读取、只在截断点之后才不同的两个参数。
  瘦身判定认的是工具自己的 `BodyParams()` 声明，所以用例要注册**真工具**，
  用桩子就等于在测桩子；同理 `TestBodyParamsNameRealSchemaKeys` 盯着"声明写错名字会静默失效"。
- 手工搭历史做配对断言时，**首条必须是 user**：`RepairHistory` 会砍掉开头的非 user 消息，
  于是紧跟其后那条 `tool_result` 被判成孤儿，测试假红（`TestDuplicateReadKeepsPairing` 撞到过）。

## 文档同步

改完代码，README.md 和本文件若与事实不符就当场更新（选项、命令、安全行为、依赖方向）。
过时的文档比没文档更糟。
