# Deep Research 系统主流程（Canonical Architecture）

版本：v1.3
日期：2026-07-17
定位：本文是系统的唯一主流程说明；字段级协议、安全、评测和部署细节见[详细设计](./deep_research_multi_agent_design.md)。

## 0. 一句话结论

整个系统只有两个智能循环：

1. **外层 Main 调度循环**：Main Agent 被多次独立调用，根据最新全局状态增量修改任务 DAG。
2. **内层 Researcher ReAct 循环**：每个 Researcher 围绕一个 SubTask 独立执行有限轮搜索、阅读、判断和补查。

两个循环之间通过结构化状态和事件通信，不共享长对话，也不依靠模型“记住上轮发生了什么”。

完整链路是：

~~~text
用户问题
→ 建立 ResearchRun 和 GlobalResearchState
→ Main 生成粗粒度 DAG
→ Scheduler 找出 READY 节点并行派发
→ Researcher 使用自己的 LocalResearchState 做有限 ReAct
→ AgentResult、Claim、Evidence 被确定性合并到全局层
→ Main 在战略边界再次被调用并 Patch DAG
→ 覆盖达标后写作
→ Citation Audit
→ 不通过则生成 repair task，通过则发布
~~~

### 0.1 三个完整 Case 审计后的实现约束

2026-07-16 对保险比较、黄金走势和房地产财政三个完整 trace 的审计确认并修复了以下约束：

- **先耗尽当前 DAG，再重规划**：已有 READY/PENDING 节点能够覆盖 gap 时，Main 不得新增同义任务；跨中英文的 objective/coverage 语义去重是第二道门禁。
- **active gap 与来源 limitation 分离**：Reader 的每条来源限制保留在 observation；只有 Researcher 的 `add_gaps/resolved_gaps` 可以修改 Local active-gap ledger，Global Reducer 再做语义去重和数量上限。
- **内容完整度不等于来源权威性**：`search_native_content` 只表示搜索接口返回了 Reader-ready 正文；Source 另存 `source_type/authority_score`。官方/一手、独立媒体和聚合/社区内容分别校准。
- **Coverage 是结构化交付状态**：除兼容字段 `coverage[target]=status` 外，`coverage_details` 记录任务、Claim、来源类型、高权威来源和未满足的来源要求。
- **Writer 使用 coverage-balanced dossier**：证据按 coverage 槽位轮转，并按交叉验证、来源权威和定量信息排序。比较、时间序列和财政口径任务有专门的同口径写作约束。
- **Audit 区分研究修复和文本修复**：引用格式、删改不支持表述等问题直接进入 audit-guided revision；确实缺证据才启动 Researcher，并使用预留的 search/tool 预算。
- **精确 token 与多语言引用**：Tokenizer 的 mapping/`BatchEncoding` 必须从 `input_ids` 计数；Markdown URL 和中文 `）。；` 等标点由统一解析器处理。

### 0.2 质量回归后的必修清单与验收条件

2026-07-17 对保险横向比较、黄金时间序列和房地产财政口径三个新 trace 的复核表明，工程正确性已经改善，但控制面仍会把有证据的任务锁在 `partial`。以下项目已按依赖顺序实现，并作为后续回归的固定验收清单：

1. **Researcher 终止协议**：输出示例不得把 `finish` 固定为 false；到达最后一步时由运行时确定性终止。`sufficient` 与“有用但仍有缺口的 partial”都必须产生明确终态，不能依赖模型记得设置一个布尔值。
2. **Task-scoped context**：Researcher 只接收与本 SubTask coverage/objective 相关的 global gaps、claims 和 conflicts。Query 去重默认限于本 SubTask；Verify/Repair 可以复用 discovery query，但必须改变来源、时间、实体或验证目标。Trace 同时记录 raw、accepted 和 duplicate-filtered queries。
3. **Partial task 可继续执行**：Main 支持 `REFINE_TASK`，将已有 partial/failed task 重新置为 pending，并更新 objective、预算和来源要求。禁止出现“Main 要 continue，但所有 ADD_TASK 都被去重且没有 READY task”的空转。
4. **自适应任务预算**：Search、Reader 和模型 step 分开计量。任务按照 standard/verify/comparison/time_series/quantitative/repair profile 获得不同的 steps、search、reader/tool 上限；全局预算仍是硬门禁。增加预算不能只制造更多低质量 Claim。
5. **规划粒度校验**：比较任务先定义统一实体集合和字段 schema；时间序列先定义区间/频率；定量问题先定义分子、分母、单位和口径。一个 SubTask 不得同时承载无法在其预算内完成的大型矩阵。
6. **来源要求落到 coverage/Claim**：`primary + independent` 不再是所有 task 的机械默认。关键排名、财务、评级和财政数字要求 primary；观点和预测要求独立交叉验证。Source authority 补齐真实机构域名，官方来源被 Reader 拒绝时必须记录具体原因。
7. **Coverage Gate 以证据充分性为准**：task status、来源要求和 coverage status 解耦；一个 partial task 也可以覆盖已满足的 coverage cell。Gate 使用 `coverage_details` 的 Claim、来源、缺口和质量，而不是让“全 partial”永远只能得到 0.5。
8. **Evidence 聚合层**：Writer 前先按 `entity × metric × period × geography × accounting_scope` 合并重复 Claim，生成稳定的 coverage dossier。比较、时间序列和财政问题使用结构化矩阵，不直接向 Writer 倾倒上百条扁平 Claim。
9. **可审计派生计算**：CAGR、占比、变化量、排名和情景结果必须保存 CalculationRecord，包括公式、输入 Evidence、单位、期间、分母和假设；Writer 不得自行补算无记录数字。
10. **确定性引用**：Writer 引用 Source/Evidence ID，Renderer 从 Evidence Store 注入精确 Markdown URL；模型不再自由拼写、缩短或改写 URL。Audit 使用稳定的 cited-evidence dossier，而不是每轮重新随机采样。
11. **Audit 修复状态机**：达到 search-audit 上限后，rewrite-only issue 仍允许一次受约束的最终修订和确定性校验；只有新证据需求才消耗 research repair round。
12. **可观测性与评测**：记录 rejection 原因（grounding/scope/number）、query filter 原因、budget saturation、dossier omission 和 citation rendering。全量分数必须同时报告 Judge total/valid/error，并在新旧运行的共同有效 ID 上比较。

验收条件：

- 最后一步无论模型是否输出 `finish=true`，每个 Researcher 都产生可解释终态；
- Main 的 `continue` 决策之后必须存在 READY/PENDING task，否则确定性转为 write/partial；
- Verify/Repair 不会被全局 query ledger 静默清空；
- Writer 使用的每个数值要么是直接 Evidence Claim，要么有 CalculationRecord；
- 最终 Markdown 中的所有 URL 都来自 Source Store；
- Audit 的 rewrite-only 问题不会因为 search round 用尽而直接终止；
- 静态、异步、动态重规划、推理网关和真实 trace 不变量测试全部通过。

### 0.3 本轮实现落点和默认值

| 能力 | 确定性实现 | 关键状态/默认值 |
|---|---|---|
| 最后一步收敛 | `agents.py` 强制 final synthesis，并区分 requested/effective/forced finish | 无证据时最后一步仍可搜索；已有证据时保留最后模型 turn 做总结 |
| Task-scoped context | `build_global_context_slice` + `query_ledger_by_task` | 本任务 query 硬去重；global/dependency query 仅提示 |
| Partial 重开 | `dag.py: REFINE_TASK` + `RunStore.clear_task_execution` | 保留历史 Evidence/query，删除旧 local/checkpoint/bundle；默认最多 2 次 attempt |
| 规划分片 | `split_overloaded_tasks` | 默认每 task 最多 2 个 coverage target；下游依赖自动 fan-out |
| 自适应预算 | `infer_research_profile/apply_adaptive_task_budgets` | standard `3 steps/18 tools`；复杂任务上限 `5 steps/36 tools/10 searches` |
| Claim 来源要求 | Reader 输出 claim-level `required_source_types` | `primary`、`independent`、`corroborated`；task 默认不再机械要求两类来源 |
| Evidence Gate | `coverage_details.quality_score` + weighted gate | partial task 的已满足 coverage cell 可以成为 covered |
| Evidence 聚合 | Claim dimensions + 语义/维度去重 + coverage round-robin | dimensions 为 entity/metric/period/geography/unit/denominator/accounting_scope |
| 派生计算 | `calculator.py` | 数字必须出现在引用 Evidence excerpt；仅开放白名单运算 |
| 确定性引用 | `[[EVIDENCE:id]]` → `render_evidence_citations` | URL 只从 SourceRecord 注入；未知 token 不生成 URL |
| Audit final pass | 稳定 report dossier + `audit_history` | search 上限后允许一次 rewrite-only 修订及再次校验 |
| 评测可比性 | `scoring.py` + `scripts/compare_race_runs.py` | 汇报 total/valid/error/valid_rate，并按共同有效 ID 计算 delta |

## 1. 先把角色简化

第一版不要设计成十几个 Agent 互相发送自然语言消息。推荐只保留三个逻辑角色。

### 1.1 Main Agent

职责：

- 理解用户目标，形成 Research Brief。
- 生成初始粗粒度任务 DAG。
- 根据新证据、覆盖缺口、冲突和预算增量修改 DAG。
- 决定继续研究、转入写作、输出有限结论或终止。
- 对最终交付负责。

Main Agent **不负责**：

- 亲自读取每个网页。
- 在每个工具结果返回后都推理一次。
- 直接修改数据库状态。
- 手动决定每个协程何时启动。
- 依赖一条从头到尾不断增长的聊天历史。

文中的 Writer 不是第四个长期自治 Agent，而是 Main 在写作阶段触发的一次受限报告生成 Activity；它只能使用通过门禁的 Claim/Evidence。

### 1.2 Researcher

职责：

- 一次只接受一个边界清晰的 SubTask。
- 在子任务内部运行有限轮 ReAct。
- 搜索、选择来源、请求 Reader 抽取、判断缺口、定向补查。
- 输出结构化 AgentResult，以及 Claim/Evidence 的 ID。

不同“专家”无需一开始实现成不同 Agent 类。以下角色可以先由同一个 Researcher Runtime 加不同 task profile 实现：

- landscape researcher
- targeted researcher
- contradiction verifier
- data analyst
- citation repair researcher

### 1.3 Citation Auditor

职责：

- 把报告拆成原子事实声明。
- 检查每个重要 Claim 是否被引用内容支持。
- 检查引用是否真的指向所述来源位置。
- 输出 pass，或结构化 repair items。

它可以复用 Researcher Runtime，但必须使用独立提示词、只读证据权限和更严格的完成门槛。

### 1.4 哪些不是 Agent

以下组件应由确定性代码实现：

- Workflow Engine
- Scheduler
- State Reducer
- Validator / Policy Engine
- Budget Manager
- Search、Fetch、Reader、PDF Parser、Compute 等工具服务
- Evidence / Claim / Artifact Store
- Event Log

判断标准很简单：需要开放式语义判断时使用 LLM；涉及状态提交、权限、依赖、预算和重试时使用代码。

## 2. 状态模型：不是一个“大 Global JSON”

推荐采用四层持久状态。模型只能看到为当前决策构造的视图，而不是所有原始数据。

| 层 | 作用域 | 内容 | 谁可以写 |
|---|---|---|---|
| Event Log | 一个 ResearchRun | tool、task、model、audit 的不可变事件 | Workflow/Tool Runtime |
| GlobalResearchState | 一个 ResearchRun | 目标、DAG、进度、coverage、gap、conflict、预算、Claim/Evidence 索引 | Global Reducer |
| LocalResearchState | 一个 SubTask | 本地计划、query/source ledger、本地发现、缺口、预算、停止判断 | Local Reducer |
| Artifact Store | 一个 ResearchRun | 网页快照、PDF、长文本、表格、Reader 输出、报告草稿 | Tool/Artifact Service |

### 2.1 Global Information 的分层视图

Global Information 是逻辑概念，不等于把全部内容塞进 Main Agent 的 prompt。

~~~text
G0 Control
  run status / state version / deadline / budget / permissions

G1 Plan
  research brief / question tree / task DAG / coverage rubric

G2 Knowledge Index
  accepted claims / evidence references / gaps / conflicts / source map

G3 Artifacts
  raw pages / PDFs / extracted passages / datasets / drafts
~~~

Main Agent 通常只接收 G0、G1、G2 的压缩视图，以及本次新增事件摘要。只有需要核验某个冲突时，才按 ID 读取少量 G3 内容。

### 2.2 GlobalResearchState 是每次运行独立的

不要建立一个所有用户、所有任务共同写入的“全局脑”。每个 ResearchRun 有自己的权威状态：

~~~text
GlobalResearchState(run_id)
~~~

跨运行复用的网页缓存、向量索引或来源信誉数据属于共享基础设施，不属于该 run 的研究事实。

### 2.3 LocalResearchState 如何产生

LocalResearchState 不是 Researcher 自己随意写的一段总结，也不是完整聊天记录。它由三种来源共同形成：

1. **Scheduler 初始化**：根据 SubTaskContract 和相关 GlobalContextSlice 建立 `LocalResearchState v0`。
2. **Tool Runtime 写事实**：记录真实 query、搜索结果、抓取状态、来源 ID、工具错误和消耗。
3. **Researcher 提议语义 Patch**：提出下一步计划、局部 Claim、gap、conflict 和完成判断。

Local Reducer 校验版本、字段权限和去重规则后才提交 patch。因此 Researcher 不能伪造“工具已成功执行”或自行增加预算。

## 3. 两个嵌套循环

~~~mermaid
flowchart TB
    U["用户问题"] --> I["初始化 ResearchRun 与全局状态"]
    I --> M["调用 Main：生成或修改 DAG"]
    M --> V["Validator 校验 DecisionPatch"]
    V --> S["Scheduler 计算 READY 节点"]

    S --> C["建立 SubTaskContract 与 LocalState v0"]
    C --> R["Researcher 决策"]
    R --> T["Search / Fetch / Reader / Compute"]
    T --> L["Local Reducer 更新 LocalState"]
    L --> D{"子任务结束？"}
    D -->|"否"| R
    D -->|"是"| A["提交 AgentResult、Claim、Evidence"]

    A --> G["Global Reducer 更新全局状态"]
    G --> W{"到达战略唤醒点？"}
    W -->|"需要重规划"| M
    W -->|"仍有可执行节点"| S
    W -->|"研究门禁通过"| WR["Writer 生成证据化报告"]
    WR --> CA["Citation Auditor"]
    CA -->|"失败：生成 repair items"| M
    CA -->|"通过"| O["发布报告与证据清单"]
~~~

这张图有一个关键含义：Researcher 的每一步不会直接唤醒 Main。多数细粒度变化由 Reducer 和 Scheduler 消化；只有影响全局策略的事件才触发新的 Main 调用。

## 4. 外层 Main 调度循环

### 4.1 Main 是一个逻辑角色，但有多次 input/output

Main Agent 在业务上贯穿整个 ResearchRun；在执行上是多次相互独立的 LLM 调用。

典型调用序列：

~~~text
Main call #1：理解任务并生成 DAG v1
→ Researcher A/B/C/D 并行
Main call #2：看到 coverage gap，生成 DAG Patch v2
→ Researcher E 与 Verifier F 并行
Main call #3：判断证据足够，进入写作
→ Citation Audit 失败
Main call #4：增加 repair tasks，删除或降级无支持 Claim
Main call #5：Audit 通过，结束
~~~

每次 Main 调用的输入都由 Context Builder 从权威状态重新构造：

~~~text
MainInput =
  ResearchBrief
  + 当前 DAG 摘要
  + coverage / gaps / conflicts
  + 本批新增 AgentResult 摘要
  + budget / deadline
  + allowed operations
~~~

输出必须是结构化 `DecisionPatch`，而不是让模型直接执行：

~~~json
{
  "base_state_version": 23,
  "operations": [
    {"op": "ADD_TASK", "task_id": "st_11", "type": "research"},
    {"op": "ADD_DEPENDENCY", "from": "st_11", "to": "verify_gate"}
  ],
  "next_wakeup": {"mode": "ON_BATCH_OR_CONFLICT", "min_completed": 2}
}
~~~

### 4.2 何时调用 Main

应该调用：

- Scope 完成，需要首次规划。
- 当前 DAG 的计划前沿已经耗尽，且仍有未覆盖的新问题。
- 当前没有 READY/RUNNING task，但研究尚未完成。
- 出现关键证据冲突或关键任务失败。
- coverage、边际收益或预算跨过阈值。
- Citation Audit 产生 repair items。
- 用户修改目标或取消任务。

不应该调用：

- 每返回一个搜索结果。
- 每抓取完一个 URL。
- 每生成一条 Reader note。
- 只是因为某个 worker 空闲。
- 当前 gap 已经由尚未执行的 pending/ready task 负责。

### 4.3 外层循环的确定性伪代码

~~~python
while not terminal(run):
    events = event_store.read_new(run)
    global_state = global_reducer.apply(global_state, events)

    scheduler.dispatch(ready_tasks(global_state))

    if wakeup_policy.should_call_main(global_state, events):
        main_input = context_builder.for_main(global_state, events)
        decision_patch = main_llm(main_input)
        validated_ops = validator.compile(decision_patch, global_state)
        global_state = state_store.compare_and_swap(validated_ops)
        continue

    if research_gate.passed(global_state):
        start_writer(global_state)
        continue

    wait_for_next_event()
~~~

LLM 决定研究策略，代码决定是否执行这个策略。

## 5. 内层 Researcher ReAct 循环

### 5.1 每个 Researcher 是 ReAct 吗

是，但它是**有边界的局部 ReAct**，而不是无限自治 Agent。

~~~text
Reason：当前证据能否回答本 SubTask？最大的 gap 是什么？
Action：SEARCH / FETCH / READ / COMPUTE / FINISH
Observation：工具返回结构化结果
Update：Local Reducer 更新 LocalResearchState
Repeat：直到完成、部分完成或预算耗尽
~~~

建议约束：

- 一个 Researcher 只负责一个 SubTask。
- 最大 ReAct step、tool calls、tokens 和 wall time 都由合同规定。
- 不允许任意创建新的 Researcher。
- 若发现超出本任务边界的新问题，只输出 `escalation/gap`，由 Main 决定是否建新任务。
- 工具结果先落 Event/Artifact Store，再给模型构造紧凑 observation。

### 5.2 每一步 Researcher 收到什么

Researcher 不需要拿完整 GlobalResearchState。每一步只接收一个最小 `ResearcherInput`：

~~~text
ResearcherInput =
  immutable SubTaskContract
  + relevant GlobalContextSlice
  + compact LocalResearchState
  + recent tool observations
  + remaining local budget
  + allowed actions
~~~

其中：

- `SubTaskContract` 定义目标、边界、交付格式、来源政策和预算。
- `GlobalContextSlice` 只包含与该任务有关的已有 Claim、Evidence、术语和冲突。
- `LocalResearchState` 提供本任务已经搜索什么、读过什么、还缺什么。
- 原始网页全文通常只保存在 Artifact Store，需要时按 ID 读取局部片段。

### 5.3 每一步 Researcher 输出什么

~~~json
{
  "base_local_version": 7,
  "assessment": {
    "coverage": "PARTIAL",
    "primary_gap": "缺少官方口径"
  },
  "proposed_patch": {
    "add_queries": ["site:gov.example official market definition"],
    "add_gap_codes": ["OFFICIAL_DEFINITION_MISSING"]
  },
  "actions": [
    {"type": "SEARCH", "query_ref": 0}
  ],
  "finish": false
}
~~~

这只是“决策和状态修改提议”。Tool Runtime 执行动作，Local Reducer 提交合法字段。

### 5.4 一次 SubTask、多次独立 inference

一个 Researcher 逻辑上持续负责同一个 SubTask，但每个 ReAct step 都是独立、有限的模型请求：

~~~text
LocalResearchState v0
→ Input/Output #1: SEARCH(q1)
→ 释放 inference admission；执行 Search/Fetch/Reader
→ Reducer + checkpoint: LocalResearchState v1
→ Input/Output #2: SEARCH(q2) 或 FINISH
→ 释放 inference admission；执行工具
→ Reducer + checkpoint: LocalResearchState v2
→ Input/Output #3: FINISH
→ AgentResult + terminal bundle
~~~

模型请求结束后不保留一条不可迁移的“Agent 会话”，搜索等待期间也不占用正在生成的请求。三层状态严格分离：

| 层 | 所有者 | 作用 |
|---|---|---|
| LocalResearchState / checkpoint | Workflow + RunStore | 正确性、恢复、审计，唯一持久事实 |
| ResearcherInput View | Context Builder | 本轮所需的稳定前缀、状态快照和最近 observation |
| KV / Prefix Cache | vLLM | 可淘汰性能优化，命中与否不能影响正确性 |

每个 step 的 request ID 由 `(run_id, subtask_id, local_version)` 生成；输出必须回传相同 `base_local_version`。工具结果归并后才把版本从 `vN` 提交为 `vN+1`。进程若在 step 边界中断，从 `checkpoints/<subtask_id>.json` 恢复来源、证据、Claim、Calculation、usage 和最后决策，不依赖聊天历史。

Context Builder 使用以下顺序构造输入，以保持 Prefix Cache 友好的共同前缀：

~~~text
System + protocol
+ original task
+ immutable research brief / SubTask contract
+ relevant global slice
+ compact LocalResearchState view
+ newest observations
+ remaining budget
~~~

输入上限按 `max_model_len - max_output_tokens - safety_tokens` 计算。配置本地 tokenizer 时使用精确 chat-template token 数；未配置时使用对中英文都偏保守的 UTF-8 估算，并优先淘汰最旧 observation 和低相关全局 Claim，绝不截断权威状态文件。

Tokenizer 返回值必须归一到 `input_ids` 后计数：`list[int]`、嵌套 batch、tensor、对象属性和 `BatchEncoding/dict` 都支持。严禁对 mapping 直接 `len(encoded)`，因为那得到的是字段数而不是 token 数。回归测试必须同时断言长 prompt 的计数显著大于短 prompt，不能只断言“未超过上限”。

Inference Gateway 在请求进入 vLLM 前提供：

- `main/researcher/auditor`、`reader`、`writer` 三类并发配额；
- 控制决策优先于 Reader，Reader 优先于长输出 Writer；
- 每个 run 的并发上限和 in-flight token admission；
- JSON Schema structured output；旧 vLLM 不支持高级参数时一次性降级为 Prompt + Validator；
- Qwen JSON 控制调用默认关闭长 thinking，Writer 仍按普通生成处理。

vLLM Prefix Cache 是推荐优化：相同的稳定 token 前缀可以复用，工具等待期间缓存块允许被淘汰，不需要为每个 SubAgent 永久固定 KV。

### 5.5 Researcher 最终给 Main 什么

Main 通常不读取整个 LocalResearchState，而是接收标准化 `AgentResult`：

~~~text
AgentResult
  subtask_id / status
  answer_summary
  claim_ids[]
  evidence_ids[]
  unresolved_gaps[]
  conflicts[]
  suggested_followups[]
  cost / tool calls / latency
~~~

完整 LocalResearchState 留给恢复、审计、调试、去重和评测使用。

## 6. 动态 DAG 到底如何工作

### 6.1 首次规划：完整骨架，不穷举未来任务

Main 第一次生成的是“完整的粗粒度骨架 DAG”：

~~~mermaid
flowchart LR
    P["Initial Plan"] --> A["背景与定义"]
    P --> B["官方数据"]
    P --> C["技术与产品"]
    P --> D["风险与反例"]
    A --> M["Merge + Coverage Gate"]
    B --> M
    C --> M
    D --> M
    M --> X{"存在 gap 或 conflict？"}
    X -->|"是"| E["Expansion Point"]
    E --> T["Targeted Research / Verify"]
    T --> M2["Merge v2"]
    M2 --> V["Verify Gate"]
    X -->|"否"| V
    V --> W["Write"]
    W --> C1["Citation Audit"]
    C1 -->|"fail"| E
    C1 -->|"pass"| R["Release"]
~~~

初始 DAG 包含：

- 现在已经明确的研究任务。
- 必然存在的 merge、coverage、verify、write、audit、release 节点。
- 对未知问题保留 expansion point。

不能要求 Main 在还没看到资料时就准确猜出所有后续查询。

### 6.2 后续规划：只提交 Patch

新证据显示“亚洲市场缺失”和“市场规模口径冲突”时，Main 不重建整个 DAG，而是：

~~~text
ADD_TASK：补充亚洲市场数据
ADD_TASK：核验两个市场规模口径
ADD_DEPENDENCY：两个新任务必须在 Verify Gate 前完成
CANCEL_TASK：取消与现有证据重复的低价值任务
~~~

已完成节点、已采集 Evidence 和执行历史保持不变。

### 6.3 谁决定本轮并行什么

Main 定义：

- task 节点
- dependencies
- priority
- coverage target
- task budget
- guard condition

Scheduler 决定：

- 当前哪些节点 READY
- 实际放行几个并发 worker
- 工具池、域名、模型和租户限流
- 公平性、重试和 backpressure

节点 READY 的条件：

~~~text
PENDING
AND dependencies satisfied
AND guard passed
AND not cancelled
AND task/run budget available
AND required resources available
~~~

所以四个概念必须分开：

| 概念 | 含义 |
|---|---|
| DAG | 整个研究任务的依赖图 |
| Ready Set | 此刻具备执行条件的节点集合 |
| Scheduler Wave | Scheduler 实际放行的一批节点 |
| Main Round | Main 被唤醒并修改一次策略/DAG |

一个 Main Round 可以产生多个 Scheduler Wave；worker 完成后，新的依赖节点也可以自动 READY，而无需再次调用 Main。

## 7. 从用户问题到最终报告的十二阶段

### 阶段 0：Create Run

输入用户问题、输出格式、时间范围、来源政策、权限、deadline 和总预算，建立 `run_id`、Event Log 和 GlobalResearchState v0。

### 阶段 1：Scope

Main 或专门的 Scope 调用生成 Research Brief：

- 问题边界与关键术语。
- 目标读者和报告形式。
- 时间、地区和实体范围。
- 必须回答的 rubric。
- 哪些事实需要高质量或第二独立来源。
- 已知歧义和需要向用户确认的问题。

### 阶段 2：Complexity Gate

判断是否需要多 Researcher：

- 简单、强顺序、少来源问题走单 Researcher。
- 可分解、需要多视角、来源广或上下文很长的问题走 Main + 多 Researcher。

单/多代理应复用同一 Evidence、Claim、Audit 和状态协议，便于等预算评测。

### 阶段 3：Initial Plan

Main call #1 生成 Question Tree、粗粒度 DAG、SubTaskContract 模板和 wakeup policy。Validator 检查 schema、环、权限和预算后原子提交。

### 阶段 4：Dispatch

Scheduler 计算 Ready Set，按优先级、关键路径、预期信息增益/成本和资源上限派发任务。

### 阶段 5：Local Research

每个 Researcher 获得独立 LocalResearchState，运行有限 ReAct。Search/Fetch/Reader 的真实结果先写 Event/Artifact Store。

搜索工具使用显式能力表统一管理 `search_pro_jina`、`search_prime`、`search_pro_ms`、
`search_live`、`search_lite` 和 `search_plus`。当前部署默认使用 `search_plus`（Baidu）。
在 URL fetch 策略为 `auto` 时，Baidu 返回的完整内容直接交给 Reader，不再抓取网页；支持搜狗的 endpoint 也采用相同策略，其余引擎
把搜索结果当作 discovery/snippet，并继续走安全 URL 校验和 Fetch。`always`/`never` 可覆盖默认策略。
搜索引擎、content kind、实际 fetch 状态和提取方法都进入 source artifact 与 trace，避免把
“未尝试抓取”误记为抓取失败。
逻辑引擎名与后端模型名解耦：`search_live` 优先映射到 `search_pro_sogou`，并在后端返回
1211（模型不存在）时回退到旧别名；`search_lite` 同理映射到 `search_pro_quark`。

### 阶段 6：Evidence Ingestion

AgentResult 被验证和标准化：

- Source 去重与来源独立性判断。
- 将 extraction quality 与 publisher authority 分开记录；`search_native_content` 不能自动获得 high confidence。
- Evidence 保存 locator、excerpt、hash 和抓取时间。
- Reader Claim 必须保留 excerpt 的主体、地区、时期、单位、分母和统计/会计口径；新增数字或扩大范围会被拒绝。
- Claim 与 Evidence 建立 supports/refutes/qualifies 关系。
- Global Reducer 更新 coverage、gap、conflict 和 budget。

### 阶段 7：Strategic Replan

只有当前 DAG 已没有 READY/PENDING 工作，或发生关键冲突/失败/Audit repair 时，Main 才读取全局摘要并输出增量 DAG Patch；否则 Scheduler 继续推进已有 DAG。新增 research task 若复用了已有 coverage target 或与现有 objective 高度相似，会被确定性编译器拒绝；对于 partial/failed task，Main 使用 `REFINE_TASK` 改变方法后重开原 task id；verify/repair task 可以有意重叠。

### 阶段 8：Research Gate

满足以下条件才进入写作：

- 关键 coverage item 达标。
- P0 Claim 有足够来源支持。
- 重大冲突已解释或明确标注无法判断。
- 最近任务的边际信息增益低于阈值，或预算接近上限。

若 deadline/budget 用尽但证据不足，进入 `PARTIAL`，不能伪装成完整结论。

### 阶段 9：Outline and Write

Writer 先依据 coverage-balanced evidence dossier 生成 evidence-backed outline，再写正文。Dossier 在每个 coverage 槽位之间轮转，并优先 corroborated、官方/一手和可比较的定量证据。Writer 只选择 `[[EVIDENCE:id]]`，Renderer 再从 SourceRecord 注入 `[说明](EXACT_URL)`；不允许模型拼写 URL 或从记忆补充未入账事实。

### 阶段 10：Citation Audit and Repair

Auditor 原子化正文 Claim，检查 entailment、citation correctness、coverage 和来源质量。

- pass：进入 Release。
- fail/rewrite：已有证据足够但引用、范围或关系写错，直接带着上一版草稿和 Audit issues 做局部修订。
- fail/research：确实缺少事实证据，回到 Main 建立定向 repair task。普通研究最多使用总 search/tool/token 预算的 80%，剩余容量只在首次 Audit 失败后释放。

### 阶段 11：Release

输出：

- 最终报告。
- 引用和来源清单。
- 关键结论对应的证据包。
- 未解决问题与限制。
- 运行 trace、成本和评测指标。

## 8. 组件之间只传八类核心对象

| 对象 | 生产者 → 消费者 | 用途 |
|---|---|---|
| ResearchBrief | Scope → Main/Researcher | 固定研究边界和验收标准 |
| MainInput | Context Builder → Main | 一次战略决策的紧凑视图 |
| DecisionPatch | Main → Validator | 增量修改 DAG 或阶段 |
| SubTaskContract | Scheduler → Researcher | 固定单个子任务目标、权限和预算 |
| ResearcherInput | Context Builder → Researcher | 某个 ReAct step 的最小上下文 |
| ResearcherStepDecision | Researcher → Runtime/Reducer | 下一步动作和 Local Patch 提议 |
| AgentResult | Researcher → Global Reducer/Main | 子任务标准化交付 |
| Claim/Evidence/Artifact | Tools/Agents → Stores | 最终可追溯的研究事实层 |

不要把“代理 A 给代理 B 发一大段总结”当成主要协议。自然语言可以作为对象中的一个字段，但不能替代 ID、版本、状态和引用关系。

## 9. 一个完整例子

用户问题：

> 研究 2025 年企业级 AI Agent 市场，比较主要技术路线、厂商、市场规模和风险。

### 9.1 Main call #1

创建 DAG v1：

~~~text
st1 定义与市场口径
st2 官方/一手市场数据
st3 技术路线与代表产品
st4 风险、失败案例与监管
st5 merge，依赖 st1-st4
st6 coverage gate，依赖 st5
st7 verify gate，依赖 st6
st8 write → st9 audit → st10 release
~~~

st1-st4 同时 READY，Scheduler 根据并发上限启动四个 Researcher。

### 9.2 Researcher st2

~~~text
step 1：搜索市场规模的一手来源
step 2：抓取三份报告；其中一份只有二手新闻引用
step 3：读取口径、年份、地域和测算方式
step 4：发现两个数字定义不同
step 5：定向搜索原始报告与方法说明
step 6：提交两个 qualified claims、五条 evidence 和一个 conflict
~~~

它的完整 LocalResearchState 不发给 Main，只提交 AgentResult 和相关 ID。

### 9.3 Main call #2

Global Reducer 发现：

- 亚洲数据 coverage 为空。
- 两个市场规模数字的统计口径冲突。
- 技术和风险部分已达标。

Main 生成 DAG v2：

~~~text
ADD st11：亚洲市场定向研究
ADD st12：市场规模口径 verifier
ADD st11/st12 → verify gate 的依赖
CANCEL 一个重复的厂商搜索任务
~~~

st11、st12 并行执行，原 DAG 其他结果保留。

### 9.4 Write、Audit、Repair

证据门禁通过后 Writer 生成报告。Auditor 发现一处“市场增长最快”的陈述没有直接证据，输出 repair item。Main 可以：

- 增加一个定向研究任务；或
- 删除“最快”，改为证据实际支持的表述。

二次 audit 通过后发布。

## 10. 必须保持的系统不变量

1. Main、Researcher 和 Auditor 都不能直接写权威状态。
2. 每个状态 patch 必须携带 base version，并由 Reducer/Store 原子提交。
3. 每个 Claim 必须关联 Evidence，或显式标为 inference/unsupported。
4. 原始来源内容是不可信输入，不能改变系统指令或提升工具权限。
5. LocalResearchState 以 `(run_id, subtask_id)` 隔离。
6. Main 只看必要的全局摘要；Researcher 只看必要的局部切片。
7. Scheduler 的并发、预算和重试不能由 LLM 绕过。
8. Audit 失败必须先分类：缺证据项回到 repair DAG；引用格式、删除不支持表述、按已有 Evidence 收窄范围等文本项进入带审计上下文的定向 Writer revision，不能无约束地“凭感觉润色”。
9. 任何时候都可以从 Event Log 和 checkpoint 恢复，而不重复已完成的外部操作。
10. 多代理效果必须与等 token、等工具、等时间的单代理基线比较。

## 11. 当前实现映射

上述主流程已经在 `drb_qwen/multi_agent/` 中实现：

| 设计组件 | 实现模块 |
|---|---|
| Global/Local State、Claim、Evidence、Calculation、SubTask | `schemas.py` |
| JSON snapshot、Event Log、Artifact、SubTask bundle | `store.py` |
| DAG Patch/REFINE_TASK、分片、自适应预算、无环校验、Ready Set | `dag.py` |
| 唯一全局合并与 Research Gate | `reducer.py` |
| Main、Researcher、Writer、Citation Auditor | `agents.py` |
| Search、Fetch、Reader 和来源标准化 | `tools.py` |
| Evidence-linked 白名单计算器 | `calculator.py` |
| Prompt 协议 | `prompts.py` |
| JSON Schema 协议 | `protocols.py` |
| token-aware ResearcherInput View | `context.py` |
| 角色调度、token admission、structured output | `inference.py` |
| vLLM HTTP 调用与逐次 token usage | `llm.py` + `async_llm_client.py` |
| URL 安全和来源独立性分组 | `security.py` |
| 外层事件驱动元工作流 | `workflow.py` |

原 [`deep_research_workflow.py`](../drb_qwen/deep_research_workflow.py) 现在只是兼容导入，现有批量生成、RACE 和 FACT 入口无需改变。系统还提供：

- `python -m drb_qwen.run_multi_agent_research`：单个任意问题。
- `python -m drb_qwen.generate_reports_async_research`：DRB JSONL 批量研究。
- `python scripts/compare_race_runs.py`：只在两次 Judge 的共同有效 task ID 上比较分数。
- `scripts/check_pipeline_static.sh`：无 GPU 的编译、DAG、恢复、动态重规划和 Audit Repair 测试。

当前持久化实现是单机 JSON durable store，适合实验和单节点运行；多节点生产部署时可以保持相同 schema，把 `RunStore` 替换为 PostgreSQL/Event Store，把外层循环迁移到 Temporal，而无需重写 Agent 协议。

## 12. 常见误解的直接回答

### Main 是连续 ReAct 吗？

不是。Main 是跨 run 的逻辑角色，但底层是多次独立 input/output；外部 GlobalResearchState 负责跨调用连续性。

### 每个 Researcher 是 ReAct 吗？

是。它在一个 SubTask 内做有轮数、工具、token 和时间上限的局部 ReAct。

### Main 每轮重新规划完整 DAG 吗？

不是。第一次规划粗粒度完整骨架，之后只提交增量 DAG Patch。

### Main 只规划本轮要跑什么吗？

也不是。Main 定义任务和依赖；Scheduler 根据当前依赖、预算和资源计算 Ready Set 并决定实际并发。

### LocalResearchState 从哪里来？

Scheduler 用合同和全局切片初始化；Tool Runtime 写客观事件；Researcher 提交语义 patch；Local Reducer 校验并合并。

### Main 能看到每个 LocalResearchState 吗？

通常不能，也不需要。Main 只接收 AgentResult、Claim/Evidence 引用、gap、conflict 和成本摘要；必要时才按 ID 下钻。

### 动态 workflow 是 LLM 生成代码吗？

不是。它是固定的元工作流，加一个运行时可修改的数据 DAG。LLM 只能输出白名单内的 DecisionPatch。

## 13. 设计依据

本流程主要吸收了以下公开资料：

- Anthropic 的多代理 Research 工程实践：Lead Agent、并行 subagents、任务边界和 Citation Agent。
  https://www.anthropic.com/engineering/multi-agent-research-system
- Google Research 对多代理扩展规律的研究：并行性、中心化协调、错误放大和任务结构。
  https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/
- ReAct：在单个任务内交替 reasoning、action 和 observation。
  https://arxiv.org/abs/2210.03629
- Magentic-One：Task Ledger、Progress Ledger 和 orchestrator-worker 设计。
  https://arxiv.org/abs/2411.04468
- STORM：多视角发现、来源研究、提纲和长文生成。
  https://arxiv.org/abs/2402.14207
- SAFE：将长回答拆为原子事实并逐条搜索验证。
  https://deepmind.google/research/publications/85420/
- DeepResearch Bench：分别评估报告、检索和引用质量。
  https://arxiv.org/abs/2506.11763
- OWASP AI Agent / Prompt Injection 指南：最小权限、外部内容不可信、结构化输出和预算限制。
  https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html
- Temporal durable execution：长运行工作流、重试和故障恢复。
  https://docs.temporal.io/
- vLLM Automatic Prefix Caching：复用相同 token 前缀的 KV block，但不把缓存作为持久状态。
  https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
- vLLM Structured Outputs：通过 JSON Schema、regex 或 grammar 约束在线推理输出。
  https://docs.vllm.ai/en/stable/features/structured_outputs/
- vLLM Engine Arguments：priority/FCFS 调度和 chunked prefill 等服务参数。
  https://docs.vllm.ai/en/stable/configuration/engine_args/

上述资料支持架构原则；本文具体的 schema、阈值、DAG 操作和组件边界属于工程设计，需要通过本项目的真实任务集和等预算 A/B 测试校准。
