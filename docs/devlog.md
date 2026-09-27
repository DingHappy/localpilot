# 开发日志 · 十日谈原始素材

征文的成稿骨架在 [`essay.md`](essay.md)。这里是**原始素材**，按天追加。

## 怎么用

每天收工时跑一次，把输出粘到下面对应的一天：

```bash
python3 scripts/devlog.py            # 今天
python3 scripts/devlog.py 2026-09-23 # 补某一天
python3 scripts/devlog.py --all      # 补整个窗口
```

脚本会自动收集当天的提交、运行记录、导出的报告、测试数、以及失败与恢复事件。
**它答不了的只有三个问题**，那三行必须手写：

- 今天想清楚了什么
- 今天错在哪（预测和实测差多少？为什么？）
- 明天第一件事

第二行最值钱。别等到第 9 天回头补 —— 那时候你只会记得结论，记不得当时为什么
判断错了，而那恰恰是征文里唯一别人写不出来的部分。

---

## Day 1 — 2026-09-20

- 提交：137 files, +13157 / -1332
- 运行：52 次（全部为模拟）
- 导出报告：`20260920T074507-sim-coding-latency.md`
- 测试：Ran 142 tests（移植前 14）

<details><summary>当天提交</summary>

- `bfad00c` Add the three architecture figures
- `f8bc57a` Fix measurement validity, stream the trace, commit the evidence
- `2c0eb3a` Port the autopilot from Intel/OpenVINO to NVIDIA DGX Spark
- `04b56c5` Baseline: Intel/OpenVINO development slice

</details>

| 时间 | 任务/优先级 | 冠军配置 | 引擎 | TTFT | tok/s | 显存 | 分数 | 次数 | 实测? |
|---|---|---|---|---|---|---|---|---|---|
| 07:36:15 | chat/balanced | `nemotron-3-nano-30b-a3b-nvfp4-vllm` | vllm | 75 ms | 113.9 | 20.9 GB | 88.9 | 1 | 模拟 |
| 07:36:16 | coding/balanced | `nemotron-3-nano-30b-a3b-nvfp4-vllm` | vllm | 75 ms | 113.9 | 20.9 GB | 88.9 | 1 | 模拟 |
| 11:59:57 | coding/latency | `nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-nvfp4-spec` | trtllm | 63 ms | 215.4 | 23.7 GB | 96.7 | 50 | 模拟 |

> 日志里还留着移植前的候选 ID（`qwen2.5-coder-*-int4-ov-gpu-int4_asym`），
> 那是同一天上午的事 —— 平台切换的痕迹自己记录了下来。

**今天想清楚了什么**

这台机器上「选哪个设备」不是问题，一个加速器一个内存池。真正的问题是服务配置，
而它难是因为**容量按总参数收费、速度只按激活参数收费**，这两个量在 128 GB
统一内存 + 窄带宽的机器上第一次解耦。整个搜索空间因此从「设备」换成了
「引擎 × 精度 × KV × 上下文 × 批宽 × 投机解码」。

**今天错在哪**

1. 把 `...-NVFP4-DSpark` 读成「为 DGX Spark 调优的目标模型」。实际 DSpark 是
   投机解码方法名，那是个 967M 的草稿头。纠正后投机解码反而成了搜索空间的一档。
2. 三个测量缺陷（并发同 prompt 撞 prefix cache、peak_memory 在生成后才读、
   max_new_tokens 只有 64）。全部在真机数据之前查出并修掉 —— 否则会带着三个
   错的数字上台。
3. 实测证据本来全在 .gitignore 里，评委翻 repo 看不到任何数据。加了 `results/`
   和 `localpilot report`。

**明天第一件事**

上云节点，跑 `localpilot doctor`，确认 platform 认成 `dgx_spark`、带宽和引擎
探测是否正确。**截图存档** —— 那是这个项目在真机上活着的第一个证据。

---

## Day 2 — 2026-09-21

- 提交：231 files, +20205 / -784
- 运行：72 次，其中 **1 次真实硬件**（只统计本机日志；DGX 节点上的实测运行状态留在节点，见下面 `-real-` 报告）
- 导出报告：20260921-document-field-validation.md, 20260921-handoff-forward-review.md, 20260921-handoff-forward-test.md, 20260921-output-strategy-comparison.md, 20260921-real-chat-load-comparison.md, 20260921-real-vision-validation.md, 20260921-schema-holdout.md, 20260921T040528-real-chat-quality.md, 20260921T122320-real-chat-latency.md, 20260921T122943-real-chat-throughput.md, 20260921T124614-real-vision-quality.md, 20260921T130219-real-vision-quality.md, 20260921T134910-real-vision-quality.md
- 测试：Ran 255 tests in 105.793s

<details><summary>当天提交</summary>

- `58705bc` evidence: evaluate handoff skill and prepare competition demo
- `485ae3d` feat: validate frozen strategy on new documents and hand off direct caller
- `d1650bf` refactor: focus skill on inference configuration handoff
- `144c62e` evidence: validate faster schema-constrained invoice responses
- `d21d3fd` feat: measure paired output strategies on identical requests
- `95b97c6` fix: preserve rejected optimization runs for review
- `a65fffd` feat: select configurations within explicit acceptance budgets
- `e07c3f3` evidence: record DGX Spark synthetic invoice acceptance
- `0d2b81b` feat: validate document fields against versioned synthetic fixtures
- `d84517b` evidence: validate real DGX Spark vision input
- `ccffd70` feat: benchmark real vision inputs
- `b847694` evidence: compare DGX Spark chat load
- `fbb18d2` evidence: record DGX Spark chat baseline
- `e70d49d` docs: publish skill behavior benchmark
- `4211246` test: distinguish scope checks from skill activation
- `c99762e` fix: narrow skill discovery to LocalPilot tasks
- `3543449` fix: preserve timed-out eval output
- `19a9a5c` fix: exclude container authoring from skill activation
- `e9e00c5` fix: model guarded profile activation in eval fixture
- `f0835fe` fix: enforce modality evidence in skill evaluations
- `8469adc` test: add isolated skill behavior evaluation
- `e06ab44` feat: make LocalPilot adaptive and evidence-driven

</details>

| 时间 | 任务/优先级 | 冠军配置 | 引擎 | TTFT | tok/s | 显存 | 分数 | 次数 | 实测? |
|---|---|---|---|---|---|---|---|---|---|
| 04:05:28 | chat/quality | `qwen3.8-27b-mlx-ollama-nvfp4` | ollama | 598 ms | 14.3 | 20.0 GB | 75.0 | 1 | **实测** |
| 06:15:17 | coding/latency | `nemotron-3.5-lightning-30b-a3b-nvf` | trtllm | 63 ms | 215.4 | 23.7 GB | 99.5 | 71 | 模拟 · 命中 |

> 峰值显存来源：planner_estimate_no_readable_source
> ⚠ 1 次实测没有 rubric 评分，质量这一维是弱信号（judge 端点未配置？）

<details><summary>失败与恢复</summary>

- 日志里有 687 条 `candidate_failed`，**全部来自单元测试**，不是实机事件：
  每跑一次测试就写入 19–22 条（`candidate_id: 'a'` 等测试夹具），当天共 42 批。
  原因是测试把运行事件写进了项目的 `logs/localpilot.jsonl`，待修。
- 真实的失败与恢复：
  - Skill 行为评测：12 个负例里 2 个误触发（安装 Ollama、纯 CPU 实机验收），严格发布门槛未过。
  - 20 张新样本验收：独立测试目录第一次启动缺少配置资源，在发出推理请求前停止；补齐后正式执行。
  - 实测 chat 冠军没有 rubric 评分（judge 端点未配置），质量一维是弱信号。

</details>

**今天想清楚了什么**（草稿，待改）：交付物不是一个一直运行的服务，而是验收过的配置、应用可以直连的调用示例和证据。固定模型、只改输出格式，完整响应就从 9.764 秒降到 3.138 秒——这个任务的等待主要花在生成了多少内容，而不是 GPU 每个 token 的解码速度。

**今天错在哪**（草稿，待改）：Skill 的边界判断还不够准，"安装 Ollama"和"纯 CPU 验收"都误触发了 LocalPilot 的工作流。README 的主要论点（稀疏比稠密快、NVFP4 比 BF16 快、投机解码 1.8×）到今天为止仍然只有模拟数字，没有一组实测对照。

**明天第一件事**（草稿，待改）：在 DGX 上用 Nemotron-30B-A3B-NVFP4 自带的 MTP 层跑投机解码开/关，填征文第四节的第一行实测数字。

---

## Day 3 — 2026-09-22

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

---

## Day 4 — 2026-09-23

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

---

## Day 5 — 2026-09-24

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

---

## Day 6 — 2026-09-25

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

---

## Day 7 — 2026-09-26

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

---

## Day 8 — 2026-09-27

本日核对以 Git 工作区、255 项单元测试、Skill 安装器、Agent 行为评测和分配的
Spark 节点为准。完整行为评测证据见 [`BENCHMARK.md`](../BENCHMARK.md)。

**今天想清楚了什么**：Skill 的正向路径已经可用，发布障碍主要是触发边界。
完整 23 案例的低并发重测中，支持任务完成 11/11，无 Skill 基线完成 3/11；
负例排除 11/12，危险操作 0 次，严格门槛仍未过。一次重测不能覆盖所有上下文
和模型波动。

**今天错在哪**：我原以为只补两条入口规则就能让负例全部通过。完整行为评测
只剩通用 CUDA 故障问句误触发：Agent 说它不属于 Skill，却依然执行了 LocalPilot
只读诊断命令。基线也这么做；夹具仅提供 LocalPilot 可执行文件，没有故障应用
或 CUDA 日志。先前四并发运行中的超时在这次两并发、300 秒限制下没有重现。

Spark 上已有约 21 GB 的 Nemotron 30B NVFP4 权重，但缓存的 vLLM 0.15.1
与模型卡推荐的 vLLM 0.27.1 不一致。隔离容器的短时兼容性烟测在模型配置
阶段失败：Transformers 不认识 `nemotron_h`，没有生成任何性能数字。原有
`lp-vllm` 和 `node-exporter` 服务未受影响。证据见
[`20260927-nemotron-compat-smoke`](../results/20260927-nemotron-compat-smoke/README.md)。

**明天第一件事**：保留 Skill 误触发这一发布限制。若需要模型精度或投机解码
对照，先解决镜像兼容，再做隔离、受控的实机实验；不得用模拟倍数填征文。

---

## Day 9 — 2026-09-28

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

---

## Day 10 — 2026-09-29 · 提交日

`[粘贴 devlog.py 输出]`

提交前核对：

- [ ] `results/` 里有至少一份 `-real-` 报告，且已 commit
- [ ] 如声称 NVFP4/BF16、稀疏/稠密或投机解码收益，逐项提供受控实测；未测则明确留白
- [x] 征文第四节改用已有真机请求策略对照，并写明原三组模型对照未验证
- [ ] `quality_judge` 不是 null（judge 端点配上了），否则在文中说明
- [ ] Demo 视频录了「第二次请求 PROFILE HIT 瞬间命中」那一幕
- [ ] README 里所有模拟数字都标着「模型推算」
- [ ] 征文里每个倍数都说清了对比两边的差异变量

**十天里最想留下的一句话**
