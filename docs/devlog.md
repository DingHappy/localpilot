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

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

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

`[粘贴 devlog.py 输出]`

**今天想清楚了什么**

**今天错在哪**

**明天第一件事**

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
- [ ] 三组单变量对照（NVFP4/BF16、稀疏/稠密、投机解码开关）都有实测数字
- [ ] 预测与实测的偏差写进了征文第四节，且给出了原因
- [ ] `quality_judge` 不是 null（judge 端点配上了），否则在文中说明
- [ ] Demo 视频录了「第二次请求 PROFILE HIT 瞬间命中」那一幕
- [ ] README 里所有模拟数字都标着「模型推算」
- [ ] 征文里每个倍数都说清了对比两边的差异变量

**十天里最想留下的一句话**
