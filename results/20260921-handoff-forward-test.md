交付建议：使用现有 **DGX Spark / NVIDIA GB10 上的 Step3-VL-10B-FP8 + vLLM 服务，应用直接发送带 JSON Schema 的图片请求**。LocalPilot 不需要持续运行；模型引擎和必要的 SSH 隧道需要保持运行。以下来自 2026-09-21 的实测记录，本次未运行推理、连接远端或修改文件，也未确认服务此刻仍在线。

**配置与接入**

| 项目 | 交付值 |
|---|---|
| 执行目标 | DGX Spark，单 NVIDIA GB10，119.7 GB 统一内存 |
| 引擎 | vLLM `0.15.1+befbc472` |
| 模型来源 | `stepfun-ai/Step3-VL-10B-FP8` |
| 请求中的模型名 | `step3-vl-10b-fp8` |
| 节点内接口 | `http://127.0.0.1:8000/v1/chat/completions` |
| 服务实测设置 | 上下文 8192、KV cache `auto`、`gpu_memory_utilization=0.5` |
| 请求设置 | 并发 1、`temperature=0`、`max_tokens=768`、严格 JSON Schema |
| 输入与输出 | PNG/JPEG 图片；提取五个指定字段 |

模型权重为 FP8，**不能据此将 KV cache 写成 FP8**。KV FP8 候选因与现有服务不符被排除，没有测得其效果。其他模型也未完成对比，因此这不是跨模型或跨引擎的最优结论。

现有记录是附着既有服务完成的，`managed_by_localpilot=false`、`launch_command=null`。交付方式是接入该服务；资料不足以提供已经验证的重建启动命令、权重路径或服务管理命令。

直接交付整个 [examples/document-extraction 目录](../examples/document-extraction/README.md)，保留其中的 `extract.py`、`prompt.txt` 和 `response-format.json`。开发者在推理节点的项目根目录执行：

```bash
python3 examples/document-extraction/extract.py /path/to/invoice.png \
  --model step3-vl-10b-fp8
```

从另一台开发机访问时，节点的 `127.0.0.1` 不是开发机地址。使用已有授权的 SSH 节点别名建立隧道：

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 YOUR_NODE_ALIAS
```

保持隧道运行，在另一个终端执行：

```bash
python3 examples/document-extraction/extract.py /path/to/invoice.png \
  --model step3-vl-10b-fp8 \
  --base-url http://127.0.0.1:18000/v1
```

这些是交给开发者的操作说明，本次没有执行。脚本仅依赖 Python 3 标准库，直接调用 vLLM，无需 LocalPilot API。它会将图片编码为 `image_url` 内容块，并实际携带提示词、Schema 和生成参数。

五个字段为 `invoice_number`、`issue_date`、`total_usd`、`seller`、`purchase_order`；值按可见内容保留为字符串，包括前导零和金额两位小数，仅缺失的 `purchase_order` 使用 `null`。取发票总额，避免误取参考报价。不要向模型发送评测用的 `expected_fields` 或 `mock_answer`。

脚本会拒绝请求失败、截断、重复字段以及字段或类型不匹配的结果；结构检查通过仍不代表内容正确。

**速度和准确性可以怎样表述**

三份记录的硬件与 benchmark 均标为 `simulated=false`，属于该 NVIDIA 节点的实测，不能表述为 Apple Silicon 本地验证。

| 证据 | 样本与条件 | 结果 |
|---|---|---|
| 输出策略配对 | 6 张不同图片，每个策略各重复 2 次；并发 1、热服务、AB/BA 交替、未清缓存；只改变 `response_format` | 平均完整响应由 **9.764 秒降至 3.138 秒，减少 67.9%**；两组均 12/12 次文档全字段正确、60/60 字段正确，无失败或截断 |
| 冻结 Schema 的留出验收 | 20 张不同新输入，各 1 次；含中英文、两种布局、清晰及轻微模糊、采购单号有无等标签 | **20/20 文档、100/100 字段正确**；平均 **4.654 秒**，观测范围 **4.299–4.849 秒**；无失败或截断 |
| READY Profile 记录 | 6 张图片用于字段验收；计时为 3 次请求、1 个不同提示输入，另有 1 次预热 | 30/30 字段正确；平均完整响应 **3.069 秒**，首 token **97.13 ms**；单流 **20.08 token/s**，聚合 **19.77 token/s** |

配对中平均输出从 194.3 token 减至 61.8 token，首 token 延迟基本相同；收益体现为完整输出更快，不能说模型解码速度提高了 67.9%。每组 12 次调用也不能算作 12 张新图片。

可以向接收者承诺的是“上述配置在这些有标签样本上得到以上结果”。**不能承诺任意真实发票 100% 正确、每张固定 3 秒、P95 时延、冷启动表现或多用户吞吐。** 留出验收没有基线组，不能从中计算优化增益；示例脚本为非流式，而配对报告使用流式计时，历史耗时不是脚本 SLA。质量采用确定性字段比对，不是独立语义评分。

如需容量参考，引擎记录的权重加已分配 KV 内存为 **55.04 GB**；候选中的 **16.66 GB 是规划估算**，不应作为实际服务占用承诺。

**验收凭据与后续边界**

保留以下原始文件作为交付附件：

- [输出策略配对](20260921-output-strategy-pair.json)：选择 `json_schema`；门槛为所有文档精确匹配、无失败或截断，并要求至少 10% 收益。数据 SHA-256：`470ed604b75f5b9e5643b6a2c83a1b057c8e786f194d4fdee5c7d4d5893b1157`。
- [留出验收](20260921-schema-holdout.json)：`ACCEPTED`，同样要求所有文档精确匹配、无失败或截断。数据 SHA-256：`572c63381a11c546643749bfdd7066adf64ee0311e182d4ebf18749115751621`。
- [Profile 验收](20260921T134910-real-vision-quality.json)：run ID `d4e95a9e-20cc-4d6f-8ee7-9ce61b7f2e3a`，Profile `52b256d0a4915bbff292`；门槛 `min_quality=1.0`，目标 `fastest_complete`；benchmark SHA-256：`c3d03eabe15fcb8f844af80bdac9d9aba012e58bbb9b2c3b4bff713df61c9092`。

配对及留出报告引用的 baseline run ID 均为 `b5073210-b185-4d45-a6ce-6a016ae02c8b`，各自还保存完整配置哈希。更改模型、引擎、提示词、字段、输出预算、并发、任务数据或验收政策后，应重新验收。

`READY` Profile 只是记录通过验收的决策，不会替应用自动添加 Schema，也不会保持引擎在线。日常业务直接调用 vLLM 即可；再次选型、调整配置或重新验收时再使用 LocalPilot。
