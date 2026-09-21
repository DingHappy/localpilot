# 输出策略的单因素对照

本实验只改变 OpenAI 请求的 `response_format`：默认文本与显式 JSON Schema。
图片、提取指令、模型、引擎配置、并发、temperature 和最大输出预算保持一致。
Schema 只声明字段名及类型，不包含答案、枚举或某张票据的真实值。

在已有服务所在的执行节点运行：

```bash
python3 -m localpilot.benchmark.paired \
  --profile-report results/20260921T130219-real-vision-quality.json \
  --dataset evals/documents/benchmark.json \
  --response-format evals/documents/response-format.json \
  --output results/output-strategy-pair.json
```

工具只连接节点回环 HTTP 服务，要求真实基线 Profile、匹配的设备指纹，以及
能核对的现有 vLLM 配置。不会下载模型或启动引擎。长实验应使用 tmux。

默认两个策略各预热一次，然后每张图每个策略测两次，交替 AB/BA 顺序；6 张图
共 24 次计分请求。已有服务的缓存不清空，结果描述暖服务上的交替请求表现，
不代表冷启动或不同图片分布下的性能。

同一次流式响应记录首字与完整耗时、引擎输出 token 数、结束原因，同时解析
最终答案并检查字段。没有正常结束或输出被截断的请求不能通过整张文档验收。
保存字段判定而不保存模型回答；每次请求后写进度，失败证据不会丢失。

选择规则预先固定：所有测试文档必须正确且完整结束，新策略平均完整耗时至少
降低 10% 才推荐新策略；否则保留通过验收的基线。若基线也未通过，则拒绝选择。
检查前后可观察服务配置一致；不一致时使对照失效。

本实验是小规模请求策略对照，不是模型权重、内核或 GPU 参数的加速实验。
生成 token 变少可以降低等待时间，但不能据此声称每 token 解码速度提高。
只有合成英文图的结果，不能外推为中文发票、复杂扫描件或生产准确率。
