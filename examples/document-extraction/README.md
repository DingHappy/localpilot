# 将验收配置接入自己的调用

这是 LocalPilot 配置交付示例：Python 直接调用既有 vLLM，使用已测的提示词、
JSON Schema、temperature=0 和 768 token 输出上限。只需要 Python 3 标准库，
无需启动或安装 LocalPilot API。模型与引擎需要事先可用。

在推理节点上，从项目根目录运行：

```bash
python3 examples/document-extraction/extract.py \
  evals/documents/holdout-v1/holdout-01.png \
  --model step3-vl-10b-fp8
```

将图片路径替换成自己的 PNG/JPEG 即可，模型名须与实际服务一致。
示例仅适用于当前五字段任务，字段是 invoice_number、issue_date、total_usd、
seller 和 purchase_order；只有缺失的 purchase_order 允许 null。
图片必须能够支持这一任务，不能把任意文档都视作已验证输入。

如需从控制端调用远端服务，先建立 SSH 本地隧道（将 `YOUR_NODE_ALIAS`
换成自己的已配置 SSH 别名）：

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 YOUR_NODE_ALIAS
```

保持隧道窗口运行，在另一个终端执行：

```bash
python3 examples/document-extraction/extract.py /path/to/image.png \
  --model step3-vl-10b-fp8 --base-url http://127.0.0.1:18000/v1
```

输出是 JSON。网络失败、输出截断、字段或类型不符时返回非零退出码，不重试。
结构检查通过不代表内容正确；没有人工标签的用户图片无法计算字段准确率。
脚本不写响应日志；重定向标准输出时由调用方决定保存位置。
此最小示例面向回环服务或 SSH 隧道，不会启动服务或暴露公网端口。

更改模型、提示词、字段、输出预算或任务数据后，应重新验收。
此示例使用非流式响应，配对性能报告使用流式计时，不能把历史耗时当作本脚本的 SLA。
