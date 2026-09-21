"""Replay committed evidence locally; never connects to or changes a service."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / 'results' / name).read_text())


def main():
    pair = read('20260921-output-strategy-pair.json')
    holdout = read('20260921-schema-holdout.json')
    rejection = read('20260921-demo-rejection-simulated.json')
    direct = read('20260921-direct-handoff-smoke.json')
    print('LocalPilot — 已保存证据回放（不是现场重新实测）')
    for label, record in [('配对实验', pair), ('新样本验收', holdout)]:
        if record['hardware']['simulated'] or record['benchmark']['simulated']:
            raise ValueError(label + '不是实机记录')
        if not record['configuration_unchanged']:
            raise ValueError(label + '服务配置发生变化')
    a, b = pair['summaries']['baseline'], pair['summaries']['json_schema']
    print(f"\n1. 同一模型、同一输入，只改变 response_format（6 张图，各重复 2 次）")
    print(f"   完整响应均值：{a['mean_total_ms']/1000:.3f}s → {b['mean_total_ms']/1000:.3f}s")
    print(f"   延迟减少：{pair['relative_latency_reduction']:.2%}；选定策略：{pair['selected_strategy']}")
    print(f"   字段正确：{a['fields_correct']}/{a['fields_total']} → {b['fields_correct']}/{b['fields_total']}")
    h = holdout['summaries']['json_schema']
    print(f"\n2. 固定策略，20 张全新合成图：{holdout['decision']}")
    print(f"   字段 {h['fields_correct']}/{h['fields_total']}；文档正确率 {h['document_accuracy']:.0%}")
    print(f"   完整响应均值 {h['mean_total_ms']/1000:.3f}s；不据此计算新的加速比例")
    print(f"\n3. 直连 vLLM 交付示例：{sum(s['document_exact'] and s['exit_code']==0 for s in direct['samples'])}/2 次调用通过")
    print('   使用 examples/document-extraction/extract.py；无需 LocalPilot 推理 API')
    print(f"\n4. 模拟失败路径：{rejection['status']}；best_profile={rejection['best_profile']}")
    print(f"   原配置记录保持不变：{rejection['demo_evidence']['current_profile_unchanged']}")
    print('   1ms 完整响应门槛是刻意构造的不可满足条件，不是实机性能结果。')
    print('\n结论边界：小规模合成数据验收，不代表任意真实文档或长时间运行质量。')


if __name__ == '__main__':
    main()
