# Synthetic invoice field acceptance

Six invented English invoice images, five fields per image: invoice number,
issue date, total USD, seller, and purchase order. Three images omit the purchase
order, which must be returned as JSON null. Alternating row order, similar
numbers, leading zeros, decimal places, and a distractor quote test key/value
association. These are test documents, not valid invoices or personal data.

Run from a source checkout with an already available OpenAI-compatible vLLM
service (no model download or service launch):

```bash
LOCALPILOT_BENCHMARK_CONFIG=evals/documents/benchmark.json \
LOCALPILOT_VLLM_BASE_URL=http://127.0.0.1:8000 \
python3 -m localpilot.cli autopilot \
  '本地发票图片字段提取，单用户质量优先' --mode vllm --no-reuse --json
python3 -m localpilot.cli report <run-id> --json
```

The benchmark config is opt-in; default vision requests retain the fast color
smoke test. Expected fields and mock answers are never included in model
requests. Each document gets one separate quality request; three streaming
requests measure performance, so latency covers a subset of the six documents.
Performance and quality are separate requests, not repeatability measurements.

Scoring parses the final JSON object, accepts a JSON code fence or a closed
reasoning prefix, rejects prose, malformed/truncated JSON, and duplicate keys.
Values and types must match exactly. Missing keys fail even if their expected
value is null; extra keys fail document acceptance. Field accuracy counts
correct expected fields; document accuracy requires every field and exact keys.
Failures stay in both denominators. The ranking quality score is document
accuracy, not field accuracy or a semantic judge score. Reports retain per-field
booleans and dataset hashes, not extracted text or model answers.

Changing benchmark data or measurement settings invalidates profile reuse.
For reproducibility, keep this config with its images and the recorded hash.
Regenerate using `python scripts/build_document_fixture.py` with Pillow 11.1.0;
Pillow is only a fixture build dependency, not a LocalPilot runtime dependency.

A pass establishes accuracy only on these six clean English synthetic images.
It does not establish Chinese tax-invoice accuracy, scan/rotation robustness,
production OCR reliability, privacy enforcement, or configuration speedup.
