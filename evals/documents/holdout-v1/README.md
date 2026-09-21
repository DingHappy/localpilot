# Frozen synthetic document holdout v1

20 new synthetic images, created before any inference on this dataset. The original
six-image dataset is unchanged. Gold values are labels used only by the evaluator;
they are never included in the model request.

- 10 Chinese and 10 English documents.
- 10 row layouts and 10 two-column layouts.
- 8 images have Gaussian blur radius 0.8; 12 remain sharp.
- 7 omit the purchase order; the expected value is null.
- All include a clearly labeled reference quote distractor and a synthetic-use notice.

The prompt, response schema, temperature 0, and 768-token output budget are frozen
from the earlier selected strategy. Each image is measured once with no warmup or
retry on an already warm service. Acceptance requires all 20 complete documents
to match all five fields exactly. Rejection must not trigger silent label or
threshold changes. These are clean synthetic layouts, not a representative
production corpus; tag groups overlap and do not support causal attribution.

Rebuild on macOS with Pillow 11.1.0 and the system CJK font:

```bash
python scripts/build_document_holdout.py --font '/System/Library/Fonts/STHeiti Medium.ttc'
```

Use the committed PNGs and JSON for exact reproduction on other operating systems;
font-version differences may change regenerated image bytes.

Run on the same inference node as the real baseline profile:

```bash
python3 -m localpilot.benchmark.paired --holdout \
  --profile-report results/20260921T130219-real-vision-quality.json \
  --dataset evals/documents/holdout-v1/benchmark.json \
  --response-format evals/documents/response-format.json \
  --output results/NEW-HOLDOUT.json
```

The runner checks the real hardware identity and the serving configuration before
and after execution. A holdout run validates the fixed strategy; it never selects
a new strategy or reports an optimization gain.
