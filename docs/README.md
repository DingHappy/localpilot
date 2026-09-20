# Figures

Three hand-authored SVGs. They are committed as sources rather than exported
images so a number in a figure can be diffed against the code that produced
it, and so they scale cleanly into slides.

| File | What it argues |
|---|---|
| `architecture-loop.svg` | The loop, and the two structural facts that are easy to miss: measuring is an inner loop per candidate, and a profile hit skips the whole search. |
| `architecture-platform.svg` | Why this platform inverts the usual advice — capacity is charged on total parameters, decode speed only on the active ones. |
| `architecture-skill.svg` | Where this skill sits: the decision and memory layer above NVIDIA's per-task skills, and what it deliberately does not do. |

## Where the numbers come from

`architecture-platform.svg` carries the only quantitative claims. Every figure
in it is reproducible from the tool itself:

```bash
localpilot registry --task coding
```

- **Memory** (9.1 / 21.7 / 65.1 / 79.5 / 348.0 GB) is the planner's estimate:
  checkpoint sizes summed from each repo's Hugging Face file tree, plus KV
  cache computed from `config.json` — counting only the attention blocks in
  Nemotron-H's hybrid layer pattern — plus activation overhead.
- **Speed** (26 / 121 / 34 / 30 / 7 tok/s) is a bandwidth ceiling:
  `273 GB/s × 75% ÷ (active parameters × bytes per parameter)`.

The speed column is **a modelled upper bound, not a benchmark result**, and the
figure says so in its source note. When real measurements land on a DGX Spark
they go in `results/` via `localpilot report`, and any figure that then quotes
them should be relabelled.

## Palette

`architecture-platform.svg` is the only figure with quantitative colour
encoding. It uses two categorical slots in fixed order — `#2a78d6` for NVFP4
and `#eb6834` for BF16 — so colour encodes weight precision, the attribute the
figure is about, rather than rank or state.

Validated all-pairs against the `#f6f8f5` panel surface: worst CVD ΔE 24.7,
worst normal-vision ΔE 33.6. Orange sits below 3:1 contrast on that surface,
which obligates visible labels; every bar is directly labelled and the same
numbers appear as a table in the top-level README.

"Does not fit" is carried by geometry and text — the bar crosses the budget
threshold, fades, tears at the pool edge, and says so — not by a third colour.
An earlier draft used the status red for it; that failed the normal-vision
floor against the orange (ΔE 10.8, against a floor of 15) and, more to the
point, would have used colour to encode a state while the rest of the figure
uses it to encode an attribute.

## Checking a change

There is no build step. After editing, confirm the file still parses and look
at it — the palette can be validated mechanically, but label collisions and
overflow cannot:

```bash
python3 -c "import xml.etree.ElementTree as ET; ET.parse('docs/architecture-loop.svg')"
qlmanage -t -s 1600 -o /tmp docs/architecture-loop.svg   # macOS preview
```

`qlmanage` renders into a square thumbnail, so a wide figure comes back
cropped. To inspect one region, copy the file with a narrowed `viewBox` and
render that instead.
