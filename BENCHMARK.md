# LocalPilot Skill behavior benchmark

This benchmark tests whether an Agent completes LocalPilot tasks more reliably
when the `local-ai-autopilot` Skill is discoverable. It measures Agent behavior,
not model-serving speed and not DGX Spark hardware performance.

The results below apply to Skill commit `42112469baacbb0765a3967197cff293e761f6aa`.
The later configuration-handoff scope revision and new optimization references
have structural/installation checks, but have not rerun this Agent behavior
benchmark. Do not treat the historical rates as verified rates for the revision.

## Result

**Run status: failed one strict release gate.** The Skill met five of six gates.
It completed every supported task, improved completion by 54.5 percentage points
over the no-Skill baseline, preserved every critical evidence invariant, and
took no observed unsafe action. Negative activation accuracy was 91.7%, below
the required 100%.

| Signal | Gate | Result | Pass |
|---|---:|---:|:---:|
| Positive activation | >= 90% | 100.0% | yes |
| Negative activation accuracy | 100% | 91.7% | no |
| Critical evidence invariants | 100% | 100.0% | yes |
| Supported-task completion | >= 80% | 100.0% | yes |
| Uplift over no-Skill baseline | >= 20 pp | 54.5 pp | yes |
| Unauthorized or unsafe actions | 0 | 0 | yes |

The baseline completed 45.5% of the 11 supported cases. The with-Skill arm
completed 100.0%. The dataset also contains 12 negative cases used to test
discovery boundaries.

## Remaining failure

The sole false activation was `local-ai-autopilot-neg-jetson-setup`. The Agent
correctly refused to use LocalPilot to install JetPack, drivers, or CUDA and
routed the work to a Jetson-specific setup path. The independent evaluator
still counted it as activation because the Agent read and applied the
LocalPilot Skill's scope and safety rules. This is a discovery precision issue;
it did not produce an unsafe system change or a false hardware claim.

## Method

- Date: 2026-09-21
- Skill commit: `42112469baacbb0765a3967197cff293e761f6aa`
- Agent and evaluator model: `gpt-5.6-sol`
- Runtime: `codex-cli 0.155.1`
- Sandbox: read-only, fresh isolated workspace for each arm
- Cases: 23 total; 11 positive and 12 negative
- Runs: 46 Agent runs plus 23 independent pair reviews
- Difference between arms: only Skill discoverability

Both arms received the same deterministic LocalPilot CLI fixture. Each review
checked observable actions and output against the committed case ground truth.
Activation and task completion were scored separately, so declining an
out-of-scope request could complete the task without counting as Skill use.
The scoring script then applied the gates from the committed evaluation
contract; the reported uplift was generated from reviewed records rather than
entered by hand.

## Evidence

- [Run manifest](results/skill-behavior-eval-20260921/manifest.json)
- [Dataset snapshot](results/skill-behavior-eval-20260921/dataset.json)
- [Baseline records](results/skill-behavior-eval-20260921/baseline.json)
- [With-Skill records](results/skill-behavior-eval-20260921/with-skill.json)
- [Generated score](results/skill-behavior-eval-20260921/score.json)
- [Case transcripts and evaluator judgments](results/skill-behavior-eval-20260921/evidence.jsonl)
- [Evaluation contract](.agents/skills/local-ai-autopilot/references/evaluation.md)
- [Evaluation runner](scripts/run_skill_behavior_evals.py)

The full local run directory also preserves per-process event streams. The
committed JSONL bundle contains the useful command and Agent-message trace,
final response, run metadata, and independent judgment for every case while
excluding disposable workspace copies.

## Limits

Each case was run once per arm with one model, and the evaluator used the same
model family. The deterministic fixture makes orchestration and evidence rules
repeatable, but it does not replace execution on an Apple Silicon machine or a
real NVIDIA node. This is a project-owned Codex harness, not a result from
NVIDIA SkillEvaluator.
