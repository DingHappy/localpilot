import unittest

from localpilot.planner.policies import PolicyEngine
from localpilot.planner.scoring import score_results
from localpilot.schemas import (
    BenchmarkMetrics,
    CandidatePlan,
    CandidateResult,
)


def result(name, ttft, throughput, memory, quality):
    candidate = CandidatePlan(
        candidate_id=name,
        model_id=name,
        source_id=name,
        device="CPU",
        precision="INT4",
        context_length=8192,
        runtime="mock",
        expected_memory_gb=memory,
        quality_score=quality,
        reason="test",
        confidence=1,
        simulated=True,
    )
    metrics = BenchmarkMetrics(
        ttft_ms=ttft,
        tpot_ms=1000 / throughput,
        throughput_tokens_s=throughput,
        total_latency_ms=ttft * 3,
        peak_memory_gb=memory,
        cpu_usage_percent=None,
        stability=1,
        quality=quality,
        runs=3,
        warmup_runs=1,
        simulated=True,
    )
    return CandidateResult(candidate, "success", [], metrics)


class ScoringTests(unittest.TestCase):
    def test_failed_quality_gate_is_not_scored(self):
        fast_bad = result("fast-bad", 100, 100, 1, 0.2)
        slower_good = result("slower-good", 300, 30, 2, 0.8)
        scored = score_results(
            [fast_bad, slower_good], "latency", PolicyEngine()
        )
        self.assertIsNone(scored[0].score)
        self.assertIsNotNone(scored[1].score)


if __name__ == "__main__":
    unittest.main()

