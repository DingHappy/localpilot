import unittest

from localpilot.executor.executor import Executor
from localpilot.runtime.mock import MockRuntime
from localpilot.schemas import CandidatePlan


def candidate(device):
    return CandidatePlan(
        candidate_id=f"candidate-{device.lower()}",
        model_id="test-coder",
        source_id="test-coder",
        device=device,
        precision="INT4",
        context_length=8192,
        runtime="mock",
        expected_memory_gb=1,
        quality_score=0.8,
        reason="test",
        confidence=1,
        simulated=True,
    )


class FailGpuRuntime(MockRuntime):
    def load_model(self, selected):
        if selected.device == "GPU":
            raise RuntimeError("injected GPU failure")
        super().load_model(selected)


class ExecutorFallbackTests(unittest.TestCase):
    def test_failure_does_not_abort_later_candidate(self):
        results = Executor(FailGpuRuntime).execute(
            [candidate("GPU"), candidate("CPU")],
            task="coding",
        )
        self.assertEqual(results[0].status, "failed")
        self.assertEqual(results[1].status, "success")
        self.assertIsNotNone(results[1].benchmark)


if __name__ == "__main__":
    unittest.main()

