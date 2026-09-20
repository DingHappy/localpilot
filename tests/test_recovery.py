import unittest
import tempfile
from pathlib import Path

from localpilot.hardware.profiler import HardwareProfiler
from localpilot.models.registry import ModelRegistry
from localpilot.orchestrator import Orchestrator
from localpilot.planner.planner import Planner
from localpilot.profiles.store import ProfileStore
from localpilot.runtime.mock import MockRuntime
from localpilot.schemas import CandidatePlan


class RecoveryPlannerTests(unittest.TestCase):
    def test_recovery_reduces_context_then_adds_cpu(self):
        hardware = HardwareProfiler().profile(simulate=True)
        model = ModelRegistry().get(
            "qwen2.5-coder-1.5b-instruct-int4-ov"
        )
        failed = CandidatePlan(
            candidate_id="failed-gpu",
            model_id=model.model_id,
            source_id=model.source_id,
            device="GPU",
            precision="INT4_ASYM",
            context_length=8192,
            runtime="mock",
            expected_memory_gb=2,
            quality_score=model.quality_score,
            reason="test",
            confidence=0.9,
            simulated=True,
        )

        recovery = Planner().recovery_plan([failed], hardware, [model])

        self.assertEqual(len(recovery), 2)
        self.assertEqual(recovery[0].device, "GPU")
        self.assertEqual(recovery[0].context_length, 4096)
        self.assertEqual(recovery[0].recovery_action, "reduce_context")
        self.assertEqual(recovery[1].device, "CPU")
        self.assertEqual(
            recovery[1].recovery_action,
            "fallback_device_and_context",
        )
        self.assertTrue(
            all(item.fallback_of == "failed-gpu" for item in recovery)
        )

    def test_recovery_attempts_are_bounded(self):
        hardware = HardwareProfiler().profile(simulate=True)
        model = ModelRegistry().get(
            "qwen2.5-coder-1.5b-instruct-int4-ov"
        )
        failed = CandidatePlan(
            candidate_id="failed-gpu",
            model_id=model.model_id,
            source_id=model.source_id,
            device="GPU",
            precision="INT4_ASYM",
            context_length=8192,
            runtime="mock",
            expected_memory_gb=2,
            quality_score=model.quality_score,
            reason="test",
            confidence=0.9,
            simulated=True,
        )
        recovery = Planner().recovery_plan([failed], hardware, [model])
        self.assertLessEqual(len(recovery), 2)


class RecoverAfterContextFailureRuntime(MockRuntime):
    def load_model(self, selected):
        if selected.context_length > 4096:
            raise RuntimeError("injected context allocation failure")
        super().load_model(selected)


class RecoveryOrchestratorTests(unittest.TestCase):
    def test_orchestrator_executes_recovery_after_all_primary_fail(self):
        def resolver(name):
            self.assertEqual(name, "mock")
            return RecoverAfterContextFailureRuntime

        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                store=ProfileStore(Path(directory)),
                runtime_resolver=resolver,
            )
            result = orchestrator.autopilot(
                "Completely local coding assistant, latency first",
                mode="mock",
                reuse_profile=False,
            )

        self.assertEqual(result.status, "READY")
        self.assertTrue(
            any(item.candidate.fallback_of for item in result.candidates)
        )
        self.assertTrue(
            any("bounded recovery" in warning for warning in result.warnings)
        )
        self.assertLessEqual(result.best_profile.candidate.context_length, 4096)


if __name__ == "__main__":
    unittest.main()
