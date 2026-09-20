import unittest

from localpilot.hardware.nvidia import NvidiaProvider
from localpilot.hardware.profiler import HardwareProfiler


class SimulatedProfileTests(unittest.TestCase):
    def test_simulated_profile_is_unambiguously_labelled(self):
        profile = HardwareProfiler().profile(simulate=True)
        self.assertTrue(profile.simulated)
        self.assertFalse(profile.real_execution_ready)
        self.assertTrue(
            any("simulation" in note.lower() for note in profile.notes),
            profile.notes,
        )

    def test_simulated_profile_models_a_unified_memory_target(self):
        profile = HardwareProfiler().profile(simulate=True)
        self.assertEqual(profile.platform_id, "dgx_spark")
        self.assertTrue(profile.unified_memory)
        self.assertEqual(profile.available_devices, ["CUDA", "CPU"])
        self.assertIsNotNone(profile.memory_bandwidth_gbps)

    def test_simulated_and_real_fingerprints_differ(self):
        """A simulated profile must never match a real one in the store.

        Sharing a fingerprint would let a modelled result be reused as if
        it had been measured on the machine.
        """
        simulated = HardwareProfiler().profile(simulate=True)
        real = HardwareProfiler().profile(simulate=False)
        self.assertNotEqual(simulated.fingerprint, real.fingerprint)


class DetectedProfileTests(unittest.TestCase):
    def test_no_accelerator_means_not_ready_and_says_why(self):
        profile = HardwareProfiler().profile(simulate=False)
        if profile.accelerator.get("detected"):
            self.skipTest("this machine has a CUDA device")
        self.assertFalse(profile.real_execution_ready)
        self.assertTrue(
            any("CUDA" in note for note in profile.notes), profile.notes
        )
        self.assertIn("CPU", profile.available_devices)

    def test_native_arm_is_not_reported_as_i386(self):
        profile = HardwareProfiler().profile(simulate=False)
        if profile.os["machine"] in {"arm64", "aarch64"}:
            self.assertNotEqual(profile.cpu["model"].lower(), "i386")

    def test_every_engine_is_probed(self):
        profile = HardwareProfiler().profile(simulate=False)
        self.assertIn("vllm", profile.engines)
        self.assertIn("trtllm", profile.engines)
        for report in profile.engines.values():
            self.assertIn("available", report)


class ProviderTests(unittest.TestCase):
    def test_capabilities_report_the_platform_shape(self):
        provider = NvidiaProvider()
        provider.profile(simulate=True)
        capabilities = provider.capabilities()
        self.assertEqual(capabilities["provider"], "nvidia")
        self.assertEqual(capabilities["platform"], "dgx_spark")
        self.assertTrue(capabilities["unified_memory"])
        self.assertIn("CUDA", capabilities["devices"])


if __name__ == "__main__":
    unittest.main()
