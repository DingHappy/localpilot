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
        if profile.platform_id == "apple_silicon":
            self.skipTest("this machine has a Metal accelerator")
        if profile.accelerator.get("detected"):
            self.skipTest("this machine has a CUDA device")
        self.assertFalse(profile.real_execution_ready)
        self.assertTrue(
            any("CUDA" in note for note in profile.notes), profile.notes
        )
        self.assertIn("CPU", profile.available_devices)

    def test_apple_silicon_is_a_real_unified_memory_target(self):
        profile = HardwareProfiler().profile(simulate=False)
        if profile.os["name"] != "Darwin" or profile.os["machine"] != "arm64":
            self.skipTest("this machine is not Apple Silicon")
        self.assertEqual(profile.platform_id, "apple_silicon")
        self.assertTrue(profile.unified_memory)
        self.assertIn("METAL", profile.available_devices)
        self.assertEqual(profile.accelerator.get("source"), "metal")

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


class PlatformDetectionTests(unittest.TestCase):
    """Which platform this is decides the memory budget.

    A misidentification silently swaps the ceiling between board VRAM and
    system memory, which rewrites every gating decision. So the profile has
    to carry what the decision was based on.
    """

    def test_the_profile_records_what_identification_was_based_on(self):
        profile = HardwareProfiler().profile(simulate=False)
        detection = profile.stack.get("platform_detection")
        self.assertIsNotNone(detection)
        self.assertIn("matched_on", detection)
        self.assertIn("gpu_names", detection)
        self.assertIn("architecture", detection)
        self.assertTrue(detection["matched_on"])

    def test_a_gpu_name_match_is_reported_as_the_reason(self):
        profiler = HardwareProfiler()
        config = {
            "platforms": {
                "dgx_spark": {
                    "detect": {"gpu_name_contains": ["GB10"]},
                    "unified_memory": True,
                },
                "cuda_discrete": {"detect": {}, "unified_memory": False},
            }
        }
        platform_id, spec, signals = profiler._identify_platform(
            {"detected": True, "names": ["NVIDIA GB10"]}, "aarch64", config
        )
        self.assertEqual(platform_id, "dgx_spark")
        self.assertTrue(spec["unified_memory"])
        self.assertIn("GB10", signals["matched_on"])

    def test_an_unrecognised_cuda_device_falls_back_and_says_so(self):
        profiler = HardwareProfiler()
        config = {
            "platforms": {
                "dgx_spark": {
                    "detect": {"gpu_name_contains": ["GB10"]},
                    "unified_memory": True,
                },
                "cuda_discrete": {"detect": {}, "unified_memory": False},
            }
        }
        platform_id, spec, signals = profiler._identify_platform(
            {"detected": True, "names": ["Some Virtualised GPU"]},
            "aarch64",
            config,
        )
        self.assertEqual(platform_id, "cuda_discrete")
        self.assertFalse(spec["unified_memory"])
        self.assertIn("fallback", signals["matched_on"])

    def test_an_arm_cpu_alone_does_not_identify_the_platform(self):
        """Jetson is also aarch64 with CUDA, and its memory differs.

        config/devices.yaml once declared an `architecture` condition that
        the code never read. Removing it is only safe if architecture can
        never be sufficient on its own.
        """
        profiler = HardwareProfiler()
        config = {
            "platforms": {
                "dgx_spark": {
                    "detect": {"gpu_name_contains": ["GB10"]},
                    "unified_memory": True,
                },
                "cuda_discrete": {"detect": {}, "unified_memory": False},
            }
        }
        platform_id, _, _ = profiler._identify_platform(
            {"detected": True, "names": ["Jetson AGX Orin"]}, "aarch64", config
        )
        self.assertNotEqual(platform_id, "dgx_spark")

    def test_declared_detection_keys_are_all_read_by_the_code(self):
        """Guards against a config key the code ignores."""
        from localpilot.utils import config_file, load_data_file

        config = load_data_file(config_file("devices.yaml"))
        understood = {"gpu_name_contains", "device_tree_model_contains"}
        for platform_id, spec in (config.get("platforms") or {}).items():
            declared = set((spec.get("detect") or {}).keys())
            self.assertTrue(
                declared <= understood,
                f"{platform_id} declares {declared - understood}, which "
                "_identify_platform does not read",
            )


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
