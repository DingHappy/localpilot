import unittest

from localpilot.hardware.profiler import HardwareProfiler


class HardwareProfilerTests(unittest.TestCase):
    def test_mock_profile_is_unambiguously_simulated(self):
        profile = HardwareProfiler().profile(simulate=True)
        self.assertTrue(profile.simulated)
        self.assertFalse(profile.real_execution_ready)
        self.assertEqual(profile.available_devices, ["CPU", "GPU", "NPU"])
        self.assertTrue(any("simulation" in note.lower() for note in profile.notes))

    def test_native_arm_is_not_reported_as_i386(self):
        profile = HardwareProfiler().profile(simulate=False)
        if profile.os["machine"] == "arm64":
            self.assertNotEqual(profile.cpu["model"].lower(), "i386")


if __name__ == "__main__":
    unittest.main()
