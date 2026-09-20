import tempfile
import unittest
from pathlib import Path

from localpilot.demo import run_demo
from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore


class DemoTests(unittest.TestCase):
    def test_demo_contrasts_manual_and_autopilot(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                store=ProfileStore(Path(directory))
            )
            demo = run_demo(mode="mock", orchestrator=orchestrator)

        self.assertFalse(demo["demo_a"]["executed"])
        self.assertGreaterEqual(len(demo["demo_a"]["user_decisions"]), 5)
        result = demo["demo_b"]["result"]
        self.assertEqual(result["status"], "READY")
        self.assertTrue(result["hardware"]["simulated"])
        self.assertIn("not a hardware benchmark", demo["disclaimer"])


if __name__ == "__main__":
    unittest.main()

