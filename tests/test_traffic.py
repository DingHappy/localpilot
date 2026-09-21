from __future__ import annotations

import unittest

from localpilot.control.traffic import TrafficGate


class TrafficGateTests(unittest.TestCase):
    def test_drain_rejects_new_work_after_existing_work_finishes(self):
        gate = TrafficGate()
        self.assertTrue(gate.acquire())
        gate.release()
        result = gate.drain(timeout=0)
        self.assertTrue(result["drained"])
        self.assertFalse(gate.acquire())
        self.assertEqual(gate.status()["state"], "DRAINING")

    def test_timeout_restores_service(self):
        gate = TrafficGate()
        self.assertTrue(gate.acquire())
        result = gate.drain(timeout=0)
        self.assertFalse(result["drained"])
        self.assertTrue(result["accepting"])
        gate.release()
        self.assertTrue(gate.acquire())
        gate.release()

    def test_resume_reopens_a_successfully_drained_gate(self):
        gate = TrafficGate()
        gate.drain(timeout=0)
        state = gate.resume()
        self.assertEqual(state["state"], "SERVING")
        self.assertTrue(gate.acquire())
        gate.release()

    def test_negative_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            TrafficGate().drain(timeout=-1)


if __name__ == "__main__":
    unittest.main()
