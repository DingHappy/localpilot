from __future__ import annotations

import unittest
from unittest.mock import patch

from localpilot.control.telemetry import RequestTelemetry


class RequestTelemetryTests(unittest.TestCase):
    def test_snapshot_contains_only_aggregate_request_data(self):
        telemetry = RequestTelemetry(max_samples=3)
        with patch("localpilot.control.telemetry.time.perf_counter", side_effect=[1.0, 1.1]):
            token = telemetry.begin()
            telemetry.complete(token, ok=True)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["window_requests"], 1)
        self.assertEqual(snapshot["request_p95_ms"], 100.0)
        self.assertEqual(snapshot["error_rate"], 0.0)
        self.assertEqual(snapshot["active_requests"], 0)
        self.assertTrue(snapshot["source_sequence"].endswith(":1"))
        self.assertNotIn("prompt", snapshot)
        self.assertNotIn("response", snapshot)

    def test_window_is_bounded_and_error_rate_uses_the_window(self):
        telemetry = RequestTelemetry(max_samples=2)
        for ok in (True, False, False):
            token = telemetry.begin()
            telemetry.complete(token, ok=ok)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["window_requests"], 2)
        self.assertEqual(snapshot["error_rate"], 1.0)
        self.assertEqual(snapshot["total_requests"], 3)
        self.assertEqual(snapshot["total_errors"], 2)

    def test_active_request_is_visible_until_completion(self):
        telemetry = RequestTelemetry()
        token = telemetry.begin()
        self.assertEqual(telemetry.snapshot()["active_requests"], 1)
        telemetry.complete(token, ok=False)
        self.assertEqual(telemetry.snapshot()["active_requests"], 0)

    def test_sequence_changes_only_when_a_request_completes(self):
        telemetry = RequestTelemetry()
        first = telemetry.snapshot()["source_sequence"]
        self.assertEqual(telemetry.snapshot()["source_sequence"], first)
        token = telemetry.begin()
        self.assertEqual(telemetry.snapshot()["source_sequence"], first)
        telemetry.complete(token, ok=True)
        self.assertNotEqual(telemetry.snapshot()["source_sequence"], first)


if __name__ == "__main__":
    unittest.main()
