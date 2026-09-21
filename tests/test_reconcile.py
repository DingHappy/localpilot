from __future__ import annotations

import tempfile
import unittest
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from helpers import candidate, metrics
from localpilot.control.reconcile import (
    ReconcilePolicy,
    Reconciler,
    RuntimeObservation,
    ServiceObjectives,
)
from localpilot import cli
from localpilot.profiles.store import ProfileStore
from localpilot.schemas import SavedProfile


REQUIREMENTS = {
    "task": "chat",
    "privacy": "local_only",
    "priority": "latency",
    "quality": "high",
    "context_length": 8192,
    "concurrency": 1,
    "capabilities": [],
    "modalities": ["text"],
    "languages": ["en"],
}


def saved_profile(key: str = "profile-a") -> SavedProfile:
    benchmark = metrics(ttft_ms=100, throughput_tokens_s=100, quality=0.9)
    benchmark.ttft_p95_ms = 120
    plan = candidate(runtime="mock", simulated=True)
    return SavedProfile(
        profile_key=key,
        hardware_fingerprint="hardware-a",
        task="chat",
        priority="latency",
        candidate=plan,
        benchmark=benchmark,
        score=90.0,
        runtime_versions={"runtime": "mock"},
        simulated=True,
        created_at="2026-09-21T00:00:00+00:00",
        last_verified_at="2026-09-21T00:00:00+00:00",
        requirements=REQUIREMENTS,
    )


class Clock:
    def __init__(self):
        self.value = datetime(2026, 9, 21, tzinfo=timezone.utc)

    def __call__(self) -> str:
        return self.value.isoformat()

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ProfileStore(Path(self.directory.name))
        self.profile = saved_profile()
        self.store.save(self.profile)
        self.clock = Clock()
        self.reconciler = Reconciler(self.store, now=self.clock)
        self.objectives = ServiceObjectives.from_profile(self.profile)

    def test_healthy_observation_keeps_the_profile(self):
        result = self.reconciler.reconcile(
            RuntimeObservation(
                timestamp=self.clock(),
                ttft_p95_ms=130,
                throughput_tokens_s=90,
                error_rate=0,
                quality=0.9,
                queue_depth=0,
                healthy=True,
            ),
            self.objectives,
            hardware_fingerprint="hardware-a",
        )
        self.assertEqual(result.decision, "KEEP")
        self.assertFalse(result.actionable)
        self.assertFalse(result.safe_execution["automatic_apply"])

    def test_one_latency_spike_does_not_trigger_reconfiguration(self):
        result = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock(), ttft_p95_ms=999),
            self.objectives,
            ReconcilePolicy(consecutive_breaches=3),
        )
        self.assertEqual(result.decision, "KEEP")
        self.assertEqual(result.current_breaches, ["ttft_p95_ms"])
        self.assertEqual(result.sustained_breaches, [])

    def test_three_latency_breaches_trigger_reconfiguration(self):
        policy = ReconcilePolicy(consecutive_breaches=3, cooldown_seconds=0)
        results = []
        for _ in range(3):
            results.append(
                self.reconciler.reconcile(
                    RuntimeObservation(timestamp=self.clock(), ttft_p95_ms=999),
                    self.objectives,
                    policy,
                )
            )
            self.clock.advance(10)
        self.assertEqual(results[-1].decision, "RECONFIGURE")
        self.assertTrue(results[-1].actionable)
        self.assertEqual(results[-1].sustained_breaches, ["ttft_p95_ms"])
        self.assertTrue(results[-1].safe_execution["requires_drain"])

    def test_polling_the_same_telemetry_window_does_not_count_twice(self):
        policy = ReconcilePolicy(consecutive_breaches=2, cooldown_seconds=0)
        duplicate = RuntimeObservation(
            timestamp=self.clock(),
            source_sequence="server-a:1",
            request_p95_ms=999,
        )
        first = self.reconciler.reconcile(duplicate, self.objectives, policy)
        self.clock.advance(10)
        second = self.reconciler.reconcile(duplicate, self.objectives, policy)
        self.assertEqual(first.decision, "KEEP")
        self.assertEqual(second.decision, "KEEP")
        self.assertEqual(second.sustained_breaches, [])

        self.clock.advance(10)
        third = self.reconciler.reconcile(
            RuntimeObservation(
                timestamp=self.clock(),
                source_sequence="server-a:2",
                request_p95_ms=999,
            ),
            self.objectives,
            policy,
        )
        self.assertEqual(third.decision, "RECONFIGURE")

    def test_sustained_quality_loss_recommends_switch(self):
        policy = ReconcilePolicy(consecutive_breaches=2, cooldown_seconds=0)
        self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock(), quality=0.5),
            self.objectives,
            policy,
        )
        self.clock.advance(10)
        result = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock(), quality=0.5),
            self.objectives,
            policy,
        )
        self.assertEqual(result.decision, "SWITCH")
        self.assertTrue(result.safe_execution["requires_canary"])
        self.assertEqual(result.safe_execution["rollback_profile_key"], "profile-a")

    def test_changed_environment_requires_rebenchmark(self):
        result = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock()),
            self.objectives,
            hardware_fingerprint="hardware-b",
        )
        self.assertEqual(result.decision, "REBENCH")
        self.assertTrue(result.environment_changed)

    def test_changed_capability_requires_switch(self):
        desired = {**REQUIREMENTS, "modalities": ["image", "text"]}
        result = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock()),
            self.objectives,
            desired_requirements=desired,
        )
        self.assertEqual(result.decision, "SWITCH")
        self.assertTrue(result.requirements_changed)

    def test_changed_concurrency_recommends_reconfiguration(self):
        desired = {**REQUIREMENTS, "concurrency": 8}
        result = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock()),
            self.objectives,
            desired_requirements=desired,
        )
        self.assertEqual(result.decision, "RECONFIGURE")

    def test_cooldown_defers_repeated_actionable_plan(self):
        policy = ReconcilePolicy(consecutive_breaches=1, cooldown_seconds=900)
        first = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock(), error_rate=0.5),
            self.objectives,
            policy,
        )
        self.clock.advance(30)
        second = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock(), error_rate=0.5),
            self.objectives,
            policy,
        )
        self.assertTrue(first.actionable)
        self.assertFalse(second.actionable)
        self.assertTrue(second.deferred_by_cooldown)
        self.assertEqual(second.decision, "REBENCH")

    def test_no_ready_profile_returns_a_non_destructive_switch_plan(self):
        self.store.stop()
        result = self.reconciler.reconcile(
            RuntimeObservation(timestamp=self.clock()), self.objectives
        )
        self.assertEqual(result.decision, "SWITCH")
        self.assertTrue(result.actionable)
        self.assertFalse(result.safe_execution["automatic_apply"])

    def test_observation_validation_rejects_invalid_rates(self):
        with self.assertRaises(ValueError):
            RuntimeObservation.from_mapping({"error_rate": 2})

    def test_objective_validation_rejects_invalid_rates(self):
        with self.assertRaises(ValueError):
            ServiceObjectives(max_error_rate=1.5)


class ReconcileCliTests(unittest.TestCase):
    def test_cli_without_a_profile_writes_a_reviewable_switch_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.dict("os.environ", {"LOCALPILOT_HOME": directory}), redirect_stdout(output):
                code = cli.main(["reconcile", "--json"])
            payload = json.loads(output.getvalue())
            plans = list((Path(directory) / "profiles" / "control" / "plans").glob("*.json"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"], "SWITCH")
        self.assertFalse(payload["safe_execution"]["automatic_apply"])
        self.assertEqual(len(plans), 1)

    @patch("localpilot.cli._post_json")
    def test_watch_stops_when_a_plan_becomes_actionable(self, post_json):
        post_json.side_effect = [
            {
                "created_at": "2026-09-21T00:00:00+00:00",
                "decision": "KEEP",
                "actionable": False,
                "reasons": ["within limits"],
            },
            {
                "created_at": "2026-09-21T00:00:01+00:00",
                "decision": "RECONFIGURE",
                "actionable": True,
                "reasons": ["sustained latency"],
            },
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(
                ["watch", "--samples", "5", "--interval", "0", "--json"]
            )
        lines = output.getvalue().splitlines()
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[-1])["decision"], "RECONFIGURE")
        self.assertEqual(post_json.call_count, 2)

    @patch("localpilot.cli._post_json")
    def test_drain_returns_nonzero_when_in_flight_work_does_not_finish(self, post_json):
        post_json.return_value = {
            "status": "TIMEOUT",
            "drained": False,
            "accepting": True,
            "active_requests": 1,
        }
        with redirect_stdout(io.StringIO()):
            code = cli.main(["drain", "--timeout", "0"])
        self.assertEqual(code, 2)

    @patch("localpilot.cli._post_json")
    def test_resume_reopens_api_traffic(self, post_json):
        post_json.return_value = {
            "state": "SERVING",
            "accepting": True,
            "active_requests": 0,
        }
        with redirect_stdout(io.StringIO()):
            code = cli.main(["resume"])
        self.assertEqual(code, 0)

    @patch("localpilot.cli._post_json")
    def test_activate_sends_the_staged_profile_key(self, post_json):
        post_json.return_value = {
            "status": "ACTIVATED",
            "profile_key": "profile-b",
            "previous_profile_key": "profile-a",
            "probe_status": "passed",
        }
        with redirect_stdout(io.StringIO()):
            code = cli.main(["activate", "profile-b"])
        self.assertEqual(code, 0)
        self.assertEqual(
            post_json.call_args.args[1], {"profile_key": "profile-b"}
        )


if __name__ == "__main__":
    unittest.main()
