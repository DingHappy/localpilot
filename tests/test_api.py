import unittest
import os
import shutil
import tempfile
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

API_AVAILABLE = all(find_spec(name) is not None for name in ("fastapi", "httpx"))


@unittest.skipUnless(API_AVAILABLE, "Install the api extra to run API tests")
class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from localpilot.api.server import create_app

        cls.state_directory = tempfile.TemporaryDirectory()
        root = Path(cls.state_directory.name)
        shutil.copytree(Path(__file__).resolve().parent.parent / "config", root / "config")
        cls.environment = patch.dict(os.environ, {"LOCALPILOT_HOME": str(root)})
        cls.environment.start()
        cls.client = TestClient(create_app())

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.environment.stop()
        cls.state_directory.cleanup()

    def test_health_lists_the_available_runtimes(self):
        body = self.client.get("/health").json()
        self.assertIn("mock", body["runtimes"])
        self.assertIn("vllm", body["runtimes"])

    def test_hardware_can_be_requested_simulated(self):
        body = self.client.get("/v1/hardware?simulate=true").json()
        self.assertTrue(body["simulated"])
        self.assertTrue(body["unified_memory"])
        self.assertEqual(body["platform_id"], "dgx_spark")

    def test_engines_report_knobs_and_probe_results(self):
        data = self.client.get("/v1/engines").json()["data"]
        by_id = {item["engine_id"]: item for item in data}
        self.assertIn("vllm", by_id)
        self.assertIn("probe", by_id["vllm"])
        self.assertIn("kv_cache_dtype", by_id["vllm"]["knobs"])

    def test_the_registry_is_sized_against_the_machine(self):
        """A catalogue without sizing is not decision-useful."""
        body = self.client.get("/v1/registry?task=coding").json()
        self.assertFalse(body["simulated"])
        data = body["data"]
        self.assertTrue(data)
        for entry in data:
            self.assertIn("sizing", entry)
            self.assertIn("fits", entry["sizing"])
            self.assertIn("max_context_at_budget", entry)
        oversized = [item for item in data if not item["sizing"]["fits"]]
        self.assertTrue(oversized, "the 550B checkpoint should not fit 128 GB")

    def test_registry_simulation_is_explicit(self):
        body = self.client.get("/v1/registry?task=coding&simulate=true").json()
        self.assertTrue(body["simulated"])

    def test_intent_parses_without_running_anything(self):
        body = self.client.post(
            "/v1/intent", json={"goal": "本地代码审查，速度优先"}
        ).json()
        self.assertEqual(body["task"], "coding")
        self.assertEqual(body["priority"], "latency")

    def test_intent_requires_a_goal(self):
        self.assertEqual(
            self.client.post("/v1/intent", json={}).status_code, 400
        )

    def test_autopilot_runs_as_a_job_and_reports_its_trace(self):
        started = self.client.post(
            "/v1/autopilot",
            json={"goal": "本地代码审查，速度优先", "mode": "mock",
                  "wait": True, "timeout": 120},
        ).json()
        self.assertEqual(started["status"], "succeeded", started.get("error"))

        polled = self.client.get(f"/v1/autopilot/{started['job_id']}").json()
        result = polled["result"]
        self.assertEqual(result["status"], "READY")
        self.assertTrue(result["agent_trace"])
        self.assertTrue(result["best_profile"]["candidate"]["engine"])

    def test_an_unknown_job_is_a_404(self):
        self.assertEqual(
            self.client.get("/v1/autopilot/not-a-job").status_code, 404
        )

    def test_profiles_and_runs_are_listable(self):
        self.client.post(
            "/v1/autopilot",
            json={"goal": "本地代码审查，速度优先", "mode": "mock",
                  "wait": True, "timeout": 120},
        )
        profiles = self.client.get("/v1/profiles").json()["data"]
        runs = self.client.get("/v1/runs").json()["data"]
        self.assertTrue(profiles)
        self.assertTrue(runs)
        self.assertIn("engine", profiles[0])

    def test_chat_serves_from_the_active_profile(self):
        self.client.post(
            "/v1/autopilot",
            json={"goal": "本地代码审查，速度优先", "mode": "mock",
                  "wait": True, "timeout": 120},
        )
        body = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        ).json()
        self.assertTrue(body["choices"][0]["message"]["content"])
        self.assertTrue(body["localpilot"]["simulated"])
        self.assertTrue(body["localpilot"]["engine"])

    def test_streaming_is_refused_rather_than_silently_ignored(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        self.assertEqual(response.status_code, 400)

    def test_api_telemetry_can_trigger_a_non_destructive_reconcile_plan(self):
        self.client.post(
            "/v1/autopilot",
            json={"goal": "本地代码审查，速度优先", "mode": "mock",
                  "wait": True, "timeout": 120},
        )
        self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "private input"}]},
        )
        telemetry = self.client.get("/v1/telemetry").json()
        self.assertGreaterEqual(telemetry["window_requests"], 1)
        self.assertGreater(telemetry["request_p95_ms"], 0)
        self.assertNotIn("private input", str(telemetry))

        plan = self.client.post(
            "/v1/reconcile",
            json={
                "thresholds": {"max_request_p95_ms": 0},
                "window": 1,
                "cooldown_seconds": 0,
            },
        ).json()
        self.assertEqual(plan["decision"], "RECONFIGURE")
        self.assertTrue(plan["actionable"])
        self.assertFalse(plan["safe_execution"]["automatic_apply"])
        self.assertTrue(plan["safe_execution"]["requires_drain"])

    def test_drain_blocks_new_requests_until_resume(self):
        self.client.post(
            "/v1/autopilot",
            json={"goal": "本地代码审查，速度优先", "mode": "mock",
                  "wait": True, "timeout": 120},
        )
        drained = self.client.post("/internal/drain", json={"timeout": 0}).json()
        self.assertTrue(drained["drained"])
        try:
            blocked = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            self.assertEqual(blocked.status_code, 503)
        finally:
            resumed = self.client.post("/internal/resume").json()
        self.assertEqual(resumed["state"], "SERVING")
        accepted = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(accepted.status_code, 200)

    def test_activation_requires_drain_and_runs_a_business_probe(self):
        job = self.client.post(
            "/v1/autopilot",
            json={"goal": "本地代码审查，速度优先", "mode": "mock",
                  "wait": True, "timeout": 120},
        ).json()
        profile_key = job["result"]["best_profile"]["profile_key"]
        refused = self.client.post(
            "/internal/activate", json={"profile_key": profile_key}
        )
        self.assertEqual(refused.status_code, 409)

        self.client.post("/internal/drain", json={"timeout": 0})
        try:
            activated = self.client.post(
                "/internal/activate", json={"profile_key": profile_key}
            )
            self.assertEqual(activated.status_code, 200, activated.text)
            body = activated.json()
            self.assertEqual(body["status"], "ACTIVATED")
            self.assertEqual(body["probe_status"], "passed")
            self.assertFalse(body["traffic"]["accepting"])
        finally:
            self.client.post("/internal/resume")

    def test_the_dashboard_is_served_without_external_assets(self):
        """A local-only tool whose UI needs a CDN contradicts itself."""
        html = self.client.get("/").text
        self.assertIn("<title>LocalPilot</title>", html)
        for marker in ("cdn.", "https://unpkg", "googleapis"):
            self.assertNotIn(marker, html)


if __name__ == "__main__":
    unittest.main()
