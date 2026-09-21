import os
import unittest
from unittest import mock

from localpilot.engines.registry import EngineRegistry, EngineSpec, probe_engine


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = EngineRegistry()

    def test_the_configured_engines_load(self):
        ids = self.registry.ids()
        for expected in (
            "vllm", "trtllm", "sglang", "nim", "llamacpp", "ollama"
        ):
            self.assertIn(expected, ids)

    def test_an_unknown_engine_raises(self):
        with self.assertRaises(KeyError):
            self.registry.get("no-such-engine")

    def test_only_http_engines_count_as_servable(self):
        """Measurement goes through one wire protocol.

        An engine with no HTTP surface cannot be timed the same way as the
        others, so it must not enter a comparison.
        """
        self.assertTrue(self.registry.servable("vllm"))
        self.assertTrue(self.registry.servable("nim"))
        self.assertFalse(self.registry.servable("transformers"))
        self.assertFalse(self.registry.servable("no-such-engine"))

    def test_knobs_and_features_are_declared(self):
        vllm = self.registry.get("vllm")
        self.assertIn("kv_cache_dtype", vllm.knobs)
        self.assertTrue(vllm.supports_feature("nvfp4"))
        self.assertTrue(vllm.supports_feature("speculative_decoding"))

    def test_llamacpp_cannot_serve_nvfp4(self):
        """GGUF-only, which rules it out of this platform's native format."""
        self.assertFalse(self.registry.get("llamacpp").supports_feature("nvfp4"))

    def test_ollama_is_the_metal_openai_compatible_engine(self):
        ollama = self.registry.get("ollama")
        self.assertEqual(ollama.openai_base_path, "/v1")
        self.assertTrue(ollama.supports_feature("metal"))
        self.assertTrue(self.registry.servable("ollama"))

    def test_probing_reports_every_engine(self):
        reports = self.registry.probe_all()
        self.assertEqual(set(reports), set(self.registry.ids()))
        for report in reports.values():
            self.assertIn("available", report)


class ContainerProbeTests(unittest.TestCase):
    def test_a_container_engine_needs_the_image_not_just_the_runtime(self):
        """Docker on PATH says nothing about whether the image is here.

        Counting it as available would put a candidate in the plan that
        cannot start without a multi-gigabyte pull. The container runtime
        is not the engine, so the host fallback must not rescue it.
        """
        spec = EngineSpec(
            engine_id="nim-test",
            display_name="test",
            kind="container",
            probe={"binary": "docker",
                   "docker_image_contains": ["nvcr.io/nim/"]},
            openai_base_path="/v1",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            report = probe_engine(spec)
        if report.get("binary_path") and not report.get("local_images"):
            self.assertFalse(report["available"])
            self.assertIn("no image matching", report["detail"])

    def test_a_missing_runtime_is_simply_unavailable(self):
        spec = EngineSpec(
            engine_id="ghost",
            display_name="test",
            kind="container",
            probe={"binary": "definitely-not-a-real-binary",
                   "docker_image_contains": ["nvcr.io/nim/"]},
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            report = probe_engine(spec)
        self.assertFalse(report["available"])
        self.assertIsNone(report["binary_path"])


class ProbeSourceTests(unittest.TestCase):
    """On DGX Spark the supported way to run vLLM is a container.

    A probe that only looks at the host PATH reports every engine missing
    on a machine that already holds the official image, which is what
    happened on the real node.
    """

    def _spec(self, kind="server"):
        return EngineSpec(
            engine_id="vllm",
            display_name="vLLM",
            kind=kind,
            probe={"python_module": "definitely_not_installed",
                   "binary": "definitely-not-a-real-binary",
                   "docker_image_contains": ["nvcr.io/nvidia/vllm"]},
            openai_base_path="/v1",
            health_path="/health",
        )

    def test_a_local_image_makes_a_server_engine_available(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("localpilot.engines.registry._local_images",
                        return_value=["nvcr.io/nvidia/vllm:26.02-py3"]), \
             mock.patch("localpilot.engines.registry._container_runtime",
                        return_value="/usr/bin/docker"):
            report = probe_engine(self._spec())
        self.assertTrue(report["available"])
        self.assertEqual(report["source"], "container")
        self.assertIn("26.02-py3", report["detail"])

    def test_a_reachable_endpoint_outranks_everything(self):
        """Something answering right now is proof, not inference."""
        with mock.patch.dict(
            os.environ, {"LOCALPILOT_VLLM_BASE_URL": "http://127.0.0.1:8000"},
            clear=True,
        ), mock.patch("localpilot.engines.registry._endpoint_answers",
                      return_value=True):
            report = probe_engine(self._spec())
        self.assertTrue(report["available"])
        self.assertEqual(report["source"], "reachable")

    def test_a_configured_but_dead_endpoint_says_so(self):
        with mock.patch.dict(
            os.environ, {"LOCALPILOT_VLLM_BASE_URL": "http://127.0.0.1:8000"},
            clear=True,
        ), mock.patch("localpilot.engines.registry._endpoint_answers",
                      return_value=False), \
             mock.patch("localpilot.engines.registry._local_images",
                        return_value=[]):
            report = probe_engine(self._spec())
        self.assertFalse(report["available"])
        self.assertIn("not answering", report["detail"])

    def test_every_report_names_the_signal_it_used(self):
        for report in EngineRegistry().probe_all().values():
            self.assertIn("source", report)
            if report["available"]:
                self.assertIn(
                    report["source"], {"reachable", "container", "host"}
                )


if __name__ == "__main__":
    unittest.main()
