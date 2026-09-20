import unittest

from localpilot.engines.registry import EngineRegistry, EngineSpec, probe_engine


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = EngineRegistry()

    def test_the_configured_engines_load(self):
        ids = self.registry.ids()
        for expected in ("vllm", "trtllm", "sglang", "nim", "llamacpp"):
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

    def test_probing_reports_every_engine(self):
        reports = self.registry.probe_all()
        self.assertEqual(set(reports), set(self.registry.ids()))
        for report in reports.values():
            self.assertIn("available", report)


class ContainerProbeTests(unittest.TestCase):
    def test_a_container_engine_needs_the_image_not_just_the_runtime(self):
        """Docker on PATH says nothing about whether the image is here.

        Counting it as available would put a candidate in the plan that
        cannot start without a multi-gigabyte pull.
        """
        spec = EngineSpec(
            engine_id="nim-test",
            display_name="test",
            kind="container",
            probe={"binary": "docker", "docker_image_prefix": "nvcr.io/nim/"},
            openai_base_path="/v1",
        )
        report = probe_engine(spec)
        if report.get("binary_path") and not report.get("local_images"):
            self.assertFalse(report["available"])
            self.assertIn("no nvcr.io/nim/* image", report["detail"])

    def test_a_missing_runtime_is_simply_unavailable(self):
        spec = EngineSpec(
            engine_id="ghost",
            display_name="test",
            kind="container",
            probe={"binary": "definitely-not-a-real-binary",
                   "docker_image_prefix": "nvcr.io/nim/"},
        )
        report = probe_engine(spec)
        self.assertFalse(report["available"])
        self.assertIsNone(report["binary_path"])


if __name__ == "__main__":
    unittest.main()
