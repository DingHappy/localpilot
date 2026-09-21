import os
import re
import unittest
from unittest import mock

from helpers import candidate
from localpilot.runtime.base import RuntimeUnavailable
from localpilot.runtime.openai_compat import (
    OllamaRuntime,
    StreamSample,
    TRTLLMRuntime,
    VLLMRuntime,
    _percentile,
)


def real(**overrides):
    plan = candidate(simulated=False, **overrides)
    plan.runtime = "vllm"
    return plan


class LaunchCommandTests(unittest.TestCase):
    """The rendered command is what the user is shown and may run by hand."""

    def test_vllm_command_carries_every_configured_knob(self):
        runtime = VLLMRuntime()
        plan = real(engine="vllm", context_length=32768, concurrency=8)
        command = runtime.build_launch_command(plan)

        self.assertEqual(command[:2], ["vllm", "serve"])
        self.assertIn(plan.source_id, command)
        self.assertIn("--max-model-len", command)
        self.assertEqual(command[command.index("--max-model-len") + 1], "32768")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "8")
        self.assertIn("--kv-cache-dtype", command)

    def test_speculative_decoding_adds_the_draft_configuration(self):
        runtime = VLLMRuntime()
        command = runtime.build_launch_command(real(speculative=True))
        rendered = " ".join(command)
        self.assertIn("--speculative-config", rendered)
        self.assertIn("test/draft", rendered)
        self.assertNotIn("{draft_source_id}", rendered)

    def test_prefix_caching_is_only_added_when_enabled(self):
        runtime = VLLMRuntime()
        plan = real()
        plan.runtime_config["enable_prefix_caching"] = False
        self.assertNotIn(
            "--enable-prefix-caching", runtime.build_launch_command(plan)
        )
        plan.runtime_config["enable_prefix_caching"] = True
        self.assertIn(
            "--enable-prefix-caching", runtime.build_launch_command(plan)
        )

    def test_each_engine_renders_its_own_flag_names(self):
        vllm = " ".join(VLLMRuntime().build_launch_command(real(engine="vllm")))
        trtllm = " ".join(
            TRTLLMRuntime().build_launch_command(real(engine="trtllm"))
        )
        self.assertIn("--max-model-len", vllm)
        self.assertIn("--max_seq_len", trtllm)
        self.assertNotIn("--max-model-len", trtllm)

    def test_ollama_launch_does_not_name_or_pull_a_model(self):
        runtime = OllamaRuntime()
        plan = real(engine="ollama")
        self.assertEqual(runtime.build_launch_command(plan), ["ollama", "serve"])

    def test_ollama_benchmark_disables_hidden_reasoning_by_default(self):
        runtime = OllamaRuntime()
        runtime.candidate = real(engine="ollama")
        with mock.patch.dict(os.environ, {}, clear=True):
            payload = runtime._payload("hello", 32, False)
        self.assertEqual(payload["reasoning_effort"], "none")

    def test_multimodal_payload_contains_the_real_image(self):
        runtime = VLLMRuntime()
        runtime.candidate = real(engine="vllm")
        runtime.model_name = "vision-model"
        payload = runtime._payload(
            {
                "text": "Name every color.",
                "image_url": "data:image/png;base64,AAAA",
            },
            32,
            False,
        )
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Name every color."})
        self.assertEqual(
            content[1]["image_url"]["url"], "data:image/png;base64,AAAA"
        )

    def test_no_template_placeholder_survives_rendering(self):
        """An unfilled {placeholder} would reach the shell verbatim.

        Braces themselves are legitimate here: vLLM's --speculative-config
        takes a JSON object. What must not survive is a template token that
        never got a value.
        """
        for runtime, engine in ((VLLMRuntime(), "vllm"), (TRTLLMRuntime(), "trtllm")):
            rendered = " ".join(
                runtime.build_launch_command(real(engine=engine, speculative=True))
            )
            leftover = re.findall(r"\{[a-z_]+\}", rendered)
            self.assertEqual(leftover, [], rendered)


class LaunchRefusalTests(unittest.TestCase):
    """Starting an engine downloads weights, so it cannot be implicit.

    One checkpoint in the shipped registry is 328 GB. The refusal has to
    hand the user the exact command instead of guessing on their behalf.
    """

    def test_no_server_and_no_opt_in_is_refused_with_the_command(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeUnavailable) as caught:
                VLLMRuntime().load_model(real())
        message = str(caught.exception)
        self.assertIn("will not start one implicitly", message)
        self.assertIn("LOCALPILOT_VLLM_BASE_URL", message)
        self.assertIn("LOCALPILOT_ALLOW_ENGINE_LAUNCH=1", message)
        self.assertIn("vllm serve", message)

    def test_a_configured_base_url_attaches_without_launching(self):
        with mock.patch.dict(
            os.environ,
            {"LOCALPILOT_VLLM_BASE_URL": "http://127.0.0.1:9999/"},
            clear=True,
        ):
            runtime = VLLMRuntime()
            runtime.load_model(real())
        self.assertEqual(runtime.base_url, "http://127.0.0.1:9999")
        self.assertFalse(runtime._owns_server)
        self.assertIsNone(runtime.process)

    def test_the_generic_variable_covers_engines_without_their_own(self):
        with mock.patch.dict(
            os.environ,
            {"LOCALPILOT_ENGINE_BASE_URL": "http://127.0.0.1:8888"},
            clear=True,
        ):
            runtime = TRTLLMRuntime()
            runtime.load_model(real(engine="trtllm"))
        self.assertEqual(runtime.base_url, "http://127.0.0.1:8888")

    def test_opting_in_targets_the_managed_port(self):
        with mock.patch.dict(
            os.environ, {"LOCALPILOT_ALLOW_ENGINE_LAUNCH": "1"}, clear=True
        ):
            runtime = VLLMRuntime()
            runtime.load_model(real())
        self.assertTrue(runtime._owns_server)
        self.assertIn("127.0.0.1", runtime.base_url)

    def test_a_simulated_candidate_is_refused_by_a_real_runtime(self):
        """The simulation must never be measured as if it were hardware."""
        with self.assertRaises(ValueError):
            VLLMRuntime().load_model(candidate(simulated=True))


class ServedModelNameTests(unittest.TestCase):
    """A repository id is not a served model name.

    An operator picks the served name with --served-model-name, and a
    container commonly serves a mounted path under a short alias, so
    sending the checkpoint's repo id gets a 404 from a server that is
    working perfectly. This was hit on the real node: vLLM served
    "step3-vl-10b-fp8" while the plan carried
    "stepfun-ai/Step3-VL-10B-FP8".
    """

    def _runtime(self, listing):
        runtime = VLLMRuntime()
        plan = real()
        runtime.spec = runtime.engines.get("vllm")
        runtime.candidate = plan
        runtime.model_name = plan.source_id
        runtime.base_url = "http://127.0.0.1:8000"
        runtime._get = lambda path, timeout=10.0: listing
        return runtime

    def test_a_different_served_name_is_adopted(self):
        runtime = self._runtime({"data": [{"id": "step3-vl-10b-fp8"}]})
        runtime._adopt_served_model_name()
        self.assertEqual(runtime.model_name, "step3-vl-10b-fp8")

    def test_a_matching_name_is_left_alone(self):
        runtime = self._runtime(
            {"data": [{"id": "other"}, {"id": "test/test-moe"}]}
        )
        runtime._adopt_served_model_name()
        self.assertEqual(runtime.model_name, "test/test-moe")

    def test_an_empty_listing_changes_nothing(self):
        runtime = self._runtime({"data": []})
        runtime._adopt_served_model_name()
        self.assertEqual(runtime.model_name, "test/test-moe")

    def test_a_failing_listing_changes_nothing(self):
        runtime = VLLMRuntime()
        plan = real()
        runtime.spec = runtime.engines.get("vllm")
        runtime.model_name = plan.source_id
        runtime.base_url = "http://127.0.0.1:8000"

        def boom(path, timeout=10.0):
            raise OSError("unreachable")

        runtime._get = boom
        runtime._adopt_served_model_name()
        self.assertEqual(runtime.model_name, "test/test-moe")


class MeasurementMathTests(unittest.TestCase):
    def test_per_stream_rate_excludes_the_first_token(self):
        """The first token is time-to-first-token, not decode.

        Counting it in the decode rate would blend prefill into a number
        that is supposed to describe generation.
        """
        sample = StreamSample(ttft_ms=100.0, total_ms=1100.0, output_tokens=101, ok=True)
        self.assertAlmostEqual(sample.decode_ms, 1000.0, places=6)
        self.assertAlmostEqual(sample.tokens_per_second, 100.0, places=6)

    def test_a_single_token_response_has_no_decode_rate(self):
        sample = StreamSample(ttft_ms=50.0, total_ms=60.0, output_tokens=1, ok=True)
        self.assertIsNone(sample.tokens_per_second)

    def test_percentiles_handle_small_samples(self):
        self.assertEqual(_percentile([5.0], 0.95), 5.0)
        self.assertEqual(_percentile([1.0, 2.0, 3.0, 4.0], 0.0), 1.0)
        self.assertEqual(_percentile([1.0, 2.0, 3.0, 4.0], 1.0), 4.0)
        self.assertIsNone(_percentile([], 0.95))


if __name__ == "__main__":
    unittest.main()
