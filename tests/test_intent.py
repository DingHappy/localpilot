import unittest

from localpilot.intent.parser import parse_intent


class TaskTests(unittest.TestCase):
    def test_chinese_coding_latency_local_only(self):
        intent = parse_intent(
            "帮我部署一个完全本地运行的代码审查 AI，代码不能上传，响应速度优先"
        )
        self.assertEqual(intent.task, "coding")
        self.assertEqual(intent.privacy, "local_only")
        self.assertEqual(intent.priority, "latency")
        self.assertIn("code_review", intent.capabilities)
        self.assertEqual(intent.preferred_language, ["zh", "en"])

    def test_each_task_family_is_recognised(self):
        cases = {
            "本地代码审查": "coding",
            "本地智能体调用工具": "agentic",
            "本地图片理解": "vision",
            "本地语音助手": "audio",
            "本地向量检索": "embedding",
            "a local assistant": "chat",
        }
        for text, expected in cases.items():
            self.assertEqual(parse_intent(text).task, expected, text)

    def test_modalities_are_extracted_from_the_request(self):
        self.assertIn("image", parse_intent("本地图片理解").modalities)
        self.assertIn("audio", parse_intent("本地语音对话").modalities)
        self.assertEqual(parse_intent("local chat").modalities, ["text"])


class PriorityTests(unittest.TestCase):
    def test_priorities_are_recognised(self):
        cases = {
            "本地代码助手，低延迟": "latency",
            "本地助手，质量优先": "quality",
            "本地助手，省内存": "low_memory",
            "本地助手，长上下文": "long_context",
            "本地助手，高吞吐": "throughput",
            "a local assistant": "balanced",
        }
        for text, expected in cases.items():
            self.assertEqual(parse_intent(text).priority, expected, text)

    def test_a_stated_concurrency_outranks_an_adjective(self):
        """A number is a stronger statement of intent than a word.

        Asking for low latency while also asking to serve 20 people is a
        throughput problem, whatever the adjective says.
        """
        intent = parse_intent("本地助手，响应速度优先，20 个用户同时用")
        self.assertEqual(intent.concurrency, 20)
        self.assertEqual(intent.priority, "throughput")

    def test_a_small_concurrency_leaves_the_priority_alone(self):
        intent = parse_intent("本地助手，响应速度优先，2 个用户")
        self.assertEqual(intent.concurrency, 2)
        self.assertEqual(intent.priority, "latency")


class ContextTests(unittest.TestCase):
    def test_context_units_are_parsed(self):
        self.assertEqual(
            parse_intent("local coding assistant with 16K context").context_length,
            16384,
        )
        self.assertEqual(parse_intent("需要 1M 上下文").context_length, 1048576)

    def test_a_long_context_request_implies_the_capability(self):
        intent = parse_intent("我要能读整个仓库的长上下文本地模型")
        self.assertGreaterEqual(intent.context_length, 131072)
        self.assertIn("long_context", intent.capabilities)

    def test_privacy_defaults_to_preferred_when_unstated(self):
        self.assertEqual(parse_intent("a coding assistant").privacy, "local_preferred")
        self.assertEqual(parse_intent("不能出网的助手").privacy, "local_only")


if __name__ == "__main__":
    unittest.main()
