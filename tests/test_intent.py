import unittest

from localpilot.intent.parser import parse_intent


class IntentParserTests(unittest.TestCase):
    def test_chinese_coding_latency_local_only(self):
        intent = parse_intent(
            "帮我部署一个完全本地运行的代码审查 AI，代码不能上传，响应速度优先"
        )
        self.assertEqual(intent.task, "coding")
        self.assertEqual(intent.privacy, "local_only")
        self.assertEqual(intent.priority, "latency")
        self.assertIn("code_review", intent.capabilities)
        self.assertEqual(intent.preferred_language, ["zh", "en"])

    def test_context_parsing(self):
        intent = parse_intent("local coding assistant with 16K context")
        self.assertEqual(intent.context_length, 16384)


if __name__ == "__main__":
    unittest.main()

