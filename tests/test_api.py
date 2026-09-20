import unittest
from importlib.util import find_spec


API_TEST_AVAILABLE = all(
    find_spec(name) is not None for name in ("fastapi", "httpx")
)


@unittest.skipUnless(
    API_TEST_AVAILABLE,
    "Install the api extra to run the API contract test",
)
class ApiContractTests(unittest.TestCase):
    def test_health_models_and_chat_completion(self):
        from fastapi.testclient import TestClient

        from localpilot.api.server import create_app

        client = TestClient(create_app())
        health = client.get("/health")
        models = client.get("/v1/models")
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "localpilot-best",
                "messages": [
                    {
                        "role": "user",
                        "content": "Review shared counter access",
                    }
                ],
                "stream": False,
            },
        )

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertTrue(health.json()["simulated"])
        self.assertEqual(models.status_code, 200)
        self.assertGreaterEqual(len(models.json()["data"]), 1)
        self.assertEqual(chat.status_code, 200)
        self.assertTrue(chat.json()["localpilot"]["simulated"])


if __name__ == "__main__":
    unittest.main()

