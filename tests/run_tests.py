import sys
import unittest
from starlette.testclient import TestClient

sys.path.insert(0, ".")
from backend.main import app


class TestCrisisGridAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)

    def test_02_seeds_endpoint(self):
        res = self.client.get("/api/seeds")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("seeds", data)
        self.assertIn(123, data["seeds"])

    def test_03_replay_endpoint(self):
        res = self.client.get("/api/replay?seed=123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["seed"], 123)
        self.assertEqual(len(data["steps"]), 51)

    def test_04_comparison_endpoint(self):
        res = self.client.get("/api/comparison?seed=123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("trained", data)
        self.assertIn("random", data)
        self.assertIn("comparison", data)
        comp = data["comparison"]
        self.assertFalse(comp["policies_match"])
        self.assertIn("decision_similarity", comp)

    def test_05_simulate_endpoint(self):
        res = self.client.post("/api/simulate", json={"seed": 123, "mode": "replay"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["seed"], 123)


    def test_06_websocket_stream(self):
        with self.client.websocket_connect("/api/ws/simulate?seed=123&mode=replay") as ws:
            init = ws.receive_json()
            self.assertEqual(init["type"], "init")
            step_count = 0
            while True:
                frame = ws.receive_json()
                if frame.get("type") == "step":
                    step_count += 1
                elif frame.get("type") == "complete":
                    self.assertIn("telemetry", frame)
                    break
            self.assertEqual(step_count, 51)


if __name__ == "__main__":
    unittest.main()
