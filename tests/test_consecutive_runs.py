"""
tests/test_consecutive_runs.py
Stress test executing multiple consecutive 50-step simulation runs via WebSockets.
Monitors memory usage (RSS MB), step completion, and stability across runs.
"""

import os
import sys
import unittest
import psutil
from starlette.testclient import TestClient

sys.path.insert(0, ".")
from backend.main import app


class TestConsecutiveRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.proc = psutil.Process(os.getpid())

    def test_consecutive_websocket_runs(self):
        num_runs = 10
        seeds = [123, 42, 999]
        memory_history = []

        print(f"\nStarting {num_runs} consecutive 50-step simulation stress runs...")

        for i in range(num_runs):
            seed = seeds[i % len(seeds)]
            initial_mem = self.proc.memory_info().rss / (1024 * 1024)

            with self.client.websocket_connect(f"/api/ws/simulate?seed={seed}&mode=replay") as ws:
                init = ws.receive_json()
                self.assertEqual(init["type"], "init")
                
                step_count = 0
                while True:
                    frame = ws.receive_json()
                    ftype = frame.get("type")
                    if ftype == "step":
                        step_count += 1
                    elif ftype == "complete":
                        self.assertIn("metrics", frame)
                        self.assertIn("telemetry", frame)
                        break

                self.assertEqual(step_count, 51)

            final_mem = self.proc.memory_info().rss / (1024 * 1024)
            memory_history.append(final_mem)
            print(f"  Run {i+1:02d}/{num_runs} (Seed {seed}): 51 steps complete | Mem: {final_mem:.2f} MB (Delta: {final_mem - initial_mem:+.2f} MB)")

        # Verify memory stability: memory delta between run 1 and run N should be bounded
        mem_growth = memory_history[-1] - memory_history[0]
        print(f"\nStress Test Complete: Initial Mem = {memory_history[0]:.2f} MB, Final Mem = {memory_history[-1]:.2f} MB, Net Delta = {mem_growth:+.2f} MB")
        
        # Verify memory stability: net memory growth across N runs should not exceed 50 MB
        self.assertLess(mem_growth, 50.0, f"Potential memory leak detected: net growth of {mem_growth:.2f} MB across {num_runs} runs")


if __name__ == "__main__":
    unittest.main()
