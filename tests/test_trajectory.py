"""Tests for TrajectoryTracker."""

import sys
import os
import json
import time
import tempfile
import importlib.util
import unittest

# Direct import to avoid triggering the full package __init__.py chain
_mod_path = os.path.join(os.path.dirname(__file__), "..", "code", "utils", "trajectory.py")
_spec = importlib.util.spec_from_file_location("trajectory", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TrajectoryTracker = _mod.TrajectoryTracker
TrajectoryStep = _mod.TrajectoryStep


class TestTrajectoryStep(unittest.TestCase):
    """Test TrajectoryStep dataclass."""

    def test_creation(self):
        step = TrajectoryStep(step_type="thought", content="test", step_number=1)
        self.assertEqual(step.step_type, "thought")
        self.assertEqual(step.content, "test")
        self.assertEqual(step.step_number, 1)

    def test_to_dict(self):
        step = TrajectoryStep(step_type="action", content="do", step_number=2, duration_ms=10.5)
        d = step.to_dict()
        self.assertEqual(d["step_type"], "action")
        self.assertEqual(d["duration_ms"], 10.5)

    def test_none_removed_from_dict(self):
        step = TrajectoryStep(step_type="thought", content="x", step_number=1)
        d = step.to_dict()
        self.assertNotIn("duration_ms", d)


class TestTrajectoryTracker(unittest.TestCase):
    """Test TrajectoryTracker functionality."""

    def setUp(self):
        self.tracker = TrajectoryTracker(agent_name="TestBot", task="Fix bug")

    def test_add_step(self):
        self.tracker.add_step("thought", "analyzing")
        self.assertEqual(len(self.tracker.steps), 1)
        self.assertEqual(self.tracker.steps[0].step_type, "thought")
        self.assertEqual(self.tracker.steps[0].step_number, 1)

    def test_auto_start(self):
        self.assertIsNone(self.tracker.start_time)
        self.tracker.add_step("thought", "go")
        self.assertIsNotNone(self.tracker.start_time)

    def test_reset(self):
        self.tracker.add_step("thought", "x")
        self.tracker.add_step("action", "y")
        self.tracker.reset()
        self.assertEqual(len(self.tracker.steps), 0)
        self.assertIsNone(self.tracker.start_time)

    def test_timer(self):
        self.tracker.start_timer()
        time.sleep(0.01)
        ms = self.tracker.stop_timer()
        self.assertGreater(ms, 5)  # at least 5ms

    def test_timer_without_start(self):
        ms = self.tracker.stop_timer()
        self.assertEqual(ms, 0.0)

    def test_token_usage(self):
        self.tracker.add_token_usage(100, 50)
        self.tracker.add_token_usage(200, 100)
        self.assertEqual(self.tracker.total_input_tokens, 300)
        self.assertEqual(self.tracker.total_output_tokens, 150)

    def test_get_stats(self):
        self.tracker.start()
        self.tracker.add_step("thought", "a")
        self.tracker.add_step("action", "b")
        self.tracker.add_step("thought", "c")
        self.tracker.add_token_usage(500, 200)
        self.tracker.end()

        stats = self.tracker.get_stats()
        self.assertEqual(stats["total_steps"], 3)
        self.assertEqual(stats["step_counts"]["thought"], 2)
        self.assertEqual(stats["step_counts"]["action"], 1)
        self.assertEqual(stats["total_tokens"], 700)
        self.assertIsNotNone(stats["elapsed_seconds"])

    def test_to_text(self):
        self.tracker.add_step("thought", "think hard")
        self.tracker.add_step("final_answer", "done")
        text = self.tracker.to_text()
        self.assertIn("Thought", text)
        self.assertIn("Final Answer", text)
        self.assertIn("think hard", text)

    def test_to_markdown(self):
        self.tracker.add_step("thought", "analyzing")
        md = self.tracker.to_markdown()
        self.assertIn("# Agent Trajectory", md)
        self.assertIn("analyzing", md)

    def test_to_json(self):
        self.tracker.add_step("thought", "go")
        j = self.tracker.to_json()
        data = json.loads(j)
        self.assertEqual(data["agent_name"], "TestBot")
        self.assertEqual(len(data["steps"]), 1)

    def test_to_dict(self):
        self.tracker.add_step("thought", "go")
        d = self.tracker.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("steps", d)

    def test_save_json(self):
        self.tracker.add_step("thought", "save test")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.tracker.save(path, "json")
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["agent_name"], "TestBot")
        finally:
            os.unlink(path)

    def test_save_markdown(self):
        self.tracker.add_step("thought", "md test")
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            self.tracker.save(path, "markdown")
            with open(path) as f:
                content = f.read()
            self.assertIn("# Agent Trajectory", content)
        finally:
            os.unlink(path)

    def test_save_text(self):
        self.tracker.add_step("thought", "txt test")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            self.tracker.save(path, "text")
            with open(path) as f:
                content = f.read()
            self.assertIn("Thought", content)
        finally:
            os.unlink(path)

    def test_save_invalid_format(self):
        with self.assertRaises(ValueError):
            self.tracker.save("/tmp/test.xyz", "xml")


if __name__ == "__main__":
    unittest.main()
