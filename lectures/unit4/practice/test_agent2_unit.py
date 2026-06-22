import unittest
from unittest.mock import patch

import agent2
import test as local_test


class Agent2FallbackTests(unittest.TestCase):
    def setUp(self):
        agent2._MODEL = None
        agent2._AGENT = None
        if hasattr(agent2, "_MODEL_LOAD_ERROR"):
            agent2._MODEL_LOAD_ERROR = None

    def test_model_load_failure_is_not_retried_for_every_question(self):
        with patch("agent2.TransformersModel", side_effect=ValueError("boom")) as model:
            self.assertIsNone(agent2._ask_plain_llm("Question 1"))
            self.assertIsNone(agent2._ask_plain_llm("Question 2"))

        self.assertEqual(model.call_count, 1)

    def test_predict_writes_trace_file_for_direct_answer(self):
        question = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
        trace_path = agent2._trace_path(question)
        if trace_path.exists():
            trace_path.unlink()

        with patch.dict("os.environ", {"AGENT2_DISABLE_ANSWER_CACHE": "1"}):
            self.assertEqual(agent2.predict(question), "Right")

        trace = agent2._load_json(trace_path, {})
        self.assertEqual(trace["answer"], "Right")
        stages = [event["stage"] for event in trace["events"]]
        self.assertIn("strategy", stages)
        self.assertIn("direct_handler", stages)
        self.assertIn("finalize", stages)


class TestRunnerHelpers(unittest.TestCase):
    def test_answer_comparison_normalizes_case_and_number_format(self):
        self.assertTrue(local_test.is_correct_answer("89706.00", "89706"))
        self.assertTrue(local_test.is_correct_answer(" right ", "Right"))
        self.assertFalse(local_test.is_correct_answer("unknown", "Right"))

    def test_submission_payload_uses_predicted_answers(self):
        rows = [
            {"Task ID": "task-1", "Predicted Answer": "A"},
            {"Task ID": "task-2", "Predicted Answer": "B"},
        ]

        payload = local_test.build_answers_payload(rows)

        self.assertEqual(
            payload,
            [
                {"task_id": "task-1", "submitted_answer": "A"},
                {"task_id": "task-2", "submitted_answer": "B"},
            ],
        )

    def test_trace_formatter_summarizes_events(self):
        trace = {
            "events": [
                {"stage": "strategy", "status": "start", "message": "Route by task type"},
                {"stage": "tool", "status": "success", "message": "Used web_search"},
            ]
        }

        rendered = local_test.format_trace(trace)

        self.assertIn("strategy", rendered)
        self.assertIn("Used web_search", rendered)


if __name__ == "__main__":
    unittest.main()
