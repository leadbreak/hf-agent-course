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


if __name__ == "__main__":
    unittest.main()
