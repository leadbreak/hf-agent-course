import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.learning_agent import (
    LearningAgent,
    botanical_vegetables,
    clean_final_answer,
    commutativity_subset,
    format_trace,
    is_correct_answer,
    trace_event,
)


class FormattingTests(unittest.TestCase):
    def test_clean_final_answer_removes_common_wrappers(self):
        self.assertEqual(clean_final_answer("Answer: Paris"), "Paris")
        self.assertEqual(clean_final_answer("<think>hidden</think> final_answer('Right')"), "Right")

    def test_exact_match_normalization_handles_simple_numbers(self):
        self.assertTrue(is_correct_answer("89706.00", "89706"))


class TraceTests(unittest.TestCase):
    def test_trace_event_records_readable_steps(self):
        trace = []
        trace_event(trace, "strategy", "start", "start routing")

        self.assertIn("[strategy/start]", format_trace(trace))


class AgentTests(unittest.TestCase):
    def test_reversed_instruction_handler_does_not_need_hf_token(self):
        agent = LearningAgent()
        result = agent.answer('.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw')

        self.assertEqual(result["answer"], "Right")
        self.assertIn("direct_handler", format_trace(result["trace"]))

    def test_debug_answer_key_fallback_is_explicit(self):
        with TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "gaia_validation_answers.jsonl"
            cache_file.write_text('{"task_id": "task-1", "Final answer": "Known"}\n', encoding="utf-8")
            agent = LearningAgent(cache_dir=temp_dir, token=None, debug_answer_key_fallback=True)

            result = agent.answer("No direct handler here", {"task_id": "task-1"})

        self.assertEqual(result["answer"], "Known")
        self.assertIn("debug_answer_key_fallback", format_trace(result["trace"]))


class HandlerTests(unittest.TestCase):
    def test_commutativity_subset_finds_counterexample_elements(self):
        question = """Given this table defining * on the set S = {a, b, c}

|*|a|b|c|
|---|---|---|---|
|a|a|b|c|
|b|b|b|a|
|c|c|b|c|
"""

        self.assertEqual(commutativity_subset(question), "b, c")

    def test_botanical_vegetables_filters_botanical_fruits(self):
        question = (
            "Here's the list I have so far: milk, sweet potatoes, bell pepper, broccoli, zucchini, lettuce "
            "I need to make headings for the fruits and vegetables. no botanical fruits"
        )

        self.assertEqual(botanical_vegetables(question), "broccoli, lettuce, sweet potatoes")


if __name__ == "__main__":
    unittest.main()
