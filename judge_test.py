from chat_client import ChatClient
from judge import Judge
from unittest.mock import MagicMock
import logging
import unittest


class TestJudge(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level="INFO")
        self.logger = logging.getLogger(name="judge_test")
        self.question = "Fake question"
        self.answers = ["Choice 1", "Choice 2"]
        self.temperature = 0.8
        self.history = []
        self.char_limit = 500

    def test_judge_incorrect_format(self):
        client = ChatClient(logger=self.logger)
        mock_return = {
            "output_strs": ["Incorrectly formatted judge output."],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
        client.chat_single_turn = MagicMock(return_value=mock_return)
        num_retries = 2
        judge = Judge(
            question=self.question,
            answers=self.answers,
            client=client,
            temperature=self.temperature,
            logger=self.logger,
            max_format_retries=num_retries,
            num_few_shot_examples=0,
        )
        resp = judge.run_single_turn(self.history, self.char_limit, self.char_limit)
        assert num_retries == client.chat_single_turn.call_count
        assert resp == mock_return["output_strs"][0]


class TestVerifyFormat(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level="INFO")
        self.logger = logging.getLogger(name="judge_test")
        self.question = "Fake question"
        self.answers = ["Choice 1", "Choice 2"]
        self.temperature = 0.8
        self.history = []
        self.char_limit = 500
        self.client = ChatClient(logger=self.logger)
        self.num_retries = 1
        self.judge = Judge(
            question=self.question,
            answers=self.answers,
            client=self.client,
            temperature=self.temperature,
            logger=self.logger,
            max_format_retries=self.num_retries,
            num_few_shot_examples=0,
        )

    def test_probabilities_format(self):
        input_responses = [
            "<CONTINUE> Probs: [0.8, 0.2]",
            "<CONTINUE> Probabilities: [0.8, 0.2]",
            "<VOTING> Probabilities: [0.8,0.2]",
            "<VOTING> Probabilities: [0.18,]",
            "<CONTINUE> Probabilities: []",
            "<CONTINUE> Probabilities: (0.8,0.2)",
        ]

        expected_ans_and_prefixes = [
            (False, "You just tried to respond by saying"),
            (True, ""),
            (True, ""),
            (False, "You just tried to respond by saying"),
            (False, "You just tried to respond by saying"),
            (False, "You just tried to respond by saying"),
        ]

        for ir, (exp_ans, exp_msg_prefix) in zip(
            input_responses, expected_ans_and_prefixes
        ):
            out_ans, out_err_msg = self.judge.verify_format(ir)
            self.assertEqual(out_ans, exp_ans)
            self.assertTrue(out_err_msg.startswith(exp_msg_prefix))

    def test_start_format(self):
        input_responses = [
            "<CONTINUE> Probabilities: [0.8, 0.2]",
            "<VOTING> Probabilities: [0.8,0.2]",
            "<VOTING>Probabilities: [0.8,0.2]",
            "<> Probabilities: [0.8, 0.2]",
            "Probabilities: [0.8, 0.2]",
            "<VOTE> Probabilities: [0.8, 0.2]",
        ]

        expected_ans_and_prefixes = [
            (True, ""),
            (True, ""),
            (True, ""),
            (False, "You just tried to respond by saying"),
            (False, "You just tried to respond by saying"),
            (False, "You just tried to respond by saying"),
        ]

        for ir, (exp_ans, exp_msg_prefix) in zip(
            input_responses, expected_ans_and_prefixes
        ):
            out_ans, out_err_msg = self.judge.verify_format(ir)
            self.assertEqual(out_ans, exp_ans)
            self.assertTrue(out_err_msg.startswith(exp_msg_prefix))


if __name__ == "__main__":
    unittest.main()
