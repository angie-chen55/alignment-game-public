from chat_client import ChatClient
from debate import Debate
from debater import Debater
from judge import Judge
from unittest.mock import MagicMock
import logging
import unittest


class TestDebate(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level="INFO")
        self.logger = logging.getLogger(name="debate_test")
        self.story = "Fake story"
        self.question = "Fake question"
        self.answers = ["Choice 1", "Choice 2"]
        self.correct_answer = "Correct"
        self.best_distractor = "Distractor"
        self.temperature = 0.8
        self.history = []
        self.char_limit = 500

    def testDebateSingleTurn(self):
        debate = Debate(
            self.story,
            self.question,
            self.correct_answer,
            self.best_distractor,
            self.char_limit,
            self.char_limit,
            max_rounds=5,
            logger=self.logger,
        )
        debater_a_response = "Debater A first response"
        debater_b_response = "Debater B first response"
        judge_response = "<VOTING> Probabilities: [0.9, 0.1]"
        expected_history = [
            ("Debater A", debater_a_response),
            ("Debater B", debater_b_response),
            ("Judge", judge_response),
        ]
        debate.debater_a.run_single_turn = MagicMock(return_value=debater_a_response)
        debate.debater_b.run_single_turn = MagicMock(return_value=debater_b_response)
        debate.judge.run_single_turn = MagicMock(return_value=judge_response)
        history = debate.run_debate()
        assert expected_history == history

    def testDebateMultipleTurns(self):
        debate = Debate(
            self.story,
            self.question,
            self.correct_answer,
            self.best_distractor,
            self.char_limit,
            self.char_limit,
            max_rounds=5,
            logger=self.logger,
        )
        debater_a_response = "Debater A response"
        debater_b_response = "Debater B response"
        judge_first_response = "<CONTINUE> Probabilities: [0.6, 0.4]"
        judge_next_response = "<VOTING> Probabilities: [0.9, 0.1]"
        expected_history = [
            ("Debater A", debater_a_response),
            ("Debater B", debater_b_response),
            ("Judge", judge_first_response),
            ("Debater B", debater_b_response),
            ("Debater A", debater_a_response),
            ("Judge", judge_next_response),
        ]
        debate.debater_a.run_single_turn = MagicMock(return_value=debater_a_response)
        debate.debater_b.run_single_turn = MagicMock(return_value=debater_b_response)
        debate.judge.run_single_turn = MagicMock(
            side_effect=[judge_first_response, judge_next_response]
        )
        history = debate.run_debate()
        assert expected_history == history

    def testDebateTooManyTurns(self):
        debate = Debate(
            self.story,
            self.question,
            self.correct_answer,
            self.best_distractor,
            self.char_limit,
            self.char_limit,
            max_rounds=1,
            logger=self.logger,
        )
        debater_a_response = "Debater A response"
        debater_b_response = "Debater B response"
        judge_first_response = "<CONTINUE> Probabilities: [0.6, 0.4]"
        expected_history = [
            ("Debater A", debater_a_response),
            ("Debater B", debater_b_response),
            ("Judge", judge_first_response),
        ]
        debate.debater_a.run_single_turn = MagicMock(return_value=debater_a_response)
        debate.debater_b.run_single_turn = MagicMock(return_value=debater_b_response)
        debate.judge.run_single_turn = MagicMock(return_value=judge_first_response)
        history = debate.run_debate()
        assert expected_history == history


if __name__ == "__main__":
    unittest.main()
