from chat_client import ChatClient
from common import RandomnessSource
from debater import Debater
from logging import Logger
from logger_utils import maybe_log
from judge import Judge
from typing import Any, Dict, List, Tuple

import random

NAME_1 = "Debater A"
NAME_2 = "Debater B"


class Debate:
    def __init__(
        self,
        story: str,
        question: str,
        correct_answer: str,
        best_distractor: str,
        char_limit: int,
        quote_char_limit: int,
        model: str = "gpt-4-1106-preview",
        max_rounds: int = 5,
        logger: Logger = None,
        debater_temperature: float = 0.7,
        judge_temperature: float = 0.7,
        num_few_shot_examples: int = 2,
        max_debater_length_retries: int = 3,
        max_judge_format_retries: int = 3,
        randomness_sources: List[RandomnessSource] = [],
        debater_to_assign_random_intervention_to: str = "random",  # one of ["random", "correct", "incorrect"]
        **randomness_kwargs,
    ):
        self.story = story
        self.question = question
        self.model = model
        self.logger = logger
        self.char_limit = char_limit
        self.quote_char_limit = quote_char_limit
        self.max_rounds = max_rounds
        self.randomness_sources = randomness_sources
        self.debater_to_assign_random_intervention_to = (
            debater_to_assign_random_intervention_to
        )
        self.randomness_kwargs = randomness_kwargs

        # Check that we have a valid combination of randomness sources
        if RandomnessSource.JUDGE_PROB in self.randomness_sources and RandomnessSource.JUDGE_PROB_SINGLE_DEBATER in self.randomness_sources:
            raise ValueError("Randomness sources cannot include both JUDGE_PROB and JUDGE_PROB_SINGLE_DEBATER.")
        if RandomnessSource.DECODING_STRAT_SINGLE_DEBATER in self.randomness_sources and RandomnessSource.DECODING_STRAT_BOTH in self.randomness_sources:
            raise ValueError("Randomness sources cannot include both DECODING_STRAT_SINGLE_DEBATER and DECODING_STRAT_BOTH.")

        if RandomnessSource.JUDGE_PROB in self.randomness_sources:
            assert "mean" in randomness_kwargs
            assert "std" in randomness_kwargs
        self.correct_debater_name = random.sample(["A", "B"], k=1)[0]
        self.distractor_debater_name = "A" if self.correct_debater_name == "B" else "B"
        maybe_log(
            logger,
            f"Debater {self.correct_debater_name} is arguing for the correct answer and debater "
            + f"{self.distractor_debater_name} is arguing for the best distractor.",
            "info",
        )
        self.chat_client = ChatClient(model=self.model, logger=self.logger)
        debater2answer = {
            "Debater A": correct_answer
            if self.correct_debater_name == "A"
            else best_distractor,
            "Debater B": correct_answer
            if self.correct_debater_name == "B"
            else best_distractor,
        }
        self.answers = [debater2answer[name] for name in ["Debater A", "Debater B"]]
        debater_temperatures = [debater_temperature, debater_temperature]
        if RandomnessSource.DECODING_STRAT_SINGLE_DEBATER in self.randomness_sources:
            # assign one debater to temperature sampling, and the other to greedy decoding
            if self.debater_to_assign_random_intervention_to == "correct":
                debater_temperatures = (
                    [debater_temperature, 0.0]
                    if self.correct_debater_name == "A"
                    else [0.0, debater_temperature]
                )
            elif self.debater_to_assign_random_intervention_to == "incorrect":
                debater_temperatures = (
                    [debater_temperature, 0.0]
                    if self.distractor_debater_name == "A"
                    else [0.0, debater_temperature]
                )
            else:
                # Randomly shuffle
                debater_temperatures = [
                    debater_temperature,
                    0.0,
                ]
                random.shuffle(debater_temperatures)
        elif RandomnessSource.DECODING_STRAT_BOTH in self.randomness_sources:
            debater_temperatures = [debater_temperature, debater_temperature]

        debater_adds_randomness = [False, False]
        if RandomnessSource.JUDGE_PROB_SINGLE_DEBATER in self.randomness_sources:
            # Decide which debater will add randomness to the judge rewards
            if self.debater_to_assign_random_intervention_to == "correct":
                debater_adds_randomness = (
                    [True, False] if self.correct_debater_name == "A" else [False, True]
                )
            elif self.debater_to_assign_random_intervention_to == "incorrect":
                debater_adds_randomness = (
                    [True, False]
                    if self.distractor_debater_name == "A"
                    else [False, True]
                )
            else:
                # randomly shuffle
                debater_adds_randomness = [True, False]
                random.shuffle(debater_adds_randomness)

        self.debater_a = Debater(
            self.story,
            self.question,
            self.answers,
            self.chat_client,
            0,
            temperature=debater_temperatures[0],
            num_few_shot_examples=num_few_shot_examples,
            logger=self.logger,
            max_length_retries=max_debater_length_retries,
            randomness_sources=randomness_sources,
            debater_adds_randomness=debater_adds_randomness[0],
            **randomness_kwargs,
        )
        self.debater_b = Debater(
            self.story,
            self.question,
            self.answers,
            self.chat_client,
            1,
            temperature=debater_temperatures[1],
            num_few_shot_examples=num_few_shot_examples,
            logger=self.logger,
            max_length_retries=max_debater_length_retries,
            randomness_sources=randomness_sources,
            debater_adds_randomness=debater_adds_randomness[1],
            **randomness_kwargs,
        )
        self.judge = Judge(
            self.question,
            self.answers,
            self.chat_client,
            max_rounds=self.max_rounds,
            temperature=judge_temperature,
            logger=self.logger,
            max_format_retries=max_judge_format_retries,
            num_few_shot_examples=num_few_shot_examples,
        )

    def run_debate(self) -> List[Tuple[str, str]]:
        num_rounds = 0
        history: List[Tuple[str, str]] = []  # pairs of (name, response)
        while num_rounds < self.max_rounds:
            order = (
                [self.debater_a, self.debater_b]
                if num_rounds % 2 == 0
                else [self.debater_b, self.debater_a]
            )
            for deb in order:
                resp = deb.run_single_turn(
                    history, self.char_limit, self.quote_char_limit
                )
                history.append((deb.name, resp))
            judge_resp = self.judge.run_single_turn(
                history,
                self.char_limit,
                self.quote_char_limit,
                is_last_round=num_rounds == (self.max_rounds - 1),
            )
            history.append(("Judge", judge_resp))
            if judge_resp.startswith("<VOTING>"):
                break
            num_rounds += 1
        self.history = history
        return self.history

    def load_history_from_dict(
        self, debate_dict: Dict[str, Any], passages_list: List[Dict[str, Any]]
    ):
        """
        Loads debate history from past data.
        Note: Does NOT re-initialize the debaters or judge.
        """
        self.history = [tuple(nested_list) for nested_list in debate_dict["history"]]
        self.correct_answer = debate_dict["correct_answer_text"]
        self.best_distractor = debate_dict["best_distractor_text"]
        self.question = debate_dict["question"]
        story = [
            row
            for row in passages_list
            if row["article_id"] == debate_dict["article_id"]
        ]
        if len(story) == 0:
            raise ValueError(
                f"Could not find article_id {debate_dict['article_id']} in passages_list."
            )
        self.story = story[0]["article"]
        self.answers = debate_dict["answers"]
        self.correct_debater_name = debate_dict["correct_debater"]
        self.distractor_debater_name = debate_dict["distractor_debater"]

    def prepare_transcript(self) -> str:
        separator = "\n---------------------------------------------\n"
        opening_prompt = f"""
Context:

{self.story}

Question: {self.question}

{NAME_1}: I claim the answer is {self.answers[0]}
{NAME_2}: I claim the answer is {self.answers[1]}{separator}\n"""
        history_str = separator.join(
            [f"{name}: {argument}" for name, argument in self.history]
        )
        transcript = opening_prompt + history_str
        return transcript
