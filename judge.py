import logging
import openai_utils
import os
import pprint
import re

from chat_client import ChatClient
from logger_utils import maybe_log
from string import Template
from typing import List

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
JUDGE_SYS_PROMPT_FP = os.path.join(__location__, "prompts/judge_sys_prompt.txt")
PRIVATE_JUDGE_PROMPT_FP = os.path.join(__location__, "prompts/private_judge_prompt.txt")

NAME_1 = "Debater A"
NAME_2 = "Debater B"


def judge_rules(word_limit: int, quote_limit: int):
    with open(JUDGE_SYS_PROMPT_FP) as f:
        judge_rules_template = Template(f.read())
    return judge_rules_template.substitute(
        word_limit=word_limit, quote_limit=quote_limit
    )


def private_judge_prompt(max_rounds=5):
    with open(PRIVATE_JUDGE_PROMPT_FP) as f:
        private_judge_template = Template(f.read())
    return private_judge_template.substitute(max_rounds=max_rounds)


def debate_few_shot_examples(num=2):

    with open(
        f"{__location__}/few-shot-examples/judge/how-to-make-friends-1.txt", "r"
    ) as f:
        example1 = f.read()

    with open(
        f"{__location__}/few-shot-examples/judge/call-him-nemesis-0.txt", "r"
    ) as f:
        example2 = f.read()

    instr = "\n\nHere are a couple examples of Debates. We are mostly providing these examples so you can understand how the game works, how often (and when) to include quotes, what your writing style should look like, how the debaters respond to the judge's questions and feedback, and so on. Do not reference these stories, arguments, quotes, etc. in your own upcoming Debate. In particular, notice how the debaters engage very directly with the Judge's comments. Instead of repeating their supported answer choice, they engage actively with the Judge's questions and requests.\n\n\n"
    examples = [
        f"Example {i+1}:\n```\n{ex}\n```\n"
        for i, ex in enumerate([example1, example2])
        if i < num
    ]
    examples = "\n".join(examples)
    return f"{instr}{examples}"


class Judge:
    def __init__(
        self,
        question: str,
        answers: List[str],
        client: ChatClient,
        max_rounds: int = 5,
        temperature: float = 0.7,
        logger: logging.Logger = None,
        max_format_retries: int = 3,
        num_few_shot_examples: int = 2,
    ):
        self.question = question
        self.answers = answers
        self.client = client
        self.logger = logger
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_format_retries = max_format_retries
        self.num_few_shot_examples = num_few_shot_examples

    def prepare_transcript(self, history):
        separator = "\n---------------------------------------------\n"
        opening_prompt = f"""
Here is the current transcript of the debate:

Question: {self.question}

{NAME_1}: I claim the answer is {self.answers[0]}
{NAME_2}: I claim the answer is {self.answers[1]}{separator}\n"""
        history_str = separator.join(
            [f"{name}: {argument}" for name, argument in history]
        )
        transcript = opening_prompt + history_str + separator + f"Judge: "
        return transcript

    def verify_format(self, response):
        error_msg = ""
        is_correct = True
        if not response.startswith("<CONTINUE>") and not response.startswith(
            "<VOTING>"
        ):
            is_correct = False
            error_msg = f'You just tried to respond by saying:\n```\n{response}\n```\nbut your response was misformatted. Your response must start with "<CONTINUE>" or "<VOTING>". Please restate your response, completing the next turn of debate as your role of Judge.'
        elif re.search("Probabilities: \[0\.[0-9]+\,.*0\.[0-9]+\]$", response) is None:
            is_correct = False
            error_msg = f'You just tried to respond by saying:\n```\n{response}\n```\nbut your response was misformatted. Your response must end with the probabilities that you assign to each answer being the correct answer, in the format "Probabilities: [0.3, 0.7]". Please restate your response, completing the next turn of debate as your role of Judge.'
        return is_correct, error_msg

    def run_single_turn(
        self,
        history,
        char_limit: int,
        quote_char_limit: int,
        is_last_round: bool = False,
    ):
        transcript = self.prepare_transcript(history)

        word_limit = char_limit / 5
        quote_limit = quote_char_limit / 5
        rules = judge_rules(word_limit, quote_limit)
        few_shot_examples = debate_few_shot_examples(num=self.num_few_shot_examples)
        self.private = private_judge_prompt(max_rounds=self.max_rounds)
        if self.num_few_shot_examples > 0:
            system_prompt = "\n".join([rules, few_shot_examples, self.private])
        else:
            system_prompt = "\n".join([rules, self.private])

        ending_prompt = f"Complete the next turn of debate as your role of Judge."
        format_correct = True
        error_msg = ""
        num_tries = 0
        while True:
            if error_msg:
                ending_prompt = error_msg
            if num_tries >= self.max_format_retries:
                maybe_log(
                    self.logger,
                    f"Judge exceeded {self.max_format_retries} retries for formatting response correctly. Exiting.",
                    level="error",
                )
                break
            if is_last_round:
                ending_prompt = f'{ending_prompt} Keep in mind that this is the last round of the debate so your response must start with "<VOTING>" and you must decide which debater\'s argument was more convincing to you.'
            msgs = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": transcript,
                },
                {
                    "role": "user",
                    "content": ending_prompt,
                },
            ]
            maybe_log(
                self.logger,
                f"Messages given to Judge:\n{pprint.pformat(msgs, indent=4)}",
                level="info",
            )
            output = self.client.chat_single_turn(
                msgs,
                max_output_length=char_limit,
                temperature=self.temperature,
                return_usage=True,
            )
            response = output["output_strs"][0]
            maybe_log(
                self.logger,
                f"Usage of {self.client.model}: {output['usage']}",
                level="info",
            )
            format_correct, error_msg = self.verify_format(response)
            num_tries += 1
            if format_correct:
                break
            maybe_log(
                self.logger,
                f"Judge did not format response correctly: {response}",
                level="warning",
            )
        return response
