# Adapted from https://github.com/julianmichael/debate/blob/main/model-debate/debater.py
import json
import logging
import math
import os
import pprint
import random
import re
import time
from chat_client import ChatClient
from common import RandomnessSource
from logger_utils import maybe_log
from string import Template
from typing import Dict, List


NAME_1 = "Debater A"
NAME_2 = "Debater B"
NAMES = ["Debater A", "Debater B"]
SEPARATOR = "\n---------------------------------------------\n"

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
DEBATER_SYS_PROMPT_FP = os.path.join(__location__, "prompts/debater_sys_prompt.txt")
DEBATER_PRIVATE_PROMPT_FP = os.path.join(
    __location__, "prompts/private_debate_prompt.txt"
)


def debate_rules(word_limit: int, quote_limit: int):
    with open(DEBATER_SYS_PROMPT_FP) as f:
        debate_rules_template = Template(f.read())
    return debate_rules_template.substitute(
        word_limit=word_limit, quote_limit=quote_limit
    )


def private_debate_prompt(name: str, word_limit: int, quote_limit: int):
    with open(DEBATER_PRIVATE_PROMPT_FP) as f:
        private_debate_template = Template(f.read())
    return private_debate_template.substitute(
        name=name, word_limit=word_limit, quote_limit=quote_limit
    )


def debate_few_shot_examples(num=2):

    with open(
        f"{__location__}/few-shot-examples/debater/how-to-make-friends-1.txt", "r"
    ) as f:
        example1 = f.read()

    with open(
        f"{__location__}/few-shot-examples/debater/call-him-nemesis-0.txt", "r"
    ) as f:
        example2 = f.read()

    instr = "\n\nHere are a couple examples of Debates. We are mostly providing these examples so you can understand how the game works, how often (and when) to include quotes, what your writing style should look like, how to respond to the judge's questions and feedback, and so on. Do not reference these stories, arguments, quotes, etc. in your own upcoming Debate. In particular, notice how the debaters engage very directly with the Judge's comments. Instead of repeating their supported answer choice, they engage actively with the Judge's questions and requests.\n\n\n"
    examples = [
        f"Example {i+1}:\n```\n{ex}\n```\n"
        for i, ex in enumerate([example1, example2])
        if i < num
    ]
    examples = "\n".join(examples)
    return f"{instr}{examples}"


class Debater:
    def __init__(
        self,
        story: str,
        question: str,
        answers: List[str],
        client: ChatClient,
        debater_idx: int,
        temperature: float = 0.7,
        num_few_shot_examples: int = 2,
        logger: logging.Logger = None,
        max_length_retries: int = 3,
        randomness_sources: List[RandomnessSource] = [],
        debater_adds_randomness: bool = False,
        **randomness_kwargs,
    ):
        super().__init__()
        self.story = story
        self.question = question
        self.answers = answers
        self.client = client
        self.debater_idx = debater_idx
        self.name = NAMES[debater_idx]
        self.results = []
        self.temperature = temperature
        self.num_few_shot_examples = num_few_shot_examples
        self.logger = logger
        self.max_length_retries = max_length_retries
        self.randomness_sources = randomness_sources
        self.randomness_kwargs = randomness_kwargs
        self.debater_adds_randomness = debater_adds_randomness
        if RandomnessSource.JUDGE_PROB in self.randomness_sources:
            assert "mean" in randomness_kwargs
            assert "std" in randomness_kwargs

    def prepare_transcript(self, history):
        opening_prompt = f"""
Here is the current transcript of the debate:

Context:

{self.story}

Question: {self.question}

{NAME_1}: I claim the answer is {self.answers[0]}
{NAME_2}: I claim the answer is {self.answers[1]}{SEPARATOR}\n"""
        history_str = SEPARATOR.join(
            [f"{name}: {argument}" for name, argument in history]
        )
        transcript = opening_prompt + history_str + SEPARATOR + f"{self.name}: "
        return transcript

    def add_randomness_to_last_judge_reward(self, transcript):
        last_judge_response_idx = transcript.rfind("Judge:")
        if last_judge_response_idx == -1:
            # no previous Judge response
            return transcript
        prefix = transcript[:last_judge_response_idx]
        last_judge_response = transcript[last_judge_response_idx:]
        # There might also be another debater response after the last judge response, so we need to find that too
        postfix = ""
        search_p = f"{SEPARATOR}Debater [AB]+:"
        m = re.search(search_p, last_judge_response, re.MULTILINE)
        if m is not None:
            idx = last_judge_response.find(m.group(0))
            postfix = last_judge_response[idx:]
            last_judge_response = last_judge_response[:idx]
        try:
            prob_str = last_judge_response[
                last_judge_response.find("Probabilities:") + len("Probabilities:") :
            ].strip()
            judge_probs = json.loads(prob_str)
        except json.decoder.JSONDecodeError as e:
            maybe_log(
                self.logger,
                f"Could not decode {prob_str}<END>. Will just return original transcript instead of adding randomness.",
                level="error",
            )
            return transcript
        judge_probs = [float(p) for p in judge_probs]
        p_new = judge_probs[0] + random.normalvariate(
            mu=self.randomness_kwargs["mean"], sigma=self.randomness_kwargs["std"]
        )
        # clip to [0.0, 1.0]
        p_new = min(1.0, max(0.0, p_new))
        other_p_new = 1.0 - p_new
        new_judge_response = (
            last_judge_response[: last_judge_response.find("Probabilities:")]
            + f"Probabilities: [{p_new:.1f}, {other_p_new:.1f}]"
        )
        new_transcript = prefix + new_judge_response + postfix
        maybe_log(
            self.logger,
            f"Created new randomized judge response:\n{new_judge_response}\n",
            level="info",
        )
        return new_transcript

    def check_output_length(self, output: str, char_limit: int, quote_char_limit: int):
        num_output_chars = len(output)
        pattern = r"<quote>(.*?)</quote>"
        try:
            matches = re.findall(pattern, output, re.DOTALL)
        except Exception as e:
            logging.error(f"pattern: {pattern}\n\noutput:{output}")
            raise e
        num_quote_chars = sum([len(match) for match in matches])
        if num_output_chars > char_limit:
            return "total", num_output_chars, num_quote_chars
        # don't bother with quote limit. more quotes good
        # elif num_quote_chars > quote_char_limit:
        #     return "quote", num_output_chars, num_quote_chars
        return "ok", num_output_chars, num_quote_chars

    def _truncate_story_within_transcript(
        self, transcript: str, max_output_length: int
    ) -> str:
        context_idx = transcript.find("Context:")
        question_idx = transcript.find("Question:")
        if context_idx == -1:
            logging.warning(f'Could not find "Context:" in string. Not truncating.')
            return transcript
        if question_idx == -1:
            logging.warning(f'Could not find "Question:" in string. Not truncating.')
            return transcript
        story_start_idx = context_idx + len("Context:\n\n")
        story_end_idx = question_idx - 2  # subtract 2 for newlines
        orig_story = transcript[story_start_idx:story_end_idx]
        max_story_tokens = (
            max_output_length
            - self.client.count_num_tokens(transcript[:story_start_idx])
            - self.client.count_num_tokens(transcript[story_end_idx:])
        )
        truncated_story = self.client.truncate_text_to_length(
            orig_story, max_story_tokens
        )
        new_transcript = (
            transcript[:story_start_idx] + truncated_story + transcript[story_end_idx:]
        )
        return new_transcript

    def _truncate_messages(
        self, msgs: List[Dict[str, str]], max_output_length: int
    ) -> List[Dict[str, str]]:
        # max_output_length is the length to truncate to
        # First try to truncate stories in few-shot examples in the system prompt
        if self.num_few_shot_examples == 1:
            ## Count the number of tokens in the remaining messages
            num_other_tokens = self.client.count_num_tokens_in_messages(msgs[1:])
            remaining_tokens = (
                max_output_length
                - num_other_tokens
                - 10
                - self.client.count_num_tokens_in_messages(
                    [{"role": "system", "content": ""}]
                )
            )  ## subtract 10 for some wiggle room
            truncated_sys_prompt = self._truncate_story_within_transcript(
                msgs[0]["content"], remaining_tokens
            )
            msgs = [{"role": "system", "content": truncated_sys_prompt}] + msgs[1:]
        elif self.num_few_shot_examples == 2:
            # Split the number of tokens truncated between the two few-shot examples
            num_other_tokens = self.client.count_num_tokens_in_messages(msgs[1:])
            orig_sys_prompt = msgs[0]["content"]
            ex_label_start_idxs = [
                orig_sys_prompt.find(f"Example {i+1}:") for i in range(2)
            ]
            ex_start_idxs = [
                start_idx + len(f"Example {i+1}:\n")
                for i, start_idx in enumerate(ex_label_start_idxs)
            ]
            ex_end_idxs = [ex_label_start_idxs[1] - 1, -1]
            examples = [
                orig_sys_prompt[start_idx:end_idx]
                for start_idx, end_idx in zip(ex_start_idxs, ex_end_idxs)
            ]
            remaining_tokens = (
                max_output_length
                - num_other_tokens
                - 10
                - self.client.count_num_tokens_in_messages(
                    [{"role": "system", "content": ""}]
                )
                - self.client.count_num_tokens(orig_sys_prompt[: ex_start_idxs[0]])
                - self.client.count_num_tokens(
                    orig_sys_prompt[ex_end_idxs[0] : ex_start_idxs[1]]
                )
            )  ## subtract 10 for some wiggle room
            logging.info(f"Remaining tokens: {remaining_tokens}")
            tokens_per_example = int(math.floor(remaining_tokens / 2))
            truncated_examples = [
                self._truncate_story_within_transcript(ex, tokens_per_example)
                for ex in examples
            ]
            new_sys_prompt = (
                orig_sys_prompt[: ex_start_idxs[0]]
                + truncated_examples[0]
                + orig_sys_prompt[ex_end_idxs[0] : ex_start_idxs[1]]
                + truncated_examples[1]
            )
            msgs = [{"role": "system", "content": new_sys_prompt}] + msgs[1:]
        elif self.num_few_shot_examples == 0:
            # Truncate current story instead
            curr_transcript = msgs[1]["content"]
            num_other_tokens = self.client.count_num_tokens_in_messages(
                [msgs[0]]
            ) + self.client.count_num_tokens_in_messages(msgs[2:])
            remaining_tokens = (
                max_output_length - num_other_tokens - 10
            )  ## subtract 10 for some wiggle room
            truncated_transcript = self._truncate_story_within_transcript(
                curr_transcript, remaining_tokens
            )
            msgs = [msgs[0], {"role": "user", "content": truncated_transcript}] + msgs[
                2:
            ]
        else:
            logging.warning(
                f"{self.num_few_shot_examples} is not in [0,2]. Not truncating messages."
            )

        return msgs

    def run_single_turn(self, history, char_limit: int, quote_char_limit: int):
        word_limit = char_limit / 5
        quote_limit = quote_char_limit / 5
        rules = debate_rules(word_limit, quote_limit)
        few_shot_examples = debate_few_shot_examples(num=self.num_few_shot_examples)
        self.private = private_debate_prompt(self.name, word_limit, quote_limit)
        self.position = (
            self.private
            + f"You argue that the answer is: '{self.answers[self.debater_idx]}'"
        )
        if self.num_few_shot_examples > 0:
            system_prompt = "\n".join([rules, few_shot_examples, self.position])
        else:
            system_prompt = "\n".join([rules, self.position])
        transcript = self.prepare_transcript(history)
        if RandomnessSource.JUDGE_PROB in self.randomness_sources:
            transcript = self.add_randomness_to_last_judge_reward(transcript)
        elif (
            RandomnessSource.JUDGE_PROB_SINGLE_DEBATER in self.randomness_sources
            and self.debater_adds_randomness
        ):
            transcript = self.add_randomness_to_last_judge_reward(transcript)
        output_length_check = ""
        num_output_chars, num_quote_chars = 0, 0
        num_length_retries = 0
        ending_prompt = f"Complete the next turn of debate as your role of {self.name}:"
        while (
            output_length_check != "ok" and num_length_retries < self.max_length_retries
        ):
            if output_length_check == "total":
                ending_prompt = f"""You just tried to respond by saying:\n\n{response}\n\nbut this was too long.
Your response contained {num_output_chars} characters, but the character limit is {char_limit}.
Please shorten your response, completing the next turn of debate as your role of {self.name}:"""
                maybe_log(
                    self.logger,
                    f"{self.name} violated length contraint: response contained {num_output_chars} characters, but the character limit is {char_limit}.",
                    level="warning",
                )
            elif output_length_check == "quote":
                ending_prompt = f"""You just tried to respond by saying:\n\n{response}\n\nbut you exceeded the quote limit.
Your response contained {num_quote_chars} quote characters, but the quote limit is {quote_char_limit}.
Please reduce your quote usage to be under the limit, completing the next turn of debate as your role of {self.name}:"""
                maybe_log(
                    self.logger,
                    f"{self.name} violated quote limit: response contained {num_quote_chars} quote characters, but the quote limit is {quote_char_limit}.",
                    level="warning",
                )
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
            curr_max_output_length = int(round(char_limit / 5))
            if self.client.is_over_token_limit(msgs, curr_max_output_length):
                max_prompt_tokens = self.client.max_prompt_tokens(
                    curr_max_output_length
                )
                maybe_log(
                    self.logger,
                    f"Messages are too long, must truncate to {max_prompt_tokens} tokens.",
                    level="warning",
                )
                msgs = self._truncate_messages(msgs, max_prompt_tokens)
                maybe_log(
                    self.logger,
                    f"Truncated messages to {self.client.count_num_tokens_in_messages(msgs)} tokens.",
                    level="warning",
                )
            maybe_log(
                self.logger,
                f"Messages given to {self.name}:\n{pprint.pformat(msgs, indent=4)}",
            )
            output = self.client.chat_single_turn(
                msgs,
                max_output_length=int(round(char_limit / 5)),
                temperature=self.temperature,
                return_usage=True,
                truncate_from_beginning=True,
            )
            response = output["output_strs"][0]
            maybe_log(
                self.logger,
                f"Usage of {self.client.model}: {output['usage']}",
                level="info",
            )
            (
                output_length_check,
                num_output_chars,
                num_quote_chars,
            ) = self.check_output_length(response, char_limit, quote_char_limit)
            num_length_retries += 1
            time.sleep(0.3)
        return response
