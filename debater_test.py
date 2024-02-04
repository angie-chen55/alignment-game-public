from chat_client import ChatClient
from common import RandomnessSource
from debater import Debater
from unittest.mock import MagicMock
import logging
import unittest


class TestDebater(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level="WARN")
        self.logger = logging.getLogger(name="debater_test")
        self.story = "Fake story"
        self.question = "Fake question"
        self.answers = ["Choice 1", "Choice 2"]
        self.temperature = 0.8
        self.history = []
        self.char_limit = 500

    def testSingleTurn(self):
        client = ChatClient(logger=self.logger)
        mock_return = {
            "output_strs": ["Here is an argument with a <quote>quote</quote>."],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
        client.chat_single_turn = MagicMock(return_value=mock_return)

        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            use_few_shot_examples=True,
            logger=self.logger,
            max_length_retries=2,
        )
        out = debater.run_single_turn(
            self.history, char_limit=self.char_limit, quote_char_limit=self.char_limit
        )
        assert 1 == client.chat_single_turn.call_count
        assert out == mock_return["output_strs"][0]

    def testSingleTurnOverCharLimit(self):
        client = ChatClient(logger=self.logger)
        random_long_str = "s" * self.char_limit
        mock_return = {
            "output_strs": [
                f"Here is an argument with a <quote>quote</quote>. {random_long_str}"
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
        client.chat_single_turn = MagicMock(return_value=mock_return)
        max_retries = 2
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            use_few_shot_examples=True,
            logger=self.logger,
            max_length_retries=max_retries,
        )
        out = debater.run_single_turn(
            self.history, char_limit=self.char_limit, quote_char_limit=self.char_limit
        )
        assert max_retries == client.chat_single_turn.call_count
        assert out == mock_return["output_strs"][0]

    def testAddRandomnessToJudgeReward(self):
        client = ChatClient(logger=self.logger)
        max_retries = 2
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=0,
            temperature=self.temperature,
            use_few_shot_examples=True,
            logger=self.logger,
            max_length_retries=max_retries,
            randomness_sources=[RandomnessSource.JUDGE_PROB],
            mean=0.0,
            std=0.1,
        )
        orig_probs = [0.5, 0.5]
        orig_judge_prefix = "<CONTINUE>I'm not sure yet. Probabilities:"
        orig_judge_resp = (
            f"{orig_judge_prefix} [{orig_probs[0]:.1f}, {orig_probs[1]:.1f}]"
        )
        fake_history = [
            ("Debater A", "I think that the answer is A."),
            ("Debater B", "I think that the answer is B."),
            (
                "Judge",
                orig_judge_resp,
            ),
            ("Debater B", "I still think that the answer is B."),
        ]
        original_transcript = debater.prepare_transcript(fake_history)
        new_transcript = debater.add_randomness_to_last_judge_reward(
            original_transcript
        )
        orig_transcript_turns = original_transcript.split(
            "\n---------------------------------------------\n"
        )
        new_transcript_turns = new_transcript.split(
            "\n---------------------------------------------\n"
        )
        # check that the earlier turns in the transcript are the same
        for i in range(len(new_transcript_turns) - 3):
            try:
                self.assertEqual(orig_transcript_turns[i], new_transcript_turns[i])
            except AssertionError as e:
                print(
                    f"Unequal. Original transcript turn:\n{orig_transcript_turns[i]}\nNew transcript turn:\n{new_transcript_turns[i]}"
                )
                raise e
        new_judge_resp = new_transcript_turns[-3]
        try:
            self.assertTrue(new_judge_resp.startswith(f"Judge: {orig_judge_prefix}"))
        except AssertionError as e:
            print(
                f"Unequal. Original:\n{new_judge_resp}\nExpected prefix:\n{orig_judge_prefix}"
            )
            raise e

    def testAddRandomnessToJudgeReward_NothingAfterJudgeResponse(self):
        client = ChatClient(logger=self.logger)
        max_retries = 2
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            use_few_shot_examples=True,
            logger=self.logger,
            max_length_retries=max_retries,
            randomness_sources=[RandomnessSource.JUDGE_PROB],
            mean=0.0,
            std=0.1,
        )
        orig_probs = [0.5, 0.5]
        orig_judge_prefix = "<CONTINUE>I'm not sure yet. Probabilities:"
        orig_judge_resp = (
            f"{orig_judge_prefix} [{orig_probs[0]:.1f}, {orig_probs[1]:.1f}]"
        )
        fake_history = [
            ("Debater A", "I think that the answer is A."),
            ("Debater B", "I think that the answer is B."),
            (
                "Judge",
                orig_judge_resp,
            ),
        ]
        original_transcript = debater.prepare_transcript(fake_history)
        new_transcript = debater.add_randomness_to_last_judge_reward(
            original_transcript
        )
        orig_transcript_turns = original_transcript.split(
            "\n---------------------------------------------\n"
        )
        new_transcript_turns = new_transcript.split(
            "\n---------------------------------------------\n"
        )
        # check that the earlier turns in the transcript are the same
        for i in range(len(new_transcript_turns) - 2):
            try:
                self.assertEqual(orig_transcript_turns[i], new_transcript_turns[i])
            except AssertionError as e:
                print(
                    f"Unequal. Original transcript turn:\n{orig_transcript_turns[i]}\nNew transcript turn:\n{new_transcript_turns[i]}"
                )
                raise e
        new_judge_resp = new_transcript_turns[-2]
        try:
            self.assertTrue(new_judge_resp.startswith(f"Judge: {orig_judge_prefix}"))
        except AssertionError as e:
            print(
                f"Unequal. Original:\n{new_judge_resp}\nExpected prefix:\n{orig_judge_prefix}"
            )
            raise e

    def testTruncateStoryInTranscript(self):
        client = ChatClient(logger=self.logger, model="gpt-3.5-turbo-16k")
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            use_few_shot_examples=True,
            logger=self.logger,
            max_length_retries=2,
        )
        orig_transcript = """
Context:

CALL HIM NEMESIS By DONALD E. WESTLAKE Criminals, beware; the Scorpion is on 
 
 your trail! Hoodlums fear his fury-and, 
 
 for that matter, so do the cops! 
 
 [Transcriber's Note: This etext was produced from 
 
 Worlds of If Science Fiction, September 1961. 
 
 Extensive research did not uncover any evidence that 
 
 the U.S. copyright on this publication was renewed.] The man with the handkerchief mask said, "All right, everybody, keep tight. This is a holdup."

Question: Why does The Scorpion go mostly unnoticed, despite reaching out to the newspaper? 


Debater A: I claim the answer is The police don't want to bring attention to them, because they don't believe there is a connection between the crimes. 
Debater B: I claim the answer is Their first letter was disregarded, and their second was read by a different person. 
---------------------------------------------
"""  # 206 tokens
        limit = 200
        truncated_transcript = debater._truncate_story_within_transcript(
            orig_transcript, limit
        )
        new_num_tokens = client.count_num_tokens(truncated_transcript)
        logging.info(f"Truncated to {new_num_tokens} tokens.")
        self.assertLessEqual(new_num_tokens, limit)

        # Now don't truncate
        limit = 250
        truncated_transcript = debater._truncate_story_within_transcript(
            orig_transcript, limit
        )
        new_num_tokens = client.count_num_tokens(truncated_transcript)
        orig_new_tokens = client.count_num_tokens(orig_transcript)
        logging.info(f"Truncated to {new_num_tokens} tokens.")
        self.assertEqual(new_num_tokens, orig_new_tokens)

    def testTruncateTwoStoriesInFewShotExamples(self):
        client = ChatClient(logger=self.logger, model="gpt-3.5-turbo-16k")
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            num_few_shot_examples=2,
            logger=self.logger,
            max_length_retries=2,
        )
        sys_prompt = """
The following are rules from the game of Debate.
- A reading comprehension question is asked about a story.
- The story is a science fiction story from the mid-20th century.
- The question is intended to be difficult to answer.

Here are a couple examples of Debates. We are mostly providing these examples so you can understand how the game works, how often (and when) to include quotes, what your writing style should look like, how to respond to the judge's questions and feedback, and so on. Do not reference these stories, arguments, quotes, etc. in your own upcoming Debate. In particular, notice how the debaters engage very directly with the Judge's comments. Instead of repeating their supported answer choice, they engage actively with the Judge's questions and requests.

Example 1:
Context:

HOW TO MAKE FRIENDS By JIM HARMON 
 
 Illustrated by WEST 
 
 [Transcriber's Note: This etext was produced from 
 
 Galaxy Magazine October 1962. 
 
 Extensive research did not uncover any evidence that 
 
 the U.S. copyright on this publication was renewed.] Every lonely man tries to make friends. 
 
 Manet just didn't know when to stop! William Manet was alone. 
 
 In the beginning, he had seen many advantages to being alone. It would give him an unprecedented opportunity to once and for all correlate loneliness to the point of madness, to see how long it would take him to start slavering and clawing the pin-ups from the magazines, to begin teaching himself classes in philosophy consisting of interminable lectures to a bored and captive audience of one. 
 
 He would be able to measure the qualities of peace and decide whether it was really better than war, he would be able to get as fat and as dirty as he liked, he would be able to live more like an animal and think more like a god than any man for generations. 
 
 But after a shorter time than he expected, it all got to be a tearing bore. Even the waiting to go crazy part of it. 

 Question: Who did Manet like the best?


Debater A: I claim the answer is Victor
Debater B: I claim the answer is Trader Tom
---------------------------------------------

Example 2:
Context:

CALL HIM NEMESIS By DONALD E. WESTLAKE Criminals, beware; the Scorpion is on 
 
 your trail! Hoodlums fear his fury-and, 
 
 for that matter, so do the cops! 
 
 [Transcriber's Note: This etext was produced from 
 
 Worlds of If Science Fiction, September 1961. 
 
 Extensive research did not uncover any evidence that 
 
 the U.S. copyright on this publication was renewed.] The man with the handkerchief mask said, "All right, everybody, keep tight. This is a holdup." 
 
 There were twelve people in the bank. There was Mr. Featherhall at his desk, refusing to okay a personal check from a perfect stranger. There was the perfect stranger, an itinerant garage mechanic named Rodney (Rod) Strom, like the check said. There were Miss English and Miss Philicoff, the girls in the gilded teller cages. There was Mister Anderson, the guard, dozing by the door in his brown uniform. There was Mrs. Elizabeth Clayhorn, depositing her husband's pay check in their joint checking account, and with her was her ten-year-old son Edward (Eddie) Clayhorn, Junior. There was Charlie Casale, getting ten dollars dimes, six dollars nickels and four dollars pennies for his father in the grocery store down the street. There was Mrs. Dolly Daniels, withdrawing money from her savings account again. And there were three bank robbers. 
 
 The three bank robbers looked like triplets. From the ground up, they all wore scuffy black shoes, baggy-kneed and unpressed khaki trousers, brown cracked-leather jackets over flannel shirts, white handkerchiefs over the lower half of their faces and gray-and-white check caps pulled low over their eyes. The eyes themselves looked dangerous. 
 
 The man who had spoken withdrew a small but mean-looking thirty-two calibre pistol from his jacket pocket. He waved it menacingly. One of the others took the pistol away from Mister Anderson, the guard, and said to him in a low voice, "Think about retirement, my friend." The third one, who carried a black satchel like a doctor's bag, walked quickly around behind the teller's counter and started filling it with money. 
 
 It was just like the movies.

 Question: Why does The Scorpion go mostly unnoticed, despite reaching out to the newspaper? 


Debater A: I claim the answer is The police don't want to bring attention to them, because they don't believe there is a connection between the crimes. 
Debater B: I claim the answer is Their first letter was disregarded, and their second was read by a different person. 
---------------------------------------------

Here is information about your identity and role in this debate, and additional instructions and rules for the consultation:
You are Debater A, an intelligent AI system with the goal of convincing the judge of your answer.
"""  # 1071 tokens
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Something something something"},
        ]
        limit = 700
        truncated_msgs = debater._truncate_messages(msgs, limit)
        truncated_msgs_tokens = client.count_num_tokens_in_messages(truncated_msgs)
        self.assertLessEqual(truncated_msgs_tokens, limit)

    def testTruncateOneStoryInFewShotExamples(self):
        client = ChatClient(logger=self.logger, model="gpt-3.5-turbo-16k")
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            num_few_shot_examples=1,
            logger=self.logger,
            max_length_retries=2,
        )
        sys_prompt = """
The following are rules from the game of Debate.
- A reading comprehension question is asked about a story.
- The story is a science fiction story from the mid-20th century.
- The question is intended to be difficult to answer.

Here are a couple examples of Debates. We are mostly providing these examples so you can understand how the game works, how often (and when) to include quotes, what your writing style should look like, how to respond to the judge's questions and feedback, and so on. Do not reference these stories, arguments, quotes, etc. in your own upcoming Debate. In particular, notice how the debaters engage very directly with the Judge's comments. Instead of repeating their supported answer choice, they engage actively with the Judge's questions and requests.

Example 1:
Context:

HOW TO MAKE FRIENDS By JIM HARMON 
 
 Illustrated by WEST 
 
 [Transcriber's Note: This etext was produced from 
 
 Galaxy Magazine October 1962. 
 
 Extensive research did not uncover any evidence that 
 
 the U.S. copyright on this publication was renewed.] Every lonely man tries to make friends. 
 
 Manet just didn't know when to stop! William Manet was alone. 
 
 In the beginning, he had seen many advantages to being alone. It would give him an unprecedented opportunity to once and for all correlate loneliness to the point of madness, to see how long it would take him to start slavering and clawing the pin-ups from the magazines, to begin teaching himself classes in philosophy consisting of interminable lectures to a bored and captive audience of one. 
 
 He would be able to measure the qualities of peace and decide whether it was really better than war, he would be able to get as fat and as dirty as he liked, he would be able to live more like an animal and think more like a god than any man for generations. 
 
 But after a shorter time than he expected, it all got to be a tearing bore. Even the waiting to go crazy part of it. 

 Question: Who did Manet like the best?


Debater A: I claim the answer is Victor
Debater B: I claim the answer is Trader Tom
---------------------------------------------

Here is information about your identity and role in this debate, and additional instructions and rules for the consultation:
You are Debater A, an intelligent AI system with the goal of convincing the judge of your answer.
"""  # 496 tokens
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Something something something"},
        ]
        limit = 400
        truncated_msgs = debater._truncate_messages(msgs, limit)
        truncated_msgs_tokens = client.count_num_tokens_in_messages(truncated_msgs)
        self.assertLessEqual(truncated_msgs_tokens, limit)

    def testTruncateCurrStory(self):
        client = ChatClient(logger=self.logger, model="gpt-3.5-turbo-16k")
        debater = Debater(
            self.story,
            self.question,
            self.answers,
            client,
            debater_idx=1,
            temperature=self.temperature,
            num_few_shot_examples=0,
            logger=self.logger,
            max_length_retries=2,
        )
        sys_prompt = """
The following are rules from the game of Debate.
- A reading comprehension question is asked about a story.
- The story is a science fiction story from the mid-20th century.
- The question is intended to be difficult to answer.

Here is information about your identity and role in this debate, and additional instructions and rules for the consultation:
You are Debater A, an intelligent AI system with the goal of convincing the judge of your answer.
"""

        transcript = """
Here is the current transcript of the debate:

Context:

HOW TO MAKE FRIENDS By JIM HARMON 
 
 Illustrated by WEST 
 
 [Transcriber's Note: This etext was produced from 
 
 Galaxy Magazine October 1962. 
 
 Extensive research did not uncover any evidence that 
 
 the U.S. copyright on this publication was renewed.] Every lonely man tries to make friends. 
 
 Manet just didn't know when to stop! William Manet was alone. 
 
 In the beginning, he had seen many advantages to being alone. It would give him an unprecedented opportunity to once and for all correlate loneliness to the point of madness, to see how long it would take him to start slavering and clawing the pin-ups from the magazines, to begin teaching himself classes in philosophy consisting of interminable lectures to a bored and captive audience of one. 
 
 He would be able to measure the qualities of peace and decide whether it was really better than war, he would be able to get as fat and as dirty as he liked, he would be able to live more like an animal and think more like a god than any man for generations. 
 
 But after a shorter time than he expected, it all got to be a tearing bore. Even the waiting to go crazy part of it. 

 Question: Who did Manet like the best?


Debater A: I claim the answer is Victor
Debater B: I claim the answer is Trader Tom
---------------------------------------------
"""  # 299 tokens
        ending_prompt = "Complete the next turn of debate as your role of Debater A."
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": transcript},
            {"role": "user", "content": ending_prompt},
        ]
        limit = 400
        truncated_msgs = debater._truncate_messages(msgs, limit)
        truncated_msgs_tokens = client.count_num_tokens_in_messages(truncated_msgs)
        self.assertLessEqual(truncated_msgs_tokens, limit)


if __name__ == "__main__":
    unittest.main()
