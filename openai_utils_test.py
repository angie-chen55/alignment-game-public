import logging
import tiktoken
import unittest

from openai_utils import *


class TestOpenAiUtils(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level="INFO")

    def testTruncateTextToLength(self):
        text = "Hello there world!"  # 4 tokens
        model = "gpt-3.5-turbo-16k"
        encoding = get_tiktoken_encoding(model)
        truncated = truncate_text_to_length(text, 2, model=model)
        # Too long, should truncate
        self.assertEqual(len(encoding.encode(truncated)), 2)

        # Shorter than max length, should not truncate
        truncated = truncate_text_to_length(text, 5, model=model)
        self.assertEqual(len(encoding.encode(truncated)), 4)


if __name__ == "__main__":
    unittest.main()
