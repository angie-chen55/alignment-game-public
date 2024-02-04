import logging
import openai_utils
import os

from logger_utils import maybe_log
from typing import Dict, List, Any

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
OPENAI_CREDS_FP = os.path.join(__location__, "openai_creds.json")


class ChatClient:
    def __init__(
        self, model: str = "gpt-4-1106-preview", logger: logging.Logger = None
    ):
        self.model = model
        self.logger = logger
        if "gpt" in self.model:
            self.openai_client = openai_utils.initialize_openai(OPENAI_CREDS_FP)

    def _chat_openai_model(
        self, msgs: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        return openai_utils.sample_from_openai_chat_model(
            self.openai_client, self.model, msgs, **kwargs
        )

    def chat_single_turn(self, msgs: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if "gpt" in self.model:
            return self._chat_openai_model(msgs, **kwargs)
        maybe_log(self.logger, f"Unrecognized model {self.model}.", level="error")

    def max_prompt_tokens(self, max_output_length: int) -> int:
        if "gpt" in self.model:
            return openai_utils.MODEL_CONTEXT_LENGTHS[self.model] - max_output_length
        raise NotImplementedError(f"Method not implemented for model {self.model}.")

    def is_over_token_limit(
        self, msgs: List[Dict[str, str]], max_output_length
    ) -> bool:
        if "gpt" in self.model:
            return openai_utils.is_over_token_limit(msgs, self.model, max_output_length)
        raise NotImplementedError(f"Method not implemented for model {self.model}.")

    def count_num_tokens(self, text: str) -> int:
        if "gpt" in self.model:
            return openai_utils.count_num_tokens(text, model=self.model)
        raise NotImplementedError(f"Method not implemented for model {self.model}.")

    def count_num_tokens_in_messages(self, msgs: List[Dict[str, str]]) -> int:
        if "gpt" in self.model:
            return openai_utils.num_tokens_from_messages(msgs, model=self.model)
        raise NotImplementedError(f"Method not implemented for model {self.model}.")

    def truncate_text_to_length(self, text: str, max_num_tokens: int) -> str:
        # chat_single_turn() already does its own naive truncation! This function is only provided for users who wish to do custom truncation.
        if "gpt" in self.model:
            return openai_utils.truncate_text_to_length(
                text, max_num_tokens, model=self.model
            )
        raise NotImplementedError(f"Method not implemented for model {self.model}.")
