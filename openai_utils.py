import backoff
import json
import logging
import math
import openai
import pprint
import tiktoken

MODEL_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
}


def initialize_openai(openai_creds_file):
    with open(openai_creds_file) as f:
        creds = json.load(f)
    client = openai.OpenAI(
        api_key=creds["api_key"], organization=creds["organization_id"]
    )
    return client


def backoff_hdlr(details):
    logging.warning(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target}".format(**details)
    )


# TODO: Implement max no. of retries
@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIError),
    on_backoff=backoff_hdlr,
)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_tiktoken_encoding(model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding


def estimate_num_chars_to_truncate(curr_no_tokens, curr_no_chars, token_limit):
    diff_tokens = curr_no_tokens - token_limit
    chars_per_token = int(math.floor(curr_no_chars / curr_no_tokens))
    if diff_tokens <= 0:
        return 0
    diff_chars = diff_tokens * chars_per_token
    if diff_chars >= curr_no_chars:
        logging.warning(
            f"Attempting to truncate {diff_chars} characters even though"
            f"the current no. of characters is {curr_no_chars}. Token limit is "
            f"{token_limit}. Returning all characters instead."
        )
        return curr_no_chars
    return diff_chars


def count_num_tokens(text, model="gpt-4"):
    encoding = get_tiktoken_encoding(model)
    return len(encoding.encode(text))


def truncate_text_to_length(text, max_tokens, model="gpt-4"):
    encoding = get_tiktoken_encoding(model)
    curr_text = text
    curr_len_in_tokens = len(encoding.encode(curr_text))
    while curr_len_in_tokens > max_tokens:
        no_chars_to_truncate = estimate_num_chars_to_truncate(
            curr_len_in_tokens, len(curr_text), max_tokens
        )
        # be conservative if the number is large
        if no_chars_to_truncate > 1000:
            no_chars_to_truncate = int(math.ceil(no_chars_to_truncate / 2))
        logging.warning(
            f"Truncating {no_chars_to_truncate} chars from str. of length {len(curr_text)} (chars)."
        )
        curr_text = curr_text[:-no_chars_to_truncate]
        curr_len_in_tokens = len(encoding.encode(curr_text))
    return curr_text


def truncate_messages(
    messages, num_output_tokens, model="gpt-4", truncate_from_beginning=False
):
    encoding = get_tiktoken_encoding(model)
    num_tokens = 2  # every reply is primed with <im_start>assistant
    end_reached = False
    output_messages = []
    truncated = 0
    if truncate_from_beginning:
        messages.reverse()
    for message in messages:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        output_message = {}
        for key, value in message.items():
            try:
                value_len = len(encoding.encode(value))
            except Exception as e:
                logging.error(
                    f"Could not encode:\nkey: {key}\nvalue: {value}\n\nAll messages:\n{pprint.pformat(messages)}"
                )
                raise e
            if num_tokens + value_len > num_output_tokens:
                end_reached = True
                while num_tokens + value_len > num_output_tokens:
                    no_chars_to_truncate = estimate_num_chars_to_truncate(
                        value_len, len(value), num_output_tokens - num_tokens
                    )
                    # be conservative if the number is large
                    if no_chars_to_truncate > 100:
                        no_chars_to_truncate = int(math.ceil(no_chars_to_truncate / 2))
                    logging.warning(
                        f"Truncating {no_chars_to_truncate} chars from value of length {len(value)}"
                    )
                    value = value[:-no_chars_to_truncate]
                    truncated += no_chars_to_truncate
                    value_len = len(encoding.encode(value))
                output_message[key] = value
                break
            num_tokens += value_len
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
            output_message[key] = value
        output_messages.append(output_message)
        if end_reached:
            break
    if truncated > 0:
        logging.warning(
            f"Truncated {truncated} characters to reach max token length of {num_output_tokens} tokens."
        )
    if truncate_from_beginning:
        output_messages.reverse()
    return output_messages


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def is_over_token_limit(msgs, model, max_output_length):
    num_tokens_left = MODEL_CONTEXT_LENGTHS[model] - max_output_length
    num_tokens = num_tokens_from_messages(msgs, model=model)
    return num_tokens > num_tokens_left


def sample_from_openai_chat_model(
    client,
    arch,
    msgs,
    max_output_length=128,  # tokens
    num_samples=1,
    temperature=1.0,
    model_id=None,
    return_usage=False,
    truncate_from_beginning=False,
):
    output = {"output_strs": []}

    if arch not in MODEL_CONTEXT_LENGTHS:
        raise ValueError(
            f"{arch} is not a supported OpenAI model. Supported models: {MODEL_CONTEXT_LENGTHS.keys()}"
        )
    model = model_id if model_id is not None else arch
    msgs = truncate_messages(
        msgs,
        MODEL_CONTEXT_LENGTHS[model] - max_output_length,
        model=arch,
        truncate_from_beginning=truncate_from_beginning,
    )
    response = completions_with_backoff(
        client,
        model=model,
        messages=msgs,
        max_tokens=max_output_length,
        n=num_samples,
        temperature=temperature,
    )
    for choice in response.choices:
        output["output_strs"].append(choice.message.content)
    if not return_usage:
        return output
    output["usage"] = response.usage
    return output
