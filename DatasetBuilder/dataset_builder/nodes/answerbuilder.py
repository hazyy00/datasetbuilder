from __future__ import annotations
import copy
import backoff
from openai import OpenAI, RateLimitError
import tiktoken
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_builder.types import PipelineData

import logging
logger = logging.getLogger(__name__)

client = OpenAI()

MAX_VICUNA_TOKEN_INPUT = 2048

_encoding = tiktoken.get_encoding("cl100k_base")

# Check token count; skip if exceeds 2048
def get_num_of_tokens(messages):
    num_tokens = 0

    if isinstance(messages, list):  # OpenAI prompt format (message list)
        # gpt-3.5-turbo setting
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(_encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    elif isinstance(messages, str):  # Plain string prompt format
        num_tokens += len(_encoding.encode(messages))
        num_tokens += 100  # for safety
    else:
        raise TypeError(f'Expected list or str for messages, got {type(messages).__name__}')

    logger.debug('Num of Tokens: %s', str(num_tokens))
    return num_tokens

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def get_answer_from_openai(cond):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=cond['prompt'],
        temperature=cond['temperature'],
        stop=None,
    )
    if not completion.choices:
        raise RuntimeError("OpenAI returned empty choices")
    return completion.choices[0].message.content

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def get_answer_from_gpt4(cond):
    messages = cond['prompt'] if isinstance(cond['prompt'], list) else [{"role": "user", "content": cond['prompt']}]
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=cond['temperature'],
    )
    if not completion.choices:
        raise RuntimeError("OpenAI returned empty choices")
    return completion.choices[0].message.content

class PromptTooLongError(Exception):
    """Raised when prompt token count exceeds the configured maximum."""
    pass

def answer_builder(input: PipelineData, cond: dict[str, Any]) -> PipelineData:
    output = copy.deepcopy(input)

    if (cond['name'] == 'GetNextQ') and (get_num_of_tokens(cond['prompt']) > MAX_VICUNA_TOKEN_INPUT):
        logger.warning('Prompt length exceeded %d tokens; skipping next question generation', MAX_VICUNA_TOKEN_INPUT)
        raise PromptTooLongError(f"Prompt exceeds {MAX_VICUNA_TOKEN_INPUT} tokens for {cond['name']}")

    if cond['model'] == 'openai':
        result = get_answer_from_openai(cond)
    elif cond['model'] == 'gpt4':
        result = get_answer_from_gpt4(cond)
    else:
        raise ValueError(f"Unsupported model type: {cond['model']}")

    output[cond['output']].append(result)
    return output
