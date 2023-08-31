import copy
import backoff
import openai
from bardapi import Bard
import tiktoken

import logging
logger = logging.getLogger(__name__)

MAX_VICUNA_TOKEN_INPUT = 2048

# token 개수 체크해서 2048 넘으면 raise
def get_num_of_tokens(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    
    if isinstance(messages, list):  # openai prompt 형태
        # gpt-3.5-turbo setting
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
        
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    elif isinstance(messages, str): # bard prompt 형태
        num_tokens += len(encoding.encode(messages))
        num_tokens += 100 # for safety
    else:
        logger.error('Invalid Message Type') 
       
    logger.debug('Num of Tokens: %s', str(num_tokens))
    return num_tokens

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries = 5)
def get_answer_from_openai(cond):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=cond['prompt'],
        temperature = cond['temperature'],
        stop = None,
        # max_tokens = 200
    )
    result = completion['choices'][0]['message']['content']
    return result

def get_answer_from_gpt4(cond):
    import requests
    data = {
        "input": {
            "input": cond['prompt']
        }
    }
    headers = {"Authorization":"Basic cliwxbbv100zn1ax74m5f63ut"}
    response = requests.post(
        "https://dashboard.scale.com/spellbook/api/v2/deploy/lc03f4y",
        json=data,
        headers=headers
    )
    result = response.json()['output']

    return result

def get_answer_from_bard(cond):
    import random
    bard_token_list = [
                        "YAiha4_0gCKfTVRs-ICT1g2vQ6nHDcQKxF5dNrO9G4vv9uPrGq1rx-UvYe2BNwrmmPj1Sg.",
                        "YQgLfvpUAJdDv8qBaqq0STruJXVbhH3zB8HvcvLPJC0jq8hxdYVUWRKtwuP47r_mkEKPmQ.",
                        "YQi2BFokUJUkoFk5oOSjbJV7Szr7Ni0Q3QBMuXJDnEwR2seeRirwYq3s6PM0md6OtZBjEw."

                    ]
    random.shuffle(bard_token_list)

    for bard_token in bard_token_list:
        try:
            for i in range(5):
                result = Bard(token=bard_token, timeout=30).get_answer(cond)["content"]
                if ('Response Error:' not in result):
                    return result
        except:
            continue
    raise Exception('Bard Evaluation Failed!')

def answer_builder(input, cond):
    output = copy.deepcopy(input)
    
    if (cond['name'] == 'GetNextQ') and (get_num_of_tokens(cond['prompt']) > MAX_VICUNA_TOKEN_INPUT):  # 프롬프트 길이가 2048를 넘고 next q를 만드는 노드일시 종료 
        return output
    
    if cond['model'] == 'openai':
        result = get_answer_from_openai(cond)
    elif cond['model'] == 'bard':
        result = get_answer_from_bard(cond)
    elif cond['model'] == 'gpt4':
        result = get_answer_from_gpt4(cond)
        
    output[cond['output']].append(result)
    return output