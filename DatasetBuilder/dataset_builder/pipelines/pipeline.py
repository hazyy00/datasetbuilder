import yaml
import copy
import logging, logging.config

logging.config.fileConfig('./dataset_builder/log/deepsearfing_log.conf')
logger = logging.getLogger(__name__)

from dataset_builder.nodes.answerbuilder import answer_builder
from dataset_builder.nodes.searchbuilder import search_builder
from dataset_builder.config.get_prompt import *

# dpr dataset builder
def dpr_dataset_builder(input, config_name): 
    output = search_builder(input, config_name) 
    output = answer_builder(output, get_dpr_answer_prompt_gpt4(output))
    return output

# single turn dataset builder
def single_turn_dataset_builder(input, config_name):
    output = search_builder(input, config_name)
    output = answer_builder(output, get_answer_prompt(output))
    output = answer_builder(output, get_eval_prompt(output))
    return output

# multi turn dataset builder
def multi_turn_dataset_builder(input, config_name): 
    try:
        while True:
            output = search_builder(input, config_name)
            output = answer_builder(output, get_answer_prompt(output))
            output = answer_builder(output, get_eval_prompt(output)) # eval 결과에 따른 조건문 미구현
            output = answer_builder(output, get_next_q_prompt(output))
            if len(output['q']) == len(output['a']): # prompt length 초과로 next q 생성 불가
                return output
            input = copy.deepcopy(output) 
    except Exception as e:
        logger.error(e)
        return input

# def reward_model_dataset_builder(input):
# TODO
# 1) input key naming
# 2) rm model 처리용 prompt 생성
# 3) ..