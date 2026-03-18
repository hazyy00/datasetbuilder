import copy
import os
import logging, logging.config

_log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')
logging.config.fileConfig(
    os.path.join(_log_dir, 'deepsearching_log.conf'),
    defaults={'logfilename': os.path.join(_log_dir, 'deepsearching.log')},
)
logger = logging.getLogger(__name__)

from dataset_builder.types import PipelineData
from dataset_builder.nodes.answerbuilder import answer_builder, PromptTooLongError
from dataset_builder.nodes.searchbuilder import search_builder
from dataset_builder.config.get_prompt import get_dpr_answer_prompt, get_answer_prompt, get_eval_prompt, get_next_q_prompt

# dpr dataset builder
def dpr_dataset_builder(input: PipelineData, config_name: str) -> PipelineData:
    output = search_builder(input, config_name)
    output = answer_builder(output, get_dpr_answer_prompt(output, model='gpt4'))
    return output

# single turn dataset builder
def single_turn_dataset_builder(input: PipelineData, config_name: str) -> PipelineData:
    output = search_builder(input, config_name)
    output = answer_builder(output, get_answer_prompt(output))
    output = answer_builder(output, get_eval_prompt(output))
    return output

# multi turn dataset builder
def multi_turn_dataset_builder(input: PipelineData, config_name: str, max_turns: int = 5) -> PipelineData:
    output = copy.deepcopy(input)
    turn = 0
    try:
        for turn in range(max_turns):
            output = search_builder(output, config_name)
            output = answer_builder(output, get_answer_prompt(output))
            output = answer_builder(output, get_eval_prompt(output))  # TODO: Add conditional logic based on eval results
            try:
                output = answer_builder(output, get_next_q_prompt(output))
            except PromptTooLongError:
                logger.info('Prompt too long on turn %d; ending multi-turn loop', turn + 1)
                return output
        logger.warning('Reached max_turns (%d) without natural termination', max_turns)
    except (KeyError, ValueError, RuntimeError) as e:
        logger.error('Multi-turn pipeline failed on turn %d: %s', turn + 1, e)
    return output
