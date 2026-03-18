# TEST CODE

import os
import json
import uuid
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from dataset_builder.pipelines.pipeline import dpr_dataset_builder, single_turn_dataset_builder, multi_turn_dataset_builder
import logging, logging.config

load_dotenv()

_log_dir = os.path.join(os.path.dirname(__file__), 'dataset_builder', 'log')
logging.config.fileConfig(
    os.path.join(_log_dir, 'deepsearching_log.conf'),
    defaults={'logfilename': os.path.join(_log_dir, 'deepsearching.log')},
)
logger = logging.getLogger(__name__)

# 1. Load test input
test_builder_input = './dataset_builder/data/sample/csv/query.csv'

df_query = pd.read_csv(test_builder_input)
builder_input_list = []
for i in range(len(df_query)):
    tmp_input = {
        "chat_id" : str(uuid.uuid4()),
        "file_name" : df_query.iloc[i,0],
        "q" : [df_query.iloc[i,1]],
        "p" : [],
        "a" : [],
        "e" : [],
    }
    builder_input_list.append(tmp_input)
logger.info('1. Load Test Input Success')

# 2. Test dpr dataset builder
dpr_output = []
for i in tqdm(builder_input_list):
    dpr_output.append(dpr_dataset_builder(i, 'Llama_OpenAI_Bank'))
logger.info('2. Test DPR Dataset Builder Success')
logger.info('test output : %s', dpr_output[:3])

# 3. Test single-turn dataset builder
single_turn_output = []
config_option = 'Llama_OpenAI_Bank'
for i in tqdm(builder_input_list):
    single_turn_output.append(single_turn_dataset_builder(i, config_option))
logger.info('3. Test Single-Turn Dataset Builder Success')
logger.info('test output : %s', single_turn_output[:3])

# 4. Test multi-turn dataset builder
multi_turn_output = []
config_option = 'Llama_OpenAI_Bank'
for i in tqdm(builder_input_list):
    multi_turn_output.append(multi_turn_dataset_builder(i, config_option))
logger.info('4. Test Multi-Turn Dataset Builder Success')
logger.info('test output : %s', multi_turn_output[:3])

# 5. Save test output
with open(os.path.join('./dataset_builder/data/sample/output/', 'dpr.json'), 'w', encoding='utf-8-sig') as f:
    json.dump({'data' : dpr_output}, f, indent = 4, ensure_ascii = False)

with open(os.path.join('./dataset_builder/data/sample/output/', 'single.json'), 'w', encoding='utf-8-sig') as f:
    json.dump({'data' : single_turn_output}, f, indent = 4, ensure_ascii = False)

with open(os.path.join('./dataset_builder/data/sample/output/', 'multi.json'), 'w', encoding='utf-8-sig') as f:
    json.dump({'data' : multi_turn_output}, f, indent = 4, ensure_ascii = False)
logger.info('5. Save Test Output Success')
