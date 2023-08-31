import copy
import json
import requests
import yaml
import openai
import os
from fastapi import HTTPException

IP = "91.203.132.7"
PORT = "4001"
backend = f'http://{IP}:{PORT}/semantic_search_test'

def load_config(config_dir):
    with open(config_dir, 'rb') as f :
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args

def post(url, data):
    try:
        response = requests.post(url, data=json.dumps(data, ensure_ascii=False, indent="\t").encode("utf-8"))

        if response.status_code == 200:
            response_json = response.json()
            return response_json
        else:
            raise HTTPException(status_code=response.status_code, detail=response)
        
    except Exception as e:
        raise e

def search_builder(input, config_name, top_k = 3):
    config_dir = './dataset_builder/config/pipeline_config.yaml'
    args = load_config(config_dir)
    
    openai.api_key = args['OpenAI']['api_key']
    os.environ['OPENAI_API_KEY'] = args['OpenAI']['api_key']
    
    if args[config_name]['model'] == 'OpenAI':
        url = backend
    elif args[config_name]['model'] == 'DPR':
        raise
    else:
        raise
        
    output = copy.deepcopy(input)
    data = {
        "query": input['q'][-1],
        "file_name": input['file_name'],
        "top_k": 10,
        "cond": {},
        "debug": False,
    }
    p = post(url, data)['response']
    p = [i['node']['text'] for i in p][:top_k]
    output['p'].append(p)
    
    return output


