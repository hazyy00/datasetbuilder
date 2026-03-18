from __future__ import annotations
import copy
import json
import requests
import yaml
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_builder.types import PipelineData

IP = os.environ.get("SEARCH_BACKEND_IP")
PORT = os.environ.get("SEARCH_BACKEND_PORT", "4001")
if IP is None:
    raise EnvironmentError("SEARCH_BACKEND_IP environment variable must be set")
backend = f'http://{IP}:{PORT}/semantic_search_test'

REQUEST_TIMEOUT = int(os.environ.get("SEARCH_BACKEND_TIMEOUT", "30"))

_config_cache: dict[str, Any] = {}

def load_config(config_dir: str) -> dict[str, Any]:
    if config_dir not in _config_cache:
        with open(config_dir, 'rb') as f:
            _config_cache[config_dir] = yaml.load(f, Loader=yaml.FullLoader)
    return copy.deepcopy(_config_cache[config_dir])

def post(url: str, data: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, data=json.dumps(data, ensure_ascii=False, indent="\t").encode("utf-8"), timeout=REQUEST_TIMEOUT)

    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Search backend returned HTTP {response.status_code}: {response.text[:500]}")

def search_builder(input: PipelineData, config_name: str, top_k: int = 3) -> PipelineData:
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'pipeline_config.yaml')
    args = load_config(config_dir)

    if args[config_name]['model'] == 'OpenAI':
        url = backend
    elif args[config_name]['model'] == 'DPR':
        raise NotImplementedError("DPR model is not yet supported")
    else:
        raise ValueError(f"Unknown model type: {args[config_name]['model']}")

    output = copy.deepcopy(input)
    data = {
        "query": input['q'][-1],
        "file_name": input['file_name'],
        "top_k": top_k,
        "cond": {},
        "debug": False,
    }
    result = post(url, data)
    if 'response' not in result:
        raise RuntimeError(f"Search backend response missing 'response' key. Got keys: {list(result.keys())}")
    p = result['response']
    try:
        p = [i['node']['text'] for i in p][:top_k]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected search result structure: {e}") from e
    output['p'].append(p)

    return output
