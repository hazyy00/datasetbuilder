# DatasetBuilder

A pipeline for generating LLM training datasets by combining semantic search with GPT-3.5/GPT-4. Given a set of questions and source documents, it retrieves relevant paragraphs, generates answers, evaluates answer quality, and optionally produces multi-turn conversations — outputting structured JSON datasets ready for fine-tuning.

## Background

This project was originally developed as part of a larger system that includes a semantic search backend for document retrieval. The search component (referenced throughout as the "search backend") is a separate service that indexes documents and serves relevant paragraphs via API — this repository contains only the dataset generation pipeline that consumes that search API.

The codebase was originally written in Korean and has been translated to English, modernized (updated to the current OpenAI SDK, secrets removed, dead code cleaned up), and hardened for production use.

## How It Works

The pipeline follows a retrieve-then-generate pattern:

1. **Search** — Sends the user's question to a semantic search backend, which returns the top-k most relevant paragraphs from the specified document.
2. **Answer** — Feeds the question + retrieved paragraphs to GPT-3.5-turbo to generate a natural language answer.
3. **Evaluate** — Sends the question, paragraphs, and generated answer to GPT-4 for quality scoring (1-10 scale).
4. **Follow-up** *(multi-turn only)* — Generates a follow-up question to continue the conversation, then repeats the cycle.

## Pipelines

| Pipeline | Function | Description |
|----------|----------|-------------|
| **DPR** | `dpr_dataset_builder()` | Search + GPT-4 paragraph relevance scoring. Outputs relevance scores for each retrieved paragraph. |
| **Single-turn** | `single_turn_dataset_builder()` | Search + answer generation + answer evaluation. One question, one answer. |
| **Multi-turn** | `multi_turn_dataset_builder()` | Iterates single-turn with follow-up question generation for up to `max_turns` (default 5) conversational turns. |

## Project Structure

```
DatasetBuilder/
  dataset_builder/
    types.py                    # PipelineData TypedDict (shared data schema)
    config/
      get_prompt.py             # Prompt templates for answer, eval, and follow-up generation
      pipeline_config.yaml      # Pipeline configuration (model selection per config name)
    nodes/
      searchbuilder.py          # Semantic search backend client
      answerbuilder.py          # OpenAI API client (GPT-3.5 / GPT-4) with retry logic
    pipelines/
      pipeline.py               # Pipeline orchestration (DPR, single-turn, multi-turn)
    data/
      sample/
        csv/query.csv           # Sample input queries
        output/                 # Generated dataset output (JSON)
    log/
      deepsearching_log.conf    # Logging configuration
  test.py                       # Integration test script
```

## Data Schema

All pipelines operate on a shared `PipelineData` dictionary:

```python
{
    "chat_id":   str,             # Unique conversation identifier
    "file_name": str,             # Source document name to search against
    "q":         list[str],       # Questions (grows with each turn)
    "p":         list[list[str]], # Retrieved paragraphs per turn
    "a":         list[str],       # Generated answers
    "e":         list[str],       # Evaluation scores/explanations
}
```

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to GPT-3.5-turbo and GPT-4
- A running semantic search backend (see [Search Backend](#search-backend) below)

### Installation

```bash
git clone https://github.com/hazyy00/datasetbuilder.git
cd datasetbuilder
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
SEARCH_BACKEND_IP=your-server-ip       # No default — must be set

# Optional
SEARCH_BACKEND_PORT=4001               # Default: 4001
SEARCH_BACKEND_TIMEOUT=30              # Request timeout in seconds. Default: 30
```

## Usage

### Running the test script

```bash
cd DatasetBuilder
python test.py
```

This reads `dataset_builder/data/sample/csv/query.csv`, runs all three pipelines against the `Llama_OpenAI_Bank` configuration, and writes output to `dataset_builder/data/sample/output/`.

### Using the pipelines in your own code

```python
import uuid
from dataset_builder.pipelines.pipeline import (
    dpr_dataset_builder,
    single_turn_dataset_builder,
    multi_turn_dataset_builder,
)

data = {
    "chat_id": str(uuid.uuid4()),
    "file_name": "your_document.pdf",
    "q": ["What is the insurance premium?"],
    "p": [],
    "a": [],
    "e": [],
}

# Paragraph relevance scoring
result = dpr_dataset_builder(data, "Llama_OpenAI_Bank")

# Single question-answer with evaluation
result = single_turn_dataset_builder(data, "Llama_OpenAI_Bank")

# Multi-turn conversation (up to 5 turns)
result = multi_turn_dataset_builder(data, "Llama_OpenAI_Bank", max_turns=5)
```

### Input CSV format

The input CSV should have two columns:

| file | query |
|------|-------|
| document_name.pdf | Your question here |

### Pipeline configuration

Pipeline configs are defined in `dataset_builder/config/pipeline_config.yaml`:

```yaml
Llama_OpenAI_Bank:
  model: 'OpenAI'
```

The `model` field determines which search backend to use. Currently only `OpenAI` is supported (`DPR` is planned but not yet implemented).

## Search Backend

This pipeline expects an external semantic search service at `http://{SEARCH_BACKEND_IP}:{SEARCH_BACKEND_PORT}/semantic_search_test`. This service is part of the broader system this project belongs to and is not included in this repository.

**Expected API contract:**

Request (POST, JSON):
```json
{
    "query": "What is the premium?",
    "file_name": "document.pdf",
    "top_k": 3,
    "cond": {},
    "debug": false
}
```

Response:
```json
{
    "response": [
        {"node": {"text": "Paragraph content here..."}},
        {"node": {"text": "Another paragraph..."}},
        {"node": {"text": "Third paragraph..."}}
    ]
}
```

## Dependencies

- **openai** — GPT-3.5-turbo and GPT-4 API calls
- **tiktoken** — Token counting for prompt length validation
- **backoff** — Exponential retry on OpenAI rate limits
- **requests** — HTTP client for the search backend
- **pyyaml** — Pipeline configuration parsing
- **pandas** / **tqdm** — Used in the test script for CSV loading and progress display
- **python-dotenv** — Environment variable loading from `.env` files
