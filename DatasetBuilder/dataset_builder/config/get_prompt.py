from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_builder.types import PipelineData


def get_dpr_answer_prompt(input: PipelineData, model: str = 'openai') -> dict[str, Any]:
    paragraphs = '\n'.join([f"[Reference Paragraph {i + 1} Start]\n{p}\n[Reference Paragraph {i + 1} End]" for i, p in enumerate(input['p'][-1])])
    num_paragraphs = len(input['p'][-1])

    if model == 'gpt4':
        prompt_content = f"""[User's Question]{input['q'][-1]}\n[User's Question End]\n\n{paragraphs}\n\n[System]\nWhich reference paragraphs are helpful for answering the user's question? Please evaluate how helpful and relevant each of the {num_paragraphs} reference paragraphs is for answering the user's question above.\nThe evaluation criteria are the degree to which the reference paragraph is helpful, and the degree to which an answer to the question can be found in the reference paragraph. Please rate each reference paragraph with a number between 1 and 10. (Important): Give a score close to 10 for reference paragraphs that are very helpful for answering the question, and a score close to 1 for those that are not helpful at all. On the first line, output the scores of the {num_paragraphs} reference paragraphs separated by spaces. On the second line, provide a detailed explanation of your evaluation. Bias or evaluation order should not affect your evaluation."""
        prompt_dict = {
            'name': 'GetDPRAnswerGPT4',
            'prompt': prompt_content,
            'model': 'gpt4',
            'output': 'a',
            'temperature': 0.01
        }
    else:
        prompt_content = f"""[User's Question]{input['q'][-1]}\n\n{paragraphs}\n\n[System]\nWhich reference paragraphs are helpful for answering the user's question? Please evaluate how helpful and relevant each of the {num_paragraphs} reference paragraphs is for answering the user's question above.\nThe evaluation criteria are the degree to which the reference paragraph is helpful, and the degree to which an answer to the question can be found in the reference paragraph. Please rate each reference paragraph with a number between 1 and 10. (Important): Give a score close to 10 for reference paragraphs that are very helpful for answering the question, and a score close to 1 for those that are not helpful at all. On the first line, output the scores of the {num_paragraphs} reference paragraphs separated by spaces. On the second line, provide a detailed explanation of your evaluation. Bias or evaluation order should not affect your evaluation."""
        prompt_dict = {
            'name': 'GetDPRAnswer',
            'prompt': [{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": prompt_content}],
            'model': 'openai',
            'output': 'a',
            'temperature': 0.01
        }

    return prompt_dict

def get_answer_prompt(input: PipelineData) -> dict[str, Any]:
    paragraphs = '\n'.join([f"- {p}" for p in input['p'][-1]])

    prompt_dict = {
        'name': 'GetAnswer',
        'prompt': [{"role": "user", "content": f"""
                    User Question: {input['q'][-1]}
                    Paragraphs:
                    {paragraphs}

                    Based on the information in the given paragraphs, generate an answer to the user question above. Generate a specific answer of 4-5 sentences.
                    Answer:
                    """}],
        'model': 'openai',
        'output': 'a',
        'temperature': 1.0
    }

    return prompt_dict

def get_eval_prompt(input: PipelineData) -> dict[str, Any]:
    paragraphs = '\n'.join([f"[Reference Paragraph {i + 1} Start]\n{p}\n[Reference Paragraph {i + 1} End]" for i, p in enumerate(input['p'][-1])])
    prompt_dict = {
        'name': 'GetEval',
        'prompt': f"""[User's Prompt]\n{input['q'][-1]}\n[User's Prompt End]\n\n{paragraphs}\n\n[Assistant 1's Answer Start]\n{input['a'][-1]}\n\n[Assistant 1's Answer End]\n\n[System]\nPlease evaluate the performance of Assistant 1's answer.\nThe evaluation criteria are the degree to which the answer meets the requirements of the prompt, accuracy, clarity, and level of detail. Rate the assistant with an integer between 1 and 10. A higher number means better performance.\nOn the first line of the output, provide the AI assistant's score. On the second line, provide a detailed explanation of your evaluation. Bias or evaluation order should not affect your evaluation.""",
        'model': 'gpt4',
        'output': 'e',
        'temperature': 0.01
    }

    return prompt_dict

def get_next_q_prompt(input: PipelineData) -> dict[str, Any]:
    paragraph_text = '\n'.join([f"- {p}" for p in input['p'][-1]])
    prompt = []
    prompt.append({"role": "system", "content": f"You generate answers based on the information in the given paragraphs for the user's questions. Paragraphs:\n{paragraph_text}"})
    for i, q in enumerate(input['q']):
        prompt.append({"role": "user", "content": f"User's Question: {q}"})
        if i < len(input['q']) - 1:
            prompt.append({"role": "assistant", "content": f"{input['a'][i]}"})
        else:
            prompt.append({"role": "assistant", "content": f"{input['a'][i]} Do you have any additional questions?"})
    prompt.append({"role": "user", "content": "Please generate a follow-up question continuing the conversation above. Follow-up question:"})

    prompt_dict = {
        'name': 'GetNextQ',
        'prompt': prompt,
        'model': 'openai',
        'output': 'q',
        'temperature': 1.0
    }

    return prompt_dict
