
def get_dpr_answer_prompt(input):
    paragraphs = '\n'.join([f"[참고 문단 {i + 1}의 시작]\n{p}\n[참고 문단 {i + 1}의 끝]" for i, p in enumerate(input['p'][-1])])
    prompt_dict = {
        'name' : 'GetDPRAnswer',
        'prompt' : [{"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
                    {"role": "user", "content": f"""[사용자의 질문]{input['q'][-1]}\n\n{paragraphs}[참고 문단 1의 시작]\n{input['p'][-1][0]}\n\n[참고 문단 1의 끝]\n\n[참고 문단 2의 시작]\n{input['p'][-1][1]}\n\n[참고 문단 2의 끝]\n\n[참고 문단 3의 시작]\n{input['p'][-1][2]}\n\n[참고 문단 3의 끝]\n\n[시스템]\n어떤 참고문단이 사용자의 질문에 답하기 도움이 되나요? 위에 나온 사용자의 질문에 답하기 위해 세 개의 참고 문단이 각각 얼마나 도움이 되고 연관성이 있는지 평가해주세요. \n평가항목은 참고 문단이 도움되는 정도, 참고 문단에서 질문에 대한 대답을 찾아낼 수 있는 정도입니다. 1에서 10사이의 숫자로 각각의 참고 문단을 평가해주세요. (중요): 질문에 대답을 하기에 매우 도움이 되는 참고 문단은 10점에 가깝게, 전혀 도움되지 않는 참고 문단은 1점에 가깝게 평가점수를 주세요. 첫 줄에는 세 개의 참고 문단의 점수를 각각 띄어서 출력해주세요. 두번째 줄에는 당신의 평가에 대한 세부적인 설명을 주세요. 편견이나 평가의 순서가 당신의 평가에 영향을 주어서는 안됩니다."""}],
        'model': 'openai',
        'output': 'a',
        'temperature': 0.01
    }
    
    return prompt_dict

def get_dpr_answer_prompt_gpt4(input):
    paragraphs = '\n'.join([f"[참고 문단 {i + 1}의 시작]\n{p}\n[참고 문단 {i + 1}의 끝]" for i, p in enumerate(input['p'][-1])])
    prompt_dict = {
        'name' : 'GetDPRAnswerGPT4',
        'prompt' : f"""[사용자의 질문]{input['q'][-1]}\n[사용자의 질문 끝]\n\n{paragraphs}\n\n[참고 문단 1의 시작]\n{input['p'][-1][0]}\n\n[참고 문단 1의 끝]\n\n[참고 문단 2의 시작]\n{input['p'][-1][1]}\n\n[참고 문단 2의 끝]\n\n[참고 문단 3의 시작]\n{input['p'][-1][2]}\n\n[참고 문단 3의 끝]\n\n[시스템]\n어떤 참고문단이 사용자의 질문에 답하기 도움이 되나요? 위에 나온 사용자의 질문에 답하기 위해 세 개의 참고 문단이 각각 얼마나 도움이 되고 연관성이 있는지 평가해주세요. \n평가항목은 참고 문단이 도움되는 정도, 참고 문단에서 질문에 대한 대답을 찾아낼 수 있는 정도입니다. 1에서 10사이의 숫자로 각각의 참고 문단을 평가해주세요. (중요): 질문에 대답을 하기에 매우 도움이 되는 참고 문단은 10점에 가깝게, 전혀 도움되지 않는 참고 문단은 1점에 가깝게 평가점수를 주세요. 첫 줄에는 세 개의 참고 문단의 점수를 각각 띄어서 출력해주세요. 두번째 줄에는 당신의 평가에 대한 세부적인 설명을 주세요. 편견이나 평가의 순서가 당신의 평가에 영향을 주어서는 안됩니다.""",
        'model': 'gpt4',
        'output': 'a',
        'temperature': 0.01
    }
    
    return prompt_dict

def get_answer_prompt(input):
    
    prompt_dict = {
        'name' : 'GetAnswer',
        'prompt' : [{"role": "user", "content": f"""
                    사용자 질문: {input['q'][-1]}
                    단락:
                    - {input['p'][-1][0]}
                    - {input['p'][-1][1]}
                    - {input['p'][-1][2]}

                    위 사용자 질문에 대해 주어진 단락에 있는 정보에 기반하여 답변을 생성해주세요. 답변은 4~5 문장 길이의 한국어로 구체적으로 생성해주세요.
                    답변:
                    """}],
        'model': 'openai',
        'output': 'a',
        'temperature': 1.0
    }
    
    return prompt_dict

def get_eval_prompt(input):
    paragraphs = '\n'.join([f"[참고 문단 {i + 1}의 시작]\n{p}\n[참고 문단 {i + 1}의 끝]" for i, p in enumerate(input['p'][-1])]) 
    prompt_dict = {
        'name' : 'GetEval',
        'prompt' : f"""[사용자의 프롬프트]\n{input['q'][-1]}\n[사용자의 프롬프트 끝]\n\n{paragraphs}\n\n[참고 문단 1의 시작]\n{input['p'][-1][0]}\n[참고 문단 1의 끝]\n[참고 문단 2의 시작]\n{input['p'][-1][1]}\n[참고 문단 2의 끝]\n[참고 문단 3의 시작]\n{input['p'][-1][2]}\n[참고 문단 3의 끝]\n\n[어시스턴트 1의 대답 시작]\n{input['a'][-1]}\n\n[어시스턴트 1의 대답 끝]\n\n[시스템]\n어시스턴트 1의 대답 성능을 평가해줘. \n평가 항목은 대답이 프롬프트에서 요구하는 사항에 적합한 정도, 정확도, 명확성 그리고 대답의 디테일이야. 1 에서 10 사이의 정수로 어시스턴트를 평가해줘. 높은 숫자일 수록 성능이 좋은거야. \n출력하는 결과의 첫 줄에는 반드시 AI 어시스턴트의 점수를 나타내줘. 두번째 줄에는 너의 평가에 대한 세부적인 설명을 줘. 편견이나 평가 순서가 너의 평가에 영향을 주어서는 안돼.""",
        'model': 'gpt4',
        'output': 'e',
        'temperature': 0.01
    }
    
    return prompt_dict

def get_next_q_prompt(input):
    prompt = []
    prompt.append({"role": "system", "content": f"당신은 사용자의 질문에 대해 주어진 단락에 있는 정보에 기반하여 답변을 생성해줍니다. 단락: - {input['p'][-1][0]}\n- {input['p'][-1][1]}\n- {input['p'][-1][2]}"})
    for i, q in enumerate(input['q']):
        prompt.append({"role": "user", "content": f"사용자의 질문: {q}"})
        if i < len(input['q']) -1:
            prompt.append({"role": "assistant", "content": f"{input['a'][i]}"})
        else:
            prompt.append({"role": "assistant", "content": f"{input['a'][i]} 추가로 궁금하신 점이 있으신가요?"})
    prompt.append({"role": "user", "content":"위 대화에 이어지는 추가 질문을 생성해주세요. 추가 질문:"})
        
    prompt_dict = {
        'name' : 'GetNextQ',
        'prompt' : prompt,
        'model': 'openai',
        'output': 'q',
        'temperature': 1.0
    }
    
    return prompt_dict

# not used
def _get_eval_prompt(input): 
    prompt_dict = {
        'name' : 'GetBardEval',
        'prompt' : f"""
                    사용자 질문: {input['q'][-1]}
                    단락:
                    - {input['p'][-1][0]}
                    - {input['p'][-1][1]}
                    - {input['p'][-1][2]}
                    AI assistant 의 답변 : {input['a'][-1]}
                    위 사용자의 질문에 대해 주어진 단락에 있는 정보를 기반으로 생성된 AI assistant 의 답변이 적절한가요?
                    """,
        'model': 'bard',
        'output': 'e'
    }
    
    return prompt_dict
