# Imports
import time
import os
import argparse
import json
import numpy as np
import pandas as pd
from openai import AzureOpenAI


def get_response(_system_prompt, _user_prompt, _model='gpt-4-better', _max_tokens=300):

    client = AzureOpenAI(
        azure_endpoint = 'https://endoqa-cornell.openai.azure.com/',
        api_key= open(os.path.join('../apikeys', 'endoqa-key.txt')).read().strip(),
        api_version="2024-02-01"
    )

    attempts = 0
    max_attempts = 3
    response = 'None'

    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model= _model,
                max_tokens=_max_tokens,
                messages = [
                    {'role': 'system', 'content': _system_prompt},
                    {'role': 'user', 'content': _user_prompt}
                ]
            )
            
            _answer = response.choices[0].message.content.strip()

            if len(_answer) > 250:
                break

        except Exception as e:
            print(f'Error {e}. Sleeping 3 seconds ...')
            time.sleep(3)
            if attempts == max_attempts-1:
                _answer = f'Error {e}'

        attempts += 1
        if _model == 'gpt-4-better':
            time.sleep(2)
        else:
            time.sleep(0.5)

    return _answer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_shots', required=True, type=str)
    args = parser.parse_args()
    number_shots = args.n_shots

    # File locations
    dir = os.getcwd()
    data_dir = os.path.join(dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_dir = os.path.join(dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load 100 samples
    sample_100 = []
    with open(os.path.join(data_dir, 'sample_100.jsonl'), 'r') as f:
        for line in f:
            sample_100.append(json.loads(line.strip()))

    sys_prompt = 'You are a trained physician. Answer this question from a fellow clinician by providing correct, relevant, and safe information. Make sure to keep your answer under 270 words and do not hedge.\n\nAnswer:'
    #270 words because of medium length of K QA responses

    if number_shots == 'five':
        with open(os.path.join(data_dir, 'five_shot_prompt.txt'), "r") as file:
            sys_prompt = file.read()
    
    print(len(sample_100))
    print(sys_prompt)

    output_path = os.path.join(output_dir,  f'kqa_answers_gpt4_{number_shots}.jsonl')
    collected_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                dct = json.loads(line.strip())
                collected_ids.add(dct['id'])

    print(len(collected_ids))

    for d in sample_100:

        if d['id'] not in collected_ids:

            start = time.time()

            us_prompt = f"Question:\n{d['Question']}\n\nAnswer:"
            
            d['answer'] = get_response(sys_prompt, us_prompt)

            with open(output_path, 'a') as file:
                json.dump(d, file)
                file.write('\n') 

            end = time.time()  
            print(d['id'], end-start)


if __name__ == "__main__":
    main()