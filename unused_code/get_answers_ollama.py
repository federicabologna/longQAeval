# Imports
import time
import os
import argparse
import json
import requests
import numpy as np
import pandas as pd
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_response(_system_prompt, _user_prompt, _model='llama3.', _max_tokens=300):

    data = {
        "model": _model,
        "messages": [
            {"role": "system","content": _system_prompt},
            {"role": "user","content": _user_prompt}
            ],
        "stream": False
    }
    
    headers = {
        'Content-Type': 'application/json'
    }

    attempts = 0
    max_attempts = 3
    response = 'None'

    while attempts < max_attempts:
        try:
            response = requests.post('http://localhost:11434/api/chat', headers=headers, json=data)
          
            _answer = response.json()['message']['content']

            if 'Error' not in _answer and len(_answer) > 250:
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

    # Get questions
    data = []
    with open(os.path.join(data_dir, 'kqa_questions.jsonl'), 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    prompt = 'You are a trained physician. Answer this question from a fellow clinician by providing correct, relevant, and safe information. Make sure to keep your answer under 270 words and do not hedge.\n\nAnswer:'
    #270 words because of medium length of K QA responses

    if number_shots == 'five':
        prompt = f'''You are a trained physician. Answer this question from a fellow clinician by providing correct, relevant, and safe information. Make sure to keep your answer under 270 words and do not hedge. Follow these steps:

        Read the question.
        Find medical information that is correct, relevant, and safe given the question asked.
        Formulate your answer in less than 270 words based on the information you have found, and do not hedge!

        Here are five example of correct, relevant, and safe answers to clinical questions:

        Example 1:
        Question: {data[0]['Question']}
        Answer: {data[0]['Free_form_answer']}

        Example 2:
        Question: {data[1]['Question']}
        Answer: {data[1]['Free_form_answer']}
        
        Example 3:
        Question: {data[2]['Question']}
        Answer: {data[2]['Free_form_answer']}

        Example 4:
        Question: {data[3]['Question']}
        Answer: {data[3]['Free_form_answer']}

        Example 5:
        Question: {data[4]['Question']}
        Answer: {data[4]['Free_form_answer']}

        Answer:
        '''
        data = data[5:]
    
    print(len(data))
    print(prompt)

    output_path = os.path.join(output_dir,  f'kqa_answers_llama_{number_shots}.jsonl')
    collected_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                dct = json.loads(line.strip())
                collected_ids.add(dct['id'])

    print(len(collected_ids))

    for d in data[:10]:

        if d['id'] not in collected_ids:

            start = time.time()

            d['answer'] = get_response(prompt, d['Question'])

            with open(output_path, 'a') as file:
                json.dump(d, file)
                file.write('\n') 

            end = time.time()  
            print(d['id'], end-start)


if __name__ == "__main__":
    main()