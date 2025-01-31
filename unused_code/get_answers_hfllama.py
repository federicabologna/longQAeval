# Imports
import time
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import transformers

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float32,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    device_map="auto",
)
    
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

def get_response(_system_prompt, _user_prompt, max_tokens=300):
    
    messages = [{'role': 'system', 'content': _system_prompt}, {'role': 'user', 'content': _user_prompt}]
    
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id = pipeline.tokenizer.eos_token_id,
    )
    
    response = outputs[0]["generated_text"][len(prompt):]
    
    return response
    
    
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

    output_path = os.path.join(output_dir,  f'kqa_answers_llama_{number_shots}.jsonl')
    collected_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                dct = json.loads(line.strip())
                collected_ids.add(dct['id'])

    print(len(collected_ids))

    for d in sample_100[:10]:

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