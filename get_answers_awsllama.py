# Imports
import time
import os
import argparse
import json
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError


{
 "modelId": "meta.llama3-1-405b-instruct-v1:0",
 "contentType": "application/json",
 "accept": "application/json",
 "body": "{\"prompt\":\"this is where you place your input text\",\"max_gen_len\":512,\"temperature\":0.5,\"top_p\":0.9}"
}


def get_response(_system_prompt, _user_prompt, _max_tokens=300):
    
    attempts = 0
    max_attempts = 3
    response = 'None'

    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime",
                          region_name="us-east-2",
                          aws_access_key_id=open(os.path.join('../../PhD/apikeys', 'aws_access_key_id.txt')).read().strip(),
                          aws_secret_access_key=open(os.path.join('../../PhD/apikeys', 'aws_secret_access_key.txt')).read().strip())
    
    # Set the model ID
    model_id = "us.meta.llama3-1-405b-instruct-v1:0"

    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {_system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {_user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": _max_tokens,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    while attempts < max_attempts:
        try:
            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_id, body=request)
            
            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            _answer = model_response["generation"].strip()
            
            if len(_answer) > 250:
                break

        except (ClientError, Exception) as e:
            print(f'Error {e}. Sleeping 3 seconds ...')
            time.sleep(3)
            if attempts == max_attempts-1:
                _answer = f'Error {e}'
        
        attempts += 1

    return _answer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_shots', required=True, type=str)
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()
    number_shots = args.n_shots
    input_file = args.input
    output_file = args.output

    # File locations
    dir = os.getcwd()
    data_dir = os.path.join(dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_dir = os.path.join(dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questions
    questions = []
    # with open(os.path.join(data_dir, 'sample_100.jsonl'), 'r') as f:
    with open(os.path.join(data_dir, input_file), 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    sys_prompt = 'You are a trained physician. Answer this question from a fellow clinician by providing correct, relevant, and safe information. Make sure to keep your answer under 270 words and do not hedge.\n\nAnswer:'
    #270 words because of medium length of K QA responses

    if number_shots == 'five':
        with open(os.path.join(data_dir, 'five_shot_prompt.txt'), "r") as file:
            sys_prompt = file.read()
    
    print(len(questions))
    print(sys_prompt)

    # output_path = os.path.join(output_dir,  f'kqa_answers_llama_{number_shots}.jsonl')
    output_path = os.path.join(output_dir,  output_file)
    collected_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                dct = json.loads(line.strip())
                collected_ids.add(dct['id'])

    print(len(collected_ids))

    for d in questions:

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