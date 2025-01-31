
# Imports
import time
import os
import argparse
import json
import requests
import numpy as np
import pandas as pd

url = 'http://localhost:11434/api/pull'

headers = {
        'Content-Type': 'application/json'
    }

data = {
  "model": "llama3.1:405b"
}

response = requests.post(url, headers=headers, json=data, stream=True)
        
# Ensure the request was successful
if response.status_code == 200:
    # Iterate over the streamed lines
    for line in response.iter_lines(decode_unicode=True):
        if line:  # Avoid processing blank lines
            try:
                # Parse the JSON object
                json_obj = json.loads(line)
                # Print the JSON object
                print(json.dumps(json_obj, indent=2))
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")