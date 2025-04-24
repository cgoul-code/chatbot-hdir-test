from huggingface_hub import list_models
import requests, os

# 1. List models under the "meta-llama" namespace
print("Available meta-llama models:")
models = list_models(filter="meta-llama")
for model in models:
    print(model)  # prints the model ID
    print('\n\n')

print("\n---\n")

os.getenv('HUGGING_FACE_AUTHORISATION')

# Replace the API URL and token below as needed.
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-hf" 
bearer = os.getenv('HUGGING_FACE_AUTHORISATION')
headers = {
    # Note the use of the "Bearer" prefix.
    "Authorization": bearer
}

payload = {
    "inputs": "Explain quantum physics like I'm five.",
    "parameters": {"temperature": 0.7, "max_new_tokens": 100}
}

print("Making an inference API call...")
response = requests.post(API_URL, headers=headers, json=payload)

# Check if the response was successful
if response.status_code == 200:
    print("Response from model:")
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}:")
    print(response.json())

