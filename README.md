# Project Overview

This project comprises multiple scripts that demonstrate various agent use cases. Each script can be run independently. Some scripts use LlamaIndex to showcase the OpenAIAgent and its tools, while others utilize LangGraph to construct structured workflows (inspired from the tutorial from  Lance from Lang chain and thropic: https://langchain-ai.github.io/langgraph/tutorials/workflows/ and video: https://www.youtube.com/watch?v=aHCDrAbH_go).

## Setup Instructions

- Make a copy of the project.
- Install the libraries: `pip install -r requirements.txt`
- Extract the zip files located in the `blobstorage/chatbot/` folder.

## Azure OpenAI Usage

If your code utilizes AzureOpenAI, you'll need an Azure subscription. Create a `.env` file in the project folder with the following environment variables to initialize the model:
Example for `.env`:
AZURE_OPENAI_MODEL_GPT4omini=<ModelName>
AZURE_OPENAI_DEPLOYMENT_NAME_GPT4omini=<DeploymentName>
AZURE_OPENAI_API_KEY_GPT4omini=<APIKey>
AZURE_OPENAI_AZURE_ENDPOINT_GPT4omini=<AzureEndpoint>
AZURE_OPENAI_API_VERSJON_GPT4omini=<APIVersion>


# Some utils

## requirements.txt
pip freeze >> requirements.txt
pip install -r requirements.txt
remember to remove pywin32==306 from requirements.txt (for deploy to Azure environement)

## Tips for reinstall llama-index:
pip install llama-index --upgrade --no-cache-dir --force-reinstall 
pip check    

## Tips for editing your github config
git config --edit






