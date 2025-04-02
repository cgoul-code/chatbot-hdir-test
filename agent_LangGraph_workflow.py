import os, getpass

# LLM og verktøy
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core import (get_response_synthesizer)
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (VectorStoreIndex, StorageContext,  load_index_from_storage)

# Import av embedding-moduler
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType

# Indeksverktøy
LLMGPT4omini = AzureOpenAI(
    model=os.getenv('AZURE_OPENAI_MODEL_GPT4omini'),
    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_GPT4omini'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_GPT4omini'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY_GPT4omini'),
    azure_endpoint=os.getenv('AZURE_OPENAI_AZURE_ENDPOINT_GPT4omini'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSJON_GPT4omini'),
    temperature=0.0,
    timeout= 120,
)

def read_index_from_storage(storage):
    storage_context = StorageContext.from_defaults(persist_dir=storage)
    return load_index_from_storage(storage_context)

# Sett Azure OpenAI-legitimasjon

llm = LLMGPT4omini

embed_model = AzureOpenAIEmbedding(
    model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
    api_key=os.getenv('AZURE_OPENAI_EMBEDDINGS_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_EMBEDDINGS_API_VERSJON'),
    azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_AZURE_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'),  
)

from typing_extensions import TypedDict

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str
   
def generate_joke(state: State) :
    messages = [
        ChatMessage(role="system", content="You are a helpfull assistant" ),
        ChatMessage(role="user", content=f"Write a short joke about {state['topic']}")  ]
    response = llm.chat(messages)
    print(f'Response generate_joke: {response.message.content}')

    return {'joke': response.message.content}

def improve_joke(state: State) :
    messages = [
        ChatMessage(role="system", content="You are a helpfull assistant" ),
        ChatMessage(role="user", content=f"Make that joke funnier by adding wordplay: {state['joke']}")  ]
    response = llm.chat(messages)
    print(f'Response improve_joke: {response.message.content}')

    return {'improved_joke': response.message.content}

def polish_joke(state: State) :
    messages = [
        ChatMessage(role="system", content="You are a helpfull assistant" ),
        ChatMessage(role="user", content=f"Make a surprising twist to this joke: {state['improved_joke']}")  ]
    response = llm.chat(messages)
    print(f'Response final_joke: {response.message.content}')

    return {'final_joke': response.message.content}

def check_punchline(state: State):
    if "?" in state['joke'] or "!" in state['joke']:
        return "Pass"
    else:
        return "Fail"
    
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

#build workflow
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Pass": "improve_joke", "Fail": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile
chain = workflow.compile()

# Show workflow
#display(Image(chain.get_graph().draw_mermaid_png()))

state = chain.invoke({"topic": "cats"})


