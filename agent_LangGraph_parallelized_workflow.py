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

from typing_extensions import TypedDict

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str
   
def call_llm_1(state: State) :
    messages = [
        ChatMessage(role="system", content="You are a helpfull assistant" ),
        ChatMessage(role="user", content=f"Write a short joke about {state['topic']}")  ]
    response = llm.chat(messages)
    print(f'Response call_llm_1: {response.message.content}')

    return {'joke': response.message.content}

def call_llm_2(state: State) :
    messages = [
        ChatMessage(role="system", content="You are a helpfull assistant" ),
        ChatMessage(role="user", content=f"Write a story about: {state['topic']}")  ]
    response = llm.chat(messages)
    print(f'Response call_llm_2: {response.message.content}')

    return {'story': response.message.content}

def call_llm_3(state: State) :
    messages = [
        ChatMessage(role="system", content="You are a helpfull assistant" ),
        ChatMessage(role="user", content=f"Write a poem about: {state['topic']}")  ]
    response = llm.chat(messages)
    print(f'Response call_llm_3: {response.message.content}')

    return {'poem': response.message.content}


def aggregator(state: State):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


    
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
#display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])