import os

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

# Approach using an agent

chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=( 
                "You are a helpful assistant, and you will be given a user request."
                "Some rules to follow:"
                "- Always answer the request using the given context information and not prior knowledge"
                "- If you cannot find the answer in the given context information, reply: 'Beklager, jeg kunne ikke finne svaret i den gitte konteksten.'."
                "- Always answer in norwegian"
                "- Always answer using the format: '\033[33mAnswer: \033[34m<answer>\033[0m'"

            ),
        )
        ,
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            ),
        ),
    ]
text_splitter = SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=75)
index:VectorStoreIndex = None

storage = './blobstorage/chatbot/ungnotobakk'
if os.path.exists(storage):
    # load the VectorStoreIndex 
    index = read_index_from_storage(storage)
else:
    print(f'Could not read index from {storage}')
    
text_qa_template =  ChatPromptTemplate(chat_text_qa_msgs)

response_synthesizer = get_response_synthesizer(
    response_mode= "tree_summarize",
    text_qa_template = text_qa_template,
    summary_template= text_qa_template, #definitly in use for response_mode = tree_summarize
    structured_answer_filtering=True, 
    verbose=True,
)

# Tool
query_engine = index.as_query_engine(
    similarity_cutoff= 0.7, 
    similarity_top_k=int(10),
    response_synthesizer=response_synthesizer)

def answer_query(query: str)->str:
    try: 
        print(f'\033[31mTool 5: answer_query \033[34m"{query}"\033[0m')
        response = query_engine.query(query)
        
        return str(response) 
    
    except Exception as e:
        print("Error article_query:", e)
        return f"Error article_query: {str(e)}"

answer_query_tool = FunctionTool.from_defaults(
    name="answer_query",
    fn=answer_query,
    description="Answer a user query about tobakk"
)
    


#def rewrite_answer(text: str, style_summary: str, style_examples: list) -> str:
def rewrite_answer(text: str, style_examples: list) -> str:
    global llm
    style_examples_str = "\n".join(style_examples)
    prompt = (
        f"Rewrite the following answer using the following style examples:\n"
        f"Style examples:\n{style_examples_str}\n\n"
        f"Original Answer:\n{text}"
    )
    messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
    response = llm.chat(messages)
    return response


rewrite_tool = FunctionTool.from_defaults(
    name="rewrite_tool",
    fn=lambda input: rewrite_answer(
        input, 
        [            
            "Hei!\n"
            "Det høres ut som du er veldig bekymret, og det er ikke noe godt! Fint at du tar kontakt med ung.no. Vi vet dessverre ikke prosentrisikoen for kreft av snus, men den er heldigvis lavere enn om du for eksempel røykte. Det er også veldig fint at du er fysisk aktiv og spiser sunt.\n"
            "Generelt kan vi si at jo mer du snuser, jo større sjans er det for å bli avhengig og få helseskader, men hvor mye som skal til, vil variere fra person til person.\n"
            "Jeg skjønner godt at du synes det er vanskelig å slutte, men det er mulig å få hjelp til dette. Helsestasjon for ungdom er gratis, anonym og de skal kunne gi råd og veiledning til å slutte. Du kan også lese artiklene under om tips til å slutte.\n"
            "Heldigvis har kroppen din en fantastisk evne til å reparere seg selv. Dersom du klarer å slutte nå, vil kroppen etter kort tid begynne å reparere eventuelle skader. Det kan kanskje være en god motivasjon til å klare det? Å bekymre seg for helsen er vondt, og det hjelper å snakke med noe om det. En god venn, en i familien du stoler på, legen din eller helsesykepleier - du kjenner selv hva som funker for deg.\n"
            "Ønsker deg alt godt!\n\n"
            "Vennlig hilsen\n\n"
            "Tobakksrådgiver i Helsedirektoratet\n"
        ]
    ),
    description="Rewrite an answer in the desired user style."
)


# Bygg agenten med verktøyene
agent = OpenAIAgent.from_tools(
    tools =  
        [answer_query_tool,
        rewrite_tool],
    llm=llm,
    verbose=True,
    system_prompt=  
        "You are a helpful assistant, and you will be given a user request."
        "Some rules to follow:"
        "- Always answer the request using the given context information and not prior knowledge"
        "- rewrite the answer using the user style"
        "- If you cannot find the answer in the given context information,"
        " reply: 'Beklager, jeg kunne ikke finne svaret i den gitte konteksten.'."
)

query = "Hei, jeg er usikker på hvordan jeg skal klare å slutte med snus, kan dere hjelpe meg med noen tips og råd?"
agent_response = agent.query(query)
print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
