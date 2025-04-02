import os
import json
import re
import requests
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
from llama_index.readers.apify import ApifyActor


# LLM og verktøy
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core import (get_response_synthesizer)
from llama_index.core import VectorStoreIndex, StorageContext
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

llm = AzureOpenAI(
    azure_endpoint="https://HelseSvar.openai.azure.com",  
    deployment_name="gpt-4o-mini",  
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2023-12-01-preview",
    temperature=0.0
)

llm = LLMGPT4omini

embed_model = AzureOpenAIEmbedding(
    model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
    api_key=os.getenv('AZURE_OPENAI_EMBEDDINGS_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_EMBEDDINGS_API_VERSJON'),
    azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_AZURE_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'),  
)

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
#                "Given the context information and not prior knowledge, "
                "Query: {query_str}\n"
                "Answer: "
            ),
        ),
    ]

text_splitter = SentenceSplitter.from_defaults(chunk_size=4096, chunk_overlap=75)
#text_splitter = SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=75)


# Eksempel på dokumentdata (simulerte artikler fra HelseNorge)
all_helsenorge_documents_list = []
filtred_documents_list = []
index:VectorStoreIndex = None

# Functions ===========================
def translate_text_apertium(text, source_lang, target_lang):
    url = "https://www.apertium.org/apy/translate"
    params = {
        "q": text,
        "langpair": f"{source_lang}|{target_lang}"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'responseData' in data and 'translatedText' in data['responseData']:
            return data['responseData']['translatedText']
        else:
            return "Translation failed."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    
def load_configuration(item, config_file, start_urls_file):
    # Load the main configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        #print(f'config:{config}')
    
    # Load the startUrls
    with open(start_urls_file, 'r', encoding='utf-8') as f:
        start_urls = json.load(f)
        #print(f'start_urls: {start_urls}')
    
    # Assign the startUrls to the config
    config[item]['startUrls'] = start_urls
    
    return config
    
def documents_from_startUrls(item):
    reader = ApifyActor(os.getenv('APIFY_KEY'))
    documents = []
    print(f'documents_from_startUrls, item:{item}')
    
    run_input = None
    if (item in ['psykiskhelse',
                 'ungnospmpsykiskhelse',
                 'ungnotobakk',
                 'ungnospmtobakk',
                 'ungnospm', 
                 'prevensjon',
                 'ungnospmprevensjon'
                 'psassa',    
                 'hvaerinnafor', 
                 'urinsyregikt',
                 'normen', 
                 'kiregelverkutv', 
                 'ledelseogkvalitet',
                 'hvilkekiregelverk',
                 'pasientbrukerrettighetsloven',
                 'rapportbrukavki',
                 'helsenorgeartikler']):

        run_input_params = load_configuration(item, f'./Scraping/{item}/config.json', f'./Scraping/{item}/startUrls.json')
        print(f'params:<{run_input_params}>')
        run_input = run_input_params.get(item, {})
        print(f'run_input:<{run_input}>')


    if not run_input:
        print("Run input not found!")
        return documents
    
def build_context(user_input=None):
    try:
        print('Build context tool')
        
        """
        Returnerer en JSON-streng med en liste over dokumenter.
        """
        global all_helsenorge_documents_list
        global filtred_documents_list
        global index
        all_helsenorge_documents_list = []  # ensure this global list is defined
        name = 'helsenorgeartikler'
        storage = './blobstorage/chatbot/helsenorgeartikler'
        
        # Define the path for the embedding store file.
        embedding_store_path = os.path.join(storage, "embedding_store.pkl")
        
        # Initialize the embedding store mapping.
        embedding_store = {}
        
        
        if os.path.exists(storage):
            print('Reading documents from storage')
            # Load embedding store if the file exists.
            if os.path.exists(embedding_store_path):
                with open(embedding_store_path, "rb") as f:
                    embedding_store = pickle.load(f)
                        
            for current_doc_id, doc_metadata in embedding_store.items():
                all_helsenorge_documents_list.append({
                    "id": current_doc_id,
                    "url": doc_metadata.get("url", ""),
                    "title": doc_metadata.get("title", ""),
                    "document_title": doc_metadata.get("document_title", ""),  # if available in the store
                    "description": doc_metadata.get("description", ""),
                    "questions": doc_metadata.get("questions_this_excerpt_can_answer", ""),
                    "keywords": doc_metadata.get("excerpt_keywords", ""),
                    "keywords_embedding": doc_metadata.get("keywords_embedding", None),
                    "summary": doc_metadata.get("section_summary", ""),
                    "updated": doc_metadata.get("updated", "")
                })
                filtred_documents_list.append({
                    "id": current_doc_id,
                    "url": doc_metadata.get("url", ""),
                    "title": doc_metadata.get("title", ""),
                    "document_title": doc_metadata.get("document_title", ""),  # if available in the store
                    "description": doc_metadata.get("description", ""),
                    "questions": doc_metadata.get("questions_this_excerpt_can_answer", ""),
                    "keywords": doc_metadata.get("excerpt_keywords", ""),
                    "summary": doc_metadata.get("section_summary", ""),
                    "updated": doc_metadata.get("updated", "")
                })
            # load the VectorStoreIndex 
            index = read_index_from_storage(storage)
            
        else:
            print('Reading pages from Helsenorge.no')
            # Process documents if storage does not yet exist.
            docs = documents_from_startUrls('helsenorgeartikler')
            print('Build_context, number of documents', len(docs))
            
            # Create the vectorIndex
            storage_context = StorageContext.from_defaults()
            nodes = text_splitter.get_nodes_from_documents(docs)
            index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
            storage_context.persist(persist_dir=storage)
            
            # iterate the documents in index
            doc_list = []
            print(f'Len:{len(index.docstore.docs.items())}')
            
    
            for doc_id, doc in index.docstore.docs.items():
                print("Document ID:", doc_id)
                print("Document content:", doc)
            
            
            # Prepare a list for index creation.
            # doc_list = []
            # for idx, doc in enumerate(docs):
                
                doc_list.append(doc)
                
                
                from llama_index.core.extractors import (
                    SummaryExtractor,
                    QuestionsAnsweredExtractor,
                    TitleExtractor,
                    KeywordExtractor
                )
                
                transformations = [
                    TitleExtractor(nodes=5, 
                        node_template=
                            "\Context: {context_str}. Give a title in norwegian that summarizes all of the unique entities, titles or themes found in the context."
                            "\nProvide the title in sentence case, meaning only the first word of each sentence and proper nouns should be capitalized.",
                        combine_template= 
                            "\Context: {context_str}. Give a title in norwegian that summarizes all of the unique entities, titles or themes found in the context."
                            "\nProvide the title in sentence case, meaning only the first word of each sentence and proper nouns should be capitalized." ),
                    QuestionsAnsweredExtractor(
                        questions=3, 
                        prompt_template=(
                            "Here is the context: {context_str}"
                            "\n\nGiven the contextual information, "
                            "\ngenerate {num_questions} questions in norwegian this context can provide specific answers to which are unlikely to be found elsewhere."
                            "\n\nHigher-level summaries in norwegian of surrounding context may be provided \nas well. "
                            "Try using these summaries to generate better questions that this context can answer."
                            "\n\n Provide questions in the following format: '<questions>...'"
                        )
                    ),
                    SummaryExtractor(
                        summaries=["prev", "self"], 
                        prompt_template=(
                            "Here is the content of the section:"
                            "\n{context_str}"
                            "\n\nSummarize the key topics and entities of the section in norwegian. "
                            "\nDo not use markdowns"
                            "\nDo not introduce the summary with sentences as 'Nøkkeltemaene i avsnittet inkluderer informasjon om...'"
                        )
                    ),
                    KeywordExtractor(
                        keywords=10, 
                        prompt_template=(
                            "Some text is provided below. Given the text, extract up to {max_keywords} keywords from the text. Avoid stopwords."
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Provide keywords in the following JSON format: {\"keywords\":[{\"keyword\": \"keyword name\"\"},...]}"
                        )
                    ),
                ]
                
                from llama_index.core.ingestion import IngestionPipeline
                pipeline = IngestionPipeline(transformations=transformations)
                print(f"Ingestion of doc: {doc.metadata}")
                new_nodes = pipeline.run(documents=[doc])
                for id, new_node in enumerate(new_nodes):
                    
                    # Get the keywords string from metadata.
                    keywords_str = new_node.metadata.get("excerpt_keywords", "")

                    # Extract the JSON content from the markdown code fences.
                    match = re.search(r"```json\s*(\{.*\})\s*```", keywords_str, re.DOTALL)
                    if match:
                        json_content = match.group(1)
                        try:
                            keywords_dict = json.loads(json_content)
                        except json.JSONDecodeError:
                            keywords_dict = {}
                    else:
                        # Fallback: try to load the string directly.
                        try:
                            keywords_dict = json.loads(keywords_str)
                        except json.JSONDecodeError:
                            keywords_dict = {}

                    # Compute an embedding for each keyword.
                    keyword_embeddings = {}
                    keywords_list = []
                    for kw_item in keywords_dict.get("keywords", []):
                        keyword = kw_item.get("keyword", "").strip()
                        if keyword:
                            keyword_embeddings[keyword] = get_embedding(keyword)
                            keywords_list.append(keyword)

                    # Save the embeddings to your embedding store.
                    #embedding_store[current_doc_id] = keyword_embeddings
                      # Create a comma separated string of the keywords.
                    comma_separated_keywords = ", ".join(keywords_list)
                    
                    embedding_store[doc_id] = {
                        "keywords_embedding": keyword_embeddings,
                        "url": new_node.metadata.get("url", ""),
                        "title": new_node.metadata.get("title", ""),
                        "document_title": new_node.metadata.get("document_title", ""),
                        "description": new_node.metadata.get("description", ""),
                        "questions_this_excerpt_can_answer": new_node.metadata.get("questions_this_excerpt_can_answer", ""),
                        "excerpt_keywords": comma_separated_keywords,
                        "section_summary": new_node.metadata.get("section_summary", ""),
                        "updated": new_node.metadata.get("updated", "")
                    }
                    
                    # Update the document's metadata with new values.
                    doc.metadata.update({
                        "url": new_node.metadata.get("url"),
                        "title": new_node.metadata.get("title"),
                        "document_title": new_node.metadata.get("document_title", ""),
                        "description": new_node.metadata.get("description"),
                        "questions_this_excerpt_can_answer": new_node.metadata.get("questions_this_excerpt_can_answer"),
                        "excerpt_keywords": comma_separated_keywords,
                        "section_summary": new_node.metadata.get("section_summary"),
                        "updated": new_node.metadata.get("updated"),
                    })
                    # # remove text from doc, we do not need et any more
                    # doc.text=""
                    
                    print(f'\n\nAdding Doc metadata: {doc.metadata}\n\n')
            
            #print(f'\n\n\033[31mtool build_contexts {all_helsenorge_documents_list}\033[0m\n\n')
            
            # # Create and persist the index.
            # storage_context = StorageContext.from_defaults()
            # nodes = text_splitter.get_nodes_from_documents(doc_list)
            # index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
            # storage_context.persist(persist_dir=storage)
            
            # Persist the embedding store mapping.
            if not os.path.exists(storage):
                os.makedirs(storage)
            with open(embedding_store_path, "wb") as f:
                pickle.dump(embedding_store, f)
        
        #print(f'\n\n\033[31mtool build_contexts {all_helsenorge_documents_list}\033[0m\n\n')
        
        return "build_context success"

    except Exception as e:
        print("Error while building context:", e)
        return json.dumps({"error": str(e)})


# Generer embeddings for alle kjente matvarer
def get_embedding(text: str) -> np.ndarray:
    """Hent en embedding-vektor ved å bruke llama_index sin AzureOpenAIEmbedding."""
    embedding = embed_model.get_text_embedding(text)
    return np.array(embedding)



# Helper function: compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # Prevent division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# TOOLS ===================================

# Tool 1: Hent relevante dokumenter basert på en brukerforespørsel
def filter_relevant_articles(query: str)->str:
    print(f'\n\033[31mTool 1: filter_relevant_documents with query:\033[34m"{query}"\033[0m')
       
    keyword_list = []                                                  
    try:
        global filtred_documents_list
        global all_helsenorge_documents_list
        
        # Compute the embedding for the input keyword/query string.
        query_embedding = get_embedding(query)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        query_embedding = query_embedding.flatten()
        
        filtred_documents_list = []
        # For cosine distance, lower values indicate higher similarity.
        # Set a threshold value (e.g., 0.20 means only documents with a distance <= 0.20 will match).
        threshold = 0.10
        
        print('Antall all_helsenorge_documents_list:', len(all_helsenorge_documents_list))
        for doc in all_helsenorge_documents_list:
            keywords_emb = doc.get("keywords_embedding")
            if not keywords_emb:
                print('Alert: Embeddings are missing for document:', doc.get("id"))
                continue

            # Compute the minimum cosine distance across all keyword embeddings for the document.
            min_distance = 1.0  # The maximum possible cosine distance is 1.
           
            for keyword, emb in keywords_emb.items():
                if not isinstance(emb, np.ndarray):
                    emb = np.array(emb)
                emb = emb.flatten()
                
                distance = cosine(query_embedding, emb)
                if distance < min_distance:
                    min_distance = distance
                    min_keyword = keyword
                        
            if min_distance <= threshold:
                filtred_documents_list.append(doc)
                #print(f"Dokument {doc['id']} - found mach with key: {min_keyword}, Minimum cosine distance: {min_distance}")
                if min_keyword not in keyword_list:
                    keyword_list.append(min_keyword)
                
        if not filtred_documents_list:
            return "No relevant documents found."
        
        # # Remove keyword embeddings from the final output.
        # for doc in filtred_documents_list:
        #     del doc["keywords_embedding"]
            
        return f"\nTool filter_relevant_documents found {len(filtred_documents_list)} with matching keywords {keyword_list}"
    
    except Exception as e:
        print("Error while filtering relevant documents:", e)
        return json.dumps({"error": str(e)})

filter_relevant_articles_tool = FunctionTool.from_defaults(
    name="filter_relevant_articles",
    fn=filter_relevant_articles,
    description="Filter the global all_helsenorge_documents_list list using keywords, an return a list of relevant documents to be used for the create_excel_file tool"

)

# Tool 2: Lag en Excel-fil med informasjon over dokumentene
def create_excel_file(query=None):
    global filtred_documents_list
    print(f'\n\033[31mTool 2: create_excel_file')
    #print(f'Filtered documents: {filtred_documents_list}')
    
    try:
        # Create a pandas DataFrame directly from the filtered documents list.
        df = pd.DataFrame(filtred_documents_list)
        file_name = "documents_report.xlsx"
        df.to_excel(file_name, index=False)
        return f"Excel-fil '{file_name}' er opprettet med {len(filtred_documents_list)} dokument(er)."
    except Exception as e:
        print("Error creating Excel file:", e)
        return f"Error creating Excel file: {str(e)}"

create_excel_tool = FunctionTool.from_defaults(
    name="create_excel_file",
    fn=create_excel_file,
    description="Create an Excel file with information of the filtred documents."
)

# Tool 3: Oversett et svar til nynorsk
def translate_answer_to_nynorsk(query:str)->str:
    print(f'\n\033[31mTool 3: translate_answer_to_nynorsk with query:\033[34m"{query}"\033[0m')
    translated_str =""
    try:        
        translated_str = translate_text_apertium(query, "nob", "nno")
        #print('\ntranslated_str=', translated_str)
        return translated_str
    
    except Exception as e:
        print("Error translate_answer_to_nynorsk:", e)
        return f"Error translate_answer_to_nynorsk: {str(e)}"
    
translate_to_nynorsk_tool = FunctionTool.from_defaults(
    name="translate_answer_to_nynorsk",
    fn=translate_answer_to_nynorsk,
    description="Translate an input to nynorsk."
)

def produce_wordcloud(query: str) -> str:
    print(f'\n\033[31mTool 4: produce_wordcloud with query:\033[34m"{query}"\033[0m')
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import nltk
    # Download the stopwords if not already available
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords

    # Get the Norwegian stopwords from nltk
    norwegian_stopwords = set(nltk_stopwords.words('norwegian'))
    # Optionally, add any additional common Norwegian words you wish to exclude:
    # norwegian_stopwords.update(['og', 'i', 'det', 'som'])
    
    norwegian_stopwords.update(['får', 'få', 'bør', 'dersom', 'ta'])

    # Combine the default English stopwords with the Norwegian ones
    combined_stopwords = set(STOPWORDS).union(norwegian_stopwords)

    # Accumulate text from all documents
    text = ""
    for doc in filtred_documents_list:
        doc_id = doc.get('id')
        document = index.docstore.get_document(doc_id)
        #print(document)
        text += document.text

    # Generate the word cloud with the combined stopwords
    ordsky = WordCloud(width=800, height=400, background_color='white', stopwords=combined_stopwords).generate(text)

    # Display the word cloud
    plt.figure(figsize=(15, 8))
    plt.imshow(ordsky, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    return "produce_wordcloud succeded"

produce_wordcloud_tool = FunctionTool.from_defaults(
    name="produce_wordcloud",
    fn=produce_wordcloud,
    description="Produce a wordcould from a list of relevant documents"
)
# Function to categorize LIX score
def categorize_lix(lix):
    if lix < 25:
        return "Svært lettlest (for barn)"
    elif 25 <= lix < 35:
        return "Lettlest (enkel litteratur, aviser)"
    elif 35 <= lix < 45:
        return "Middels vanskelig (standard aviser, generell sakprosa)"
    elif 45 <= lix < 55:
        return "Vanskelig (akademiske tekster, offisielle dokumenter)"
    else:
        return "Svært vanskelig (vitenskapelig litteratur)"



def calculate_readability_index(query: str) -> str:
    print(f'\n\033[31mTool 4: calculate_readability_index with query:\033[34m"{query}"\033[0m')


    # Accumulate text from all documents
    text = ""
    match = False
    for doc in all_helsenorge_documents_list:
        doc_url = doc.get('url')

        if doc_url == query:
            # url is found in the document list
            doc_id = doc.get('id')
            document = index.docstore.get_document(doc_id)
            text = document.text
            
            # Count words
            words = text.split()
            num_words = len(words)

            # Count sentences (assuming sentences end with '.', '!', or '?')
            num_sentences = len(re.split(r'[.!?]', text)) - 1  # Remove trailing empty splits

            # Count long words (more than 6 letters)
            num_long_words = sum(1 for word in words if len(re.sub(r'[^a-zA-Z]', '', word)) > 6)

            # Calculate LIX
            lix_score = (num_words / num_sentences) + (num_long_words / num_words) * 100
            
            # Get LIX category
            lix_category = categorize_lix(lix_score)

            return (f"Readability_index for {query} is {lix_score} witch correspond to {lix_category}")
    

    return "calculate_readability_index failed"



calculate_readability_index_tool = FunctionTool.from_defaults(
    name="calculate_readability_index",
    fn=calculate_readability_index,
    description="Calculate the readability index from a spesific url"
)

# END TOOL DEFINITION ================================

# Eksempelbruk
if __name__ == "__main__":
    
    build_context()
    
    text_qa_template =  ChatPromptTemplate(chat_text_qa_msgs)

    response_synthesizer = get_response_synthesizer(
        response_mode= "tree_summarize",
        text_qa_template = text_qa_template,
        summary_template= text_qa_template, #definitly in use for response_mode = tree_summarize
        structured_answer_filtering=True, 
        verbose=True,
    )
    
    # Tool 5 (must be build after build_context)
    query_engine = index.as_query_engine(
        similarity_cutoff= 0.7, 
        similarity_top_k=int(10),
        response_synthesizer=response_synthesizer)
    
        
    def article_query(query: str)->str:
        
        try: 
        
            print(f'\033[31mTool 5: article_query \033[34m"{query}"\033[0m')
            response = query_engine.query(query)
            
            # Filter out nodes with score=None
            nodes_with_scores = [node for node in response.source_nodes if node.score is not None]
            sorted_nodes = sorted(nodes_with_scores, key=lambda x: x.score, reverse=True)
            
            # you can map the nodes back to the documents
            top_4_nodes = sorted_nodes[:4]
            
            # Print all document IDs in the docstore
            #print("Available document IDs:", list(index.docstore.docs.keys()))
            
            respons_str =""
            for idx, node in enumerate(top_4_nodes):
                #print(f"\nnode.id_{idx}: {node.node_id}  {node}")
                doc_id = node.node_id
                doc = index.docstore.get_document(doc_id)
            
                respons_str = (f'\n\033[33mThe most relevant article for your question is: \033[34m{doc.metadata.get("url")}\033[0m'
                    f'\n\033[33mTitle: \033[34m{doc.metadata.get("title")}\033[0m'
                    f'\n\033[33mRelevans factor is: \033[34m{node.score:.2f}\033[0m'
                    f'\n\033[33mKeywords are: \033[34m{doc.metadata.get("keywords")}\033[0m'
                    '\n\033[33mThe article is answering following questions:'
                    f'\n\033[34m{doc.metadata.get("questions_this_excerpt_can_answer")}\033[0m'
                    )
                break
        
            print(str(response) + respons_str)
            
            return str(response) + respons_str
        except Exception as e:
            print("Error article_query:", e)
            return f"Error article_query: {str(e)}"
    
    article_query_tool = FunctionTool.from_defaults(
        name="article_query",
        fn=article_query,
        description="Answer a user query about the articles from HelseNorge.no"
    )



    from llama_index.llms.azure_openai import AzureOpenAI   

    # Bygg agenten med verktøyene
    agent = OpenAIAgent.from_tools(
        tools =  
            [            article_query_tool,
            filter_relevant_articles_tool,
            create_excel_tool,
            translate_to_nynorsk_tool,
            calculate_readability_index_tool],
        llm=llm,
        verbose=True,
        system_prompt=  
            "You are a helpful assistant, and you will be given a user request."
            "Some rules to follow:"
            "- Always answer the request using the given context information and not prior knowledge"
            "- If you cannot find the answer in the given context information,"
            " reply: 'Beklager, jeg kunne ikke finne svaret i den gitte konteksten.'."
    )
    
    # query = "Filter the articles ments from helsenorge.no based on keyword 'snus', and make an Excel-file with informasjon"
    # print(f'\n========================\nAgent called with query: "{query}"')
    # agent_response = agent.query(query)
    # print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
    
    
    query = "Gi meg tips for hvordan slutte med snus"
    print(f'\n========================\nAgent called with query: "{query}"')
    agent_response = agent.query(query)
    print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
    
    query = "Gi meg tips for hvordan velge prevensjonsmiddel, oversett svaret til nynorsk"
    print(f'\n========================\nAgent called with query: "{query}"')
    agent_response = agent.query(query)
    print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
 
    # agent_response = agent.query(
    #     "Når sank titanic?"
    # )
    # print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
    
    query = "Get all the articles from helsenorge.no, and make an Excel-file with information"
    print(f'\n========================\nAgent called with query: "{query}"') 
    agent_response = agent.query(query)
    print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
    
    query = "Calculate the readability index for this url: 'https://www.helsenorge.no/fodsel/fodselens-ulike-faser/'"
    print(f'\n========================\nAgent called with query: "{query}"')
    agent_response = agent.query(query)
    print(f'\n\033[33mAgent_response: \033[34m{agent_response}\033[0m' )
    
    # query = "Filter the articles ments from helsenorge.no based on keyword 'svangerskap', and produce a word cloud from the list"
    # print(f'\n========================\nAgent called with query: "{query}"')
    # agent_response = agent.query(query)
    # print(agent_response)
    