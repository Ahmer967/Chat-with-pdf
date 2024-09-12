from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
import chromadb
from llama_index.core.node_parser import SentenceWindowNodeParser
import uuid
from llama_index.core.ingestion import IngestionPipeline
import requests
import os

reorder = LongContextReorder()

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=10,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

text_splitter = SentenceSplitter(chunk_size=4000, chunk_overlap=0)


llm = OpenAI(model = 'gpt-4o', api_key = 'OPENAI_API_KEY')

embed_model = OpenAIEmbedding(
model="text-embedding-3-large",
api_key = 'OPENAI_API_KEY',
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

sentence_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)
def upload_vdb(chat_id, urls):
    files = []
    try:
        count = 0
        for url in urls:
            filename = str(uuid.uuid4().hex)[::-1]+".pdf"
            response = requests.get(url)
            response.raise_for_status()  
            with open(f'{filename}', 'wb') as f:
                f.write(response.content)
                files.append(f'{filename}')
                count+=1
            print(f"Downloaded {filename}")
    except Exception as e:
        print(f'Downloading File Failed: {count}')

    documents = SimpleDirectoryReader(input_files=files).load_data()
    cleaned_docs = []
    for d in documents:
        cleaned_docs.append(d)
    metadata_additions = {"user_id": chat_id}
    [cd.metadata.update(metadata_additions) for cd in cleaned_docs]

    pipeline = IngestionPipeline(
        name=chat_id,
        project_name=chat_id,
         transformations=[
            node_parser,       
             embed_model,
            ],
        vector_store=vector_store
         )

    run = pipeline.run(documents=cleaned_docs, inplace=True)
    pipeline.disable_cache = True
    
    try:
        os.remove(f'{filename}')
    except Exception as e:
        print('Remove')
