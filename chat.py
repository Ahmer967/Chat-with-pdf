from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core import Settings
from llama_index.core import StorageContext
from db import vector_store

llm = OpenAI(model = 'gpt-4o', api_key = 'OPENAI_API_KEY')


embed_model = OpenAIEmbedding(
model="text-embedding-3-large",
api_key = 'OPENAI_API_KEY',
)


Settings.llm = llm
Settings.embed_model = embed_model

storage_context = StorageContext.from_defaults(vector_store=vector_store)

sentence_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

reorder = LongContextReorder()
from openai import OpenAI
client = OpenAI(api_key='OPENAI_API_KEY')

from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def chat(chat_id, query):
    retriever = VectorIndexRetriever(index=sentence_index, 
                                    filters=MetadataFilters(
                                    filters=[
                                        ExactMatchFilter(
                                            key="user_id",
                                            value=chat_id,
                                        ),
                                        ],
                                    ),
                                    node_postprocessors=[reorder,
                                        MetadataReplacementPostProcessor(target_metadata_key="window"),
                                    ],
                                    similarity_top_k=7)
    ret = retriever.retrieve(query)
    text = ''
    for i in ret:
        text+= f"**Page: {str(i.metadata['page_label'])}**  Info:  {i.metadata['window']}\n\n"
    
    try:
        response = client.chat.completions.create(
        model='gpt-4o',
        temperature=0,
        messages=[
            {"role": "user", "content": f"""
            Based on the Information answer the User Question it should be properly formatted with details. \n\n
            Information: {text} \n\n

            User Question: {query}

            """}

        ]
    )
        generated_text = response.choices[0].message.content
        return generated_text, ret

    except Exception as e:
        print(f"Error in Chat: {e}")
