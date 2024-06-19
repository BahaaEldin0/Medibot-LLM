import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

import os


openai_api_key = os.getenv("OPENAI_API_KEY")

ada_embeddings = OpenAIEmbeddings(model='text-embedding-ada-002',api_key=openai_api_key,base_url="https://api.openai.com/v1")

client = MilvusClient(
    uri="http://logisti.ai:19530",
)

@tool("Import_VectorDB_Searcher")
def Import_VectorDB_Searcher(query: str) -> str:
    """This tool searches in a vectorDB collection. It returns results."""

    query_vectors = ada_embeddings.embed_query(query)
    res = client.search(
        collection_name='JSON_RAG_FINAL',
        data=[query_vectors],
        limit=3,
        output_fields=['content']
    )

    d1 = res[0][0]["entity"]["content"]
    d2 = res[0][1]["entity"]["content"]
    d3 = res[0][2]["entity"]["content"]
    out = d1 +"\n"+ d2 +"\n"+ d3

    if query.lower() in d1.lower():
        return d1
    elif query.lower() in d2.lower():
        return d2
    elif query.lower() in d3.lower():
        return d3
    else:

        return out.lower()