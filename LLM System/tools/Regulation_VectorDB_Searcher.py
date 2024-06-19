import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
import os


openai_api_key = os.getenv("OPENAI_API_KEY")

gpt_4o = ChatOpenAI(model="gpt-4o",openai_api_key=openai_api_key, temperature=0, streaming=True)
ada_embeddings = OpenAIEmbeddings(model='text-embedding-ada-002',api_key=openai_api_key,base_url="https://api.openai.com/v1")

client = MilvusClient(
    uri="http://logisti.ai:19530",
)
@tool("Regulation_VectorDB_Searcher")
def Regulation_VectorDB_Searcher(question: str) -> str:
  """This tool searches in a vectorDB collection and returns results."""
  query_vectors = ada_embeddings.embed_query(question)
  res = client.search(
      "bold_head",
      data=[query_vectors],
      limit=3,
      output_fields=['metadata','id']
  )
  cunx = ""
  for i in res[0]:
    content = i['entity']['metadata'][0]['prechunk'] + "\n\n" + \
              i['entity']['metadata'][0]['content'] + "\n\n" + \
              i['entity']['metadata'][0]['postchunk']

    cunx += f"""
    Title:{i['entity']['metadata'][0]['title']}
    ID:{i['id']}
    Content:{content}
    references:{i['entity']['metadata'][0]['references']}

    """
  return cunx

