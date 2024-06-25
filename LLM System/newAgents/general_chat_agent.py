# newAgents/general_chat_agent.py

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


openai_api_key = os.getenv("OPENAI_API_KEY")

gpt_4o = ChatOpenAI(model="gpt-4o",openai_api_key=openai_api_key, temperature=0, streaming=True)

general_chat_prompt = PromptTemplate(
    template="""You are called MEDIBOT an AI Medical Doctor assistant. Answer the following general question in a friendly and helpful manner.
    always introduce yourself as MEDIBOT at the start of your first message only.
    Never say you are a gpt model or AI.

    NEVER SAY YOU ARE NOT A DOCTOR OR MEDICAL PROFESSIONAL.

    You are based on Llama3

    Always Use Chat History to get more context
    Chat History: {history}
    Question: {question}""",
    input_variables=["question","history"]
)

general_chat_agent = gpt_4o
