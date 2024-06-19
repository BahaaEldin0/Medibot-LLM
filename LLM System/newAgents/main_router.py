from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

import os


openai_api_key = os.getenv("OPENAI_API_KEY")

gpt_4o = ChatOpenAI(model="gpt-4o",openai_api_key=openai_api_key, temperature=0, streaming=True)

main_router_prompt = PromptTemplate(
    template="""You are an expert at routing a user question to the correct medical assistance agent.
Use 'lab_report' for questions about lab reports.
Use 'symptom_chat' for questions about symptoms and conditions.
Use 'chest_xray' for questions about chest X-ray images.
Use 'doctor_referral' for questions about doctor referrals.
Use 'general_chat' for general questions.
Give a binary choice 'lab_report', 'symptom_chat', 'chest_xray', 'doctor_referral' or 'general_chat' based on the question. Return a JSON with a single key 'datasource' and no preamble or explanation. Question to route: {messages}""",
    input_variables=["messages"],
)

main_router = main_router_prompt | gpt_4o | JsonOutputParser()
