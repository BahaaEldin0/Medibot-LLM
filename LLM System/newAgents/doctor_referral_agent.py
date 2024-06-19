from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


openai_api_key = os.getenv("OPENAI_API_KEY")
gpt_4o = ChatOpenAI(model="gpt-4o",openai_api_key=openai_api_key, temperature=0, streaming=True)

doctor_referral_prompt = PromptTemplate(
    template="""You are a medical assistant that refers users to the right doctor based on their medical conditions.

User Condition: {question}

Always Use Chat History to get more context
Chat History: {history}

Recommended Doctor:
- Specialty:
- Doctor's Name (if available):
- Contact Information (if available):
- Next Steps for Appointment:""",
    input_variables=["question","history"],
)

doctor_referral_agent = gpt_4o
