from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


openai_api_key = os.getenv("OPENAI_API_KEY")

gpt_4o = ChatOpenAI(model="gpt-4o",openai_api_key=openai_api_key, temperature=0, streaming=True)

lab_report_prompt = PromptTemplate(
    template="""You are an AI assistant that helps users understand their lab reports. Extract the relevant medical data from the given lab report and provide an analysis.

Lab Report: {question}

Always Use Chat History to get more context
Chat History: {history}

Analysis:
- Key Findings:
- Medical Interpretation:
- Recommended Next Steps:""",
    input_variables=["question","history"],
)

lab_report_agent = gpt_4o
