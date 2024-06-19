from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# from tools.ChestXrayClassifier import ChestXrayClassifier
import os


openai_api_key = os.getenv("OPENAI_API_KEY")
gpt_4o = ChatOpenAI(model="gpt-4o",openai_api_key=openai_api_key, temperature=0, streaming=True)

chest_xray_prompt = PromptTemplate(
    template="""You are a medical assistant that classifies chest X-ray images to identify potential diseases. Use the ChestXrayClassifier tool to classify the image and provide the diagnosis.

Chest X-ray Image: {question}

Always Use Chat History to get more context
Chat History: {history}

Diagnosis:
- Disease Identified:
- Confidence Level:
- Recommended Next Steps:""",
    input_variables=["question","history"],
)

chest_xray_agent = gpt_4o
