import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from graph import finalAgent
from langchain_core.messages import HumanMessage
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel


load_dotenv()

PORT = os.getenv("PORT") or 5001 
app = FastAPI()

# Define routes and logic for user interaction with agents here

# Define request body model
class ChatRequest(BaseModel):
    message: str
    user_name: str
    user_id: str
    conversation_id: str
    history: list


@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message
    user_name = request.user_name
    user_id = request.user_id
    conversation_id = request.conversation_id
    
    response_state, response_message = "success", "Hello, how can I help you today?"
    history = request.history


    ques = {"messages": [HumanMessage(content=message)],"history":history}
    
    start = time.time()
    output = finalAgent.invoke(ques)
    end = time.time()
    print("Time taken to process the request: ", end-start)
    response_message = output["messages"][-1].content
    
    response_data = {
        "user_name": user_name,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "question": message,
        "answer": response_message,
        "state": response_state
    }
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
