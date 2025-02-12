from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load Hugging Face chatbot model
chatbot_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")

app = FastAPI()

class UserMessage(BaseModel):
    message: str

@app.post("/chat/")
def chat(user_input: UserMessage):
    response = chatbot_pipeline(user_input.message)
    return {"reply": response[0]["generated_text"]}

