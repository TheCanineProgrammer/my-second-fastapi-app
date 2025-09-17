from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from huggingface_hub import hf_hub_download
import os
import openai
import pandas as pd
from rapidfuzz import process
import logging

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

app = FastAPI()

class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

local_path = hf_hub_download(
    repo_id="The-CaPr-2025/base_products",
    filename="base_products.parquet",
    repo_type="dataset"
)

base_df = pd.read_parquet(local_path)
random_key_to_names = base_df.set_index("random_key")[["english_name", "persian_name"]].to_dict(orient="index")

# Prepare list of all names for fuzzy matching
all_names = list(base_df.persian_name) + list(base_df.english_name)

class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

@app.post("/chat")
async def assistant(request: ChatRequest):
    last_message = request.messages[-1].content.strip()

    # --------
    # Scenario 1: Sanity checks
    # --------
    if last_message.lower() == "ping":
        return {"message": "pong", "base_random_keys": None, "member_random_keys": None}

    if "return base random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        return {"message": None, "base_random_keys": [key], "member_random_keys": None}

    if "return member random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        return {"message": None, "base_random_keys": None, "member_random_keys": [key]}

    # --------
    # Scenario 2: Map query to a base product key using OpenAI
    # --------
    prompt = f"""
                You are an assistant that maps a user query to exactly one base product key.

                Available base products:
                {random_key_to_names}

                User query: "{last_message}"

                Instructions:
                - Return exactly one base key that matches the query.
                - If no match is found, return null.
                - Output ONLY the key or null, nothing else.
                """

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=10,
            temperature=0.3
        )
        key = response.choices[0].text.strip()
        if key.lower() == "null" or key not in random_key_to_names:
            key = None
    except Exception as e:
        key = None  # fallback if OpenAI fails

    # --------
    # Fallback: rapidfuzz search if OpenAI fails
    # --------
    if not key:
        best_match, score, idx = process.extractOne(last_message, all_names)
        if score > 70:  # only accept strong matches
            key = base_df.iloc[idx]["random_key"]

    return {
        "message": None,
        "base_random_keys": [key] if key else None,
        "member_random_keys": None
    }