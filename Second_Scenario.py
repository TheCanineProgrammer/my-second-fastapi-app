from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from datasets import load_dataset
import os
import openai
import pandas as pd
from rapidfuzz import process, fuzz
import logging

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ------------------------------
# OpenAI API setup
# ------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI()

# ------------------------------
# Pydantic models
# ------------------------------
class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

# ------------------------------
# Load dataset
# ------------------------------

dataset = load_dataset("The-CaPr-2025/base_products")  # pulls from HF Hub
base_df = dataset["train"].to_pandas()

#random_key_to_names = base_df.set_index("random_key")[["english_name", "persian_name"]].to_dict(orient="index")
all_names = list(base_df.persian_name) + list(base_df.english_name)

# ------------------------------
# Endpoint
# ------------------------------
@app.post("/chat")
async def assistant(request: ChatRequest):
    last_message = request.messages[-1].content.strip()
    
    # Log incoming request
    logging.info(f"Received chat_id={request.chat_id}, message={last_message}")

    # --------
    # Scenario 1: Sanity checks
    # --------
    if last_message.lower() == "ping":
        response = {"message": "pong", "base_random_keys": None, "member_random_keys": None}
        logging.info(f"Response: {response}")
        return response

    if "return base random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        response = {"message": None, "base_random_keys": [key], "member_random_keys": None}
        logging.info(f"Response: {response}")
        return response

    if "return member random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        response = {"message": None, "base_random_keys": None, "member_random_keys": [key]}
        logging.info(f"Response: {response}")
        return response

    # --------
    # Scenario 2: Map query to a base product key using OpenAI
    # --------
    prompt = f'''
    کاربر یک محصول را توصیف کرده است.
    لطفاً فقط نام یا توضیح استاندارد همان محصول را بازگردان (مثلاً «فرشینه مخمل ترمزگیر عرض 1 متر طرح آشپزخانه کد 04»).
    
    کوئری کاربر: "{last_message}"'''

    key = None
    try:
        client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You normalize user product queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        predicted_name = response.choices[0].message.content.strip()
        logging.info(f"{predicted_name}")
    except Exception:
        predicted_name = ""
    
    key = None
    if predicted_name:
        match, score, idx = process.extractOne(
            predicted_name,
            all_names,
            scorer=fuzz.token_sort_ratio
        )

        if score > 70:  # accept only strong matches
            key = base_df.iloc[idx]["random_key"]

    response = {
        "message": None,
        "base_random_keys": [key] if key else None,
        "member_random_keys": None
    }

    # Log outgoing response
    logging.info(f"Response: {response}")
    return response
