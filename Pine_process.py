import os
import pandas as pd
import numpy as np
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "stock" 
index = pc.Index(index_name)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load CSV file
df = pd.read_csv("screener_data.csv")

def get_embedding(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None 
    
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")  # ✅ 1536 Model
    embedding = response.data[0].embedding  # Extract embedding
    if any(np.isnan(embedding)) or any(np.isinf(embedding)):
        return None

    return embedding

def format_metadata(row):
    return {
        "company": row["Company"],
        "market_cap": row["Market Cap"],
        "stock_pe": row["Stock P/E"],
        "roe": row["ROE"],
        "revenue_breakup": row["Revenue Breakup"],
        "current_price": row["Current Price"],
        "high_low": row["High/Low"],
        "book_value": row["Book Value"],
        "dividend": row["Dividend Yield"],
        "roce": row["ROCE"],
        "face_value": row["Face Value"],
        "description": row["Description"]
    }


vectors = []
for _, row in df.iterrows():
    unique_id = str(uuid.uuid4())
    text_data = row["Description"]

    embedding = get_embedding(text_data)

    if embedding: 
        metadata = format_metadata(row)

        # Append to batch upsert
        vectors.append((unique_id, embedding, metadata))

if vectors:
    index.upsert(vectors)
    print("✅ Data successfully pushed to Pinecone with OpenAI embeddings!")
    
