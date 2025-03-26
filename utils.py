import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import os
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "stock"  # Ensure this index exists in Pinecone
index = pc.Index(index_name)

# Initialize OpenAI Client for embeddings
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_pinecone(query_text, top_k=3):
    # Generate embedding for the query text
    query_embedding = client.embeddings.create(input=[query_text], model="text-embedding-3-small").data[0].embedding

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    results_dict = []
    for match in results.matches:
        results_dict.append({
            "similarity_score": match.score,
            "company": match.metadata.get("company", "N/A"),
            "market_cap": match.metadata.get("market_cap", "N/A"),
            "description": match.metadata.get("description", "N/A"),
            "revenue_breakup": match.metadata.get("revenue_breakup", "N/A"),
            "current_price": match.metadata.get("current_price", "N/A"),
            "stock_pe": match.metadata.get("stock_pe", "N/A"),
            "roe": match.metadata.get("roe", "N/A"),
            "high_low": match.metadata.get("high_low", "N/A"),
            "book_value": match.metadata.get("book_value", "N/A"),
            "dividend": match.metadata.get("dividend", "N/A"),
            "roce": match.metadata.get("roce", "N/A"),
            "face_value": match.metadata.get("face_value", "N/A")
        })

    return results_dict


def get_llm_response(query_text, context, model="gpt-4o-mini"):
    """
    Sends user query and retrieved context to OpenAI LLM for a response.
    
    :param query_text: The user's input query.
    :param context: Retrieved relevant information from vector DB.
    :param model: OpenAI model to use (default: gpt-4-turbo).
    :return: LLM-generated response.
    """

    prompt = f"""
    You are a highly knowledgeable Stock Intelligencne assistant. Answer the user's query based on the provided context. answer only provided quext_text. Avoid saying from provided context i found this/that.
    
    **Context:** 
    {context}
    
    **User Query:** 
    {query_text}

    Provide a clear and concise response.
    """

    # Send request to OpenAI API
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content




llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)
filename = "data/screener_data.csv"

def summerize_csv(filename):

    df = pd.read_csv(filename, low_memory=False)

    pandas_agent = create_pandas_dataframe_agent(
        llm=llm_gpt,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": "True"},
    )
    
    data_summary = {}

    data_summary["initial_data_sample"] = df.head()

    
    data_summary["missing_values"] = pandas_agent.run(
        "Is there any missing values in dataset and how many? Your response should be like 'There are X number of missing values in this dataset' replace missing values to 'X'"
    )

    data_summary["dupplicate_values"] = pandas_agent.run(
        "Is there any duplicate values in dataset and how many? Your response should be like 'There are X number of dupplicate values in this dataset' replace missing values to 'X'"
    )

    data_summary["essential_metrics"] = df.describe()

    return data_summary


# get dataframe


def get_dataframe(filename):

    df = pd.read_csv(filename, low_memory=False)
    return df



def analyze_trend(filename, variable):

    df = pd.read_csv(filename, low_memory=False)

    pandas_agent = create_pandas_dataframe_agent(
        llm=llm_gpt,
        df=df,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": "True"},
        allow_dangerous_code=True,
        
    )

    trend_response = pandas_agent.run(
        f"Interpret the trend of this shortly: {variable}. Do not reject the interpretation!. The rows of the dataset is historical. So you can do interpreting with looking the rows of dataset"
    )

    return trend_response



def ask_question(filename, question):

    df = pd.read_csv(filename, low_memory=False)

    pandas_agent = create_pandas_dataframe_agent(
        llm=llm_gpt,
        df=df,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": "True"},
        allow_dangerous_code=True,
    )

    AI_response = pandas_agent.run(question)

    return AI_response