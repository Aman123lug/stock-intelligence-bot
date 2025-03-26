import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import query_pinecone, get_llm_response
from lida import Manager, llm
from dotenv import load_dotenv
from io import BytesIO
import time
import base64
from PIL import Image
import seaborn as sns
import re

sns.set_theme(style="whitegrid")

load_dotenv()
def fix_lida_specification(spec):
    """Fix deprecated seaborn parameters in LIDA-generated code"""
    # Fix palette without hue warning
    spec = re.sub(
        r"sns\.barplot\(([^)]*)palette=['\"][^'\"]+['\"]([^)]*)\)",
        r"sns.barplot(\1hue=\2, palette='viridis', legend=False\2)",
        spec
    )

    spec = spec.replace("ci=None", "errorbar=None")

    if "return plt;" in spec and "def plot(" not in spec:
        spec = f"def plot(data):\n    {spec}\n    return plt"
    
    return spec

# Data cleaning function
def clean_numeric_data(df):
    """Clean numeric columns by removing currency symbols and commas"""
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        if col == 'company':
            continue
            
        cleaned_df[col] = cleaned_df[col].astype(str)
        
        cleaned_df[col] = cleaned_df[col].str.replace(r'[^\d.]', '', regex=True)
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        cleaned_df[col] = cleaned_df[col].fillna(0)
        
    return cleaned_df

# Visualization generation function
def generate_visualizations(data, lida, n_charts=4, selected_goal=None):
    """Generate and display visualizations from cleaned data"""
    try:
        # Generate summary
        summary = lida.summarize(data)
        
        # Generate goals
        if selected_goal:
            goals = lida.goals(summary, n=min(n_charts, 3), persona=selected_goal.question)
        else:
            goals = lida.goals(summary, n=n_charts)
        
        charts = []
        for goal in goals:
            try:
                if hasattr(goal, 'specification'):
                    goal.specification = goal.specification.replace(
                        "ci=None", "errorbar=None"
                    ).replace(
                        "palette='viridis'", 
                        "palette='viridis', hue='company', legend=False, errorbar=None"
                    )
                
                chart = lida.visualize(summary=summary, goal=goal)
                if chart:
                    charts.append(chart)
            except Exception as e:
                st.warning(f"Couldn't generate chart: {str(e)}")
                continue
        
        return charts
        
    except Exception as e:
        st.error(f"Visualization generation failed: {str(e)}")
        return []

# Load data
df = pd.read_csv("data/screener_data_cleaned.csv")  # Replace with your file path

# Select columns including company name
numerical_cols = ["company", "market_cap", "stock_pe", "roe", "current_price", 
                 "high_low", "book_value", "dividend", "roce", "face_value"]

st.title("Stock Intelligence RAG App 🌊")
st.divider()

lida = Manager(text_gen=llm("openai"))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False
if "selected_goal" not in st.session_state:
    st.session_state.selected_goal = None

# Example prompts
example_prompts = [
    "What was the net profit revenue breakup of TCS in the last quarter compare to others?",
    "Compare the ROE of the companies, including BAJAJFINSV, and HDFC.",
    "Compare face value and dividend for BAJAJ-AUTO and HDFC",
    "Tell me the market capital of HINDUNILVR, ITC compare to others?",
    "Generate a chart comparing market capital across KOTAKBANK and HDFC."
]

# Example prompt buttons
button_cols = st.columns(3)
button_pressed = ""
if button_cols[0].button(example_prompts[0]):
    button_pressed = example_prompts[0]
elif button_cols[1].button(example_prompts[1]):
    button_pressed = example_prompts[1]
elif button_cols[2].button(example_prompts[2]):
    button_pressed = example_prompts[2]

button_cols_2 = st.columns(3)
if button_cols_2[0].button(example_prompts[3]):
    button_pressed = example_prompts[3]
elif button_cols_2[1].button(example_prompts[4]):
    button_pressed = example_prompts[4]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.selected_goal:
    prompt = st.session_state.selected_goal.question
else:
    prompt = st.chat_input("Ask about stock insights") or button_pressed

if prompt:

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response
    context = query_pinecone(prompt)
    response = get_llm_response(prompt, context)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    
    with st.spinner("Generating Visualizations..."):
        try:
            # Get and clean data
            raw_data = pd.DataFrame(context)[numerical_cols]
            data = clean_numeric_data(raw_data)
            
            
            summary = lida.summarize(data)
            goals = lida.goals(summary, n=5) if not st.session_state.selected_goal else lida.goals(summary, n=3, persona=st.session_state.selected_goal.question)
    
            charts = []
            for goal in goals:
                try:
                    if hasattr(goal, 'specification'):
                        goal.specification = fix_lida_specification(goal.specification)
                    
                    chart = lida.visualize(summary=summary, goal=goal)
                    if chart:
                        charts.append(chart)
                except Exception as e:
                    st.warning(f"Couldn't generate chart: {goal.question}\nError: {str(e)}")
                    continue
            
            
            if not charts:
                st.warning("No charts could be generated. Please check your data.")
            else:
                cols = st.columns(2)
                for i, chart in enumerate(charts[:4]):
                    with cols[i % 2]:
                        try:
                            img_data = base64.b64decode(chart[0].raster)
                            st.image(Image.open(BytesIO(img_data)), 
                                    caption=f"Chart {i+1}", 
                                    use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying chart {i+1}: {str(e)}")
        
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")