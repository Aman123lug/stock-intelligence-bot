import streamlit as st
import pandas as pd
from dotenv import load_dotenv

import base64
from io import BytesIO
from PIL import Image



load_dotenv()
# # Load 
# df = pd.read_csv("data/screener_data_cleaned.csv")

import streamlit as st
from lida import Manager, llm

st.title("LIDA Chat Logs 📊")

lida = Manager(text_gen=llm("openai"))
summary = lida.summarize("data/screener_data_cleaned.csv")
goals = lida.goals(summary, n=2)
charts = lida.visualize(summary=summary, goal=goals[0])

st.subheader("Goals:")
st.write(goals)

# Extract the raster data
raster_data = charts[0].raster

# Decode the base64 raster data
image_data = base64.b64decode(raster_data)
image = Image.open(BytesIO(image_data))

# Display the image in Streamlit
st.image(image, caption="Generated Chart")
