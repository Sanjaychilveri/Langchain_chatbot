# LANGCHAIN_API_KEY="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"
# OPENAI_API_KEY="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"
# LANGCHAIN_PROJECT="Langchain_Ollama"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import os
import streamlit as st
from dotenv import load_dotenv

# ## environment variables call


# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith tracking

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

## creating chatbot

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful assitance please provide response to the useer question"),
        ("user","Question:{question}")
    ]
    
)

# Streamlit framework

st.title("langchain demo with llama2 API")
input_text=st.text_input("Search the topic you want")

# open AI LLM call

llm=Ollama(model="llama2")
output_parser=StrOutputParser()

##Chain

chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))