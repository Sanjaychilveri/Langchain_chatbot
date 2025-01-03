import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

## use timer to see how much it si taking

import time

from dotenv import load_dotenv
load_dotenv()

## Load the groq api key
groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) ## taking first 50 doc
    st.session_state.vectors_db=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
st.title("ChatGroq Demo")
llm=ChatGroq(api_key=groq_api_key,model='mixtral-8x7b-32768')

prompt=ChatPromptTemplate.from_template(
"""
Answer the question based on the context only.
PLease provide the best and accurate response
based on the question
<context>
{context}
<context>
Questions:{input}    
"""
)

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors_db.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)


prompt=st.text_input("Write your prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response Time :",time.process_time()-start)
    st.write(response['answer'])


    # With streamlit Expander

    with st.expander('Document Similarity Search'):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------------------------")