from langchain_ollama import ChatOllama
import streamlit as st

from langchain.globals import set_debug

set_debug(True)

# create model
llm = ChatOllama(model = "gemma:2b")

# put a title to your web page
st.title("Ask Anything!")

# get question from user
prompt = st.text_input("Enter the question")

if prompt:
    # get the answer from llm
    response = llm.invoke(prompt)

    st.write(response.content)