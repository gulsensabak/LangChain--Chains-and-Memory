from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import streamlit as st


# create the model
llm = ChatOllama(model = "gemma:2b")

# create the prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an Agile Coach."
         "Answer any questions related to the agile process"),
         ("human", "{input}")
    ]
)

# put a title
st.title("Agile Guide")

# create input variables
input = st.text_input("Enter the question: ")

# create the chain
chain = prompt_template | llm

if input:
    # run the chain
    response = chain.invoke({"input": input})
    st.write(response.content)