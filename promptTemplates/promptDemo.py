from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate

# create model
llm = ChatOllama(model = "mistral:latest")

# create prompt template
prompt_template = PromptTemplate(
    input_variables=["country", "num_of_paras", "language"],
    template= """You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cusine of {country}
    Answer in {num_of_paras} short paras in {language}"""
)

# put a title on web page
st.title("Cuisine Info")

# get the input in prompt_template from user
country = st.text_input("Enter the country:")
num_of_paras = st.number_input("Enter the num_of_paras:", min_value = 1, max_value = 5)
language = st.text_input("Enter the language:")

if country and num_of_paras and language:
    # run model with respect to prompt template
    response = llm.invoke(prompt_template.format(country = country,
                                                 num_of_paras = num_of_paras,
                                                 language = language))
    st.write(response.content)
