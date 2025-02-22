from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate

# create the model 
llm = ChatOllama(model = "mistral:latest")

# create the prompt_template
prompt_template = PromptTemplate(
    input_variables= ["city", "month", "language", "budget"],
    template= """
    Welcome to the {city} travel guide!
    If you are visiting in {month}, here is what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for travelling on a {budget} budget.
    Enjoy your trip!
    """
)

# put a title
st.title("Travel Guide for you")

# create input variables
city = st.text_input("Enter the city: ")
month = st.text_input("Enter the month of travel: ")
language = st.text_input("Enter the language: ")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

if city and month and language and budget:
    response = llm.invoke(prompt_template.format(
        city = city,
        month = month,
        language = language,
        budget = budget
    ))
    st.write(response.content)