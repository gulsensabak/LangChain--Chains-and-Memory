from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st

# create a model
llm = ChatOllama(model = "gemma:2b")

# create prompt
interview_tips = PromptTemplate(
    input_variables= ["company", "position", "strengths", "weaknesses"],
    template="""You are a career coach. Provide tailored interview tips for the
    position of {position} at {company}.
    Highlight your strengths in {strengths} and prepare for questions
    about your weaknesses such as {weaknesses}.""")

# put a title 
st.title("Interview Tips Generator")

# create variables
company = st.text_input("Company Name: ")
position = st.text_input("Position Title: ")
strengths = st.text_input("Your Strengths: ", height = 100)
weaknesses = st.text_input("Your Weaknesses: ", height = 100)

if company and position and strengths and weaknesses:
    response = llm.invoke(interview_tips.format(
        company = company,
        position = position,
        strengths = strengths,
        weaknesses = weaknesses
    ))
    st.write(response.content)