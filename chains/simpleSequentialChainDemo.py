from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

# create the model
llm = ChatOllama(model = "gemma:2b")

# create title prompt
title_prompt = PromptTemplate(
    input_variables= ["topic"],
    template= """You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title
    """
)

# create speech_prompt
speech_prompt = PromptTemplate(
    input_variables= ["title"],
    template= """
        You need to write a powerful speech of 350 words
        for the following title: {title}
        """
)

# create chains
first_chain = title_prompt | llm | StrOutputParser()
second_chain = speech_prompt | llm

# connect chains to create simple sequential chain
final_chain = first_chain | second_chain

st.title("Speech Generator")

# create input variables
topic = st.text_input("Enter the topic: ")

if topic:
    response = final_chain.invoke({"topic": topic})
    st.write(response.content)