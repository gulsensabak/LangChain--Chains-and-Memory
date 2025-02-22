from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

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
    input_variables= ["title", "emotion"],
    template= """
        You need to write a powerful {emotion} speech of 350 words
        for the following title: {title}
         **Output must be a valid JSON object** with two keys:
        - `"title"`: The speech title
        - `"speech"`: The full speech text
        - `"emotion"`: The emotion in text

        
        Make sure the output is valid JSON.
        """
)

# create chains
first_chain = title_prompt | llm | StrOutputParser()
second_chain = speech_prompt | llm | JsonOutputParser()

# connect chains to create simple sequential chain

# since we have an input that comes from external user we use sub phase between chains
final_chain = first_chain | (lambda title:{"title": title, "emotion": emotion}) | second_chain

st.title("Speech Generator")

# create input variables
topic = st.text_input("Enter the topic: ")
emotion = st.text_input("Enter the emotion: ")

if topic and emotion:
    response = final_chain.invoke({"topic": topic})
    st.write(response)