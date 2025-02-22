from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st

# create the models
llm1 = ChatOllama(model = "gemma:2b")
llm2 = ChatOllama(model = "mistral:latest")

# create prompt template:
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
    - Introduction
    - 3 main points with subpoints
    - Conclusion
    """
)

introduction_prompt = PromptTemplate(
    input_variables= ["outline"],
    template="""You are a professional blogger.
    Write an engaging introduction paragraph based on the following
    outline: {outline}
    The introduction should hook the reader and provide a brief
    overview of the topic.
    """
)


# create chains
first_chain = outline_prompt | llm2 | StrOutputParser()
second_chain = introduction_prompt | llm1

# finalize chains
overall_chain = first_chain | second_chain 

st.title("Blog Post Generator")

topic = st.text_input("Input Topic")

if topic:
    response = overall_chain.invoke({"topic": topic})
    st.write(response.content)