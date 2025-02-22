from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st

# create model
llm = ChatOllama(model = "gemma:2b")

# create prompts
product_prompt = PromptTemplate(
    input_variables= ["product_name", "features"],
    template="""
        You are an experienced marketing specialist.
        Create a catchy subject line for a marketing
        email promoting the following product: {product_name}.
        Highlight these features: {features}.
        Respond with only the subject line
    """
)

target_prompt = PromptTemplate(
    input_variables= ["product_name", "subject_line", "audience"],
    template="""
        Write a marketing email of 300 words for the
        product: {product_name}. Use the subject line:
        {subject_line}. Tailor the message for the
        following target audience: {audience}.
        Format the output as a JSON object with three
        keys: 'subject', 'audience', 'email' and fill
        them with respective values
    """
)


# create chains
first_chain = product_prompt | llm | StrOutputParser()
second_chain = target_prompt | llm | JsonOutputParser()

# merge chains
finalized_chain = first_chain | (lambda subject_line: {"subject_line": subject_line, "product_name": product_name, "audience": audience}) | second_chain

# put a title
st.title("Marketing Email Generator")

# create variables
product_name = st.text_input("Input Product Name")
features = st.text_input("Input Product Features (comma-seperated)")
audience = st.text_input("Input Target Audience")

if product_name and features and audience:
    response = finalized_chain.invoke({"product_name": product_name, 
                                       "features": features})
    st.write(response.content)