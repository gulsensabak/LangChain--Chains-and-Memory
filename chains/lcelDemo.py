from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st


# create the model
llm = ChatOllama(model = "gemma:2b")

# create the prompt
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
st.title("LCEL Demo")

# create input variables
city = st.text_input("Enter the city: ")
month = st.text_input("Enter the month of travel: ")
language = st.text_input("Enter the language: ")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])


# create the chain
chain = prompt_template | llm

if city and month and language and budget:
    # run the chain
    response = chain.invoke({"city": city,
                             "month": month,
                             "language": language,
                             "budget": budget})
    st.write(response.content)