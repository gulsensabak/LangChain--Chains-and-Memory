from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 
import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# create the model
llm = ChatOllama(model = "gemma:2b")

# create the prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an Agile Coach." 
         "Answer any questions related to the agile process"),
        # to keep the chat history
        # this message placeholder acts as a placeholder where the chat history can be injected later on
        # this is where the entire chat history that is our previous request and the responses from the AI model will be injected
        # runnable with message history will inject them. We call it in below
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}")
    ]
)

# create the chain
chain = prompt_template | llm

# create a list to add request and responses by RunnableWithMessageHistory
history_for_chain = StreamlitChatMessageHistory()

# create chain with history to inject the request and responses
# RunnableWithMessageHistory inject the previous response and request to the ChatPromptTemplate
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    # should have the same name with the variable name corresponding to human
    input_messages_key= "input",
    # should have match whatever we have in messagePlaceholder
    history_messages_key= "chat_history"
)

# put a title
st.title("Agile Guide")

# create input variables
input = st.text_input("Enter the question: ")

if input:
    # run the chain
    response = chain_with_history.invoke({"input": input},
                                            {"configurable": {
                                            "session_id": "abc123"
                                        }})
    st.write(response.content)