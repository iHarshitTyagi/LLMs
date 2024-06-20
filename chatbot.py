# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

'''
ChatPromptTemplate: This is a tool from langchain-core that helps us create structured prompts for our chatbot.
Ollama: This is the module that gives us access to the LLaMA model.
streamlit as st: This imports the Streamlit library and allows us to use it with the shorthand st.
'''

# Set up the Streamlit framework
st.title('Langchain Chatbot With LLAMA2 model')  # Set the title of the Streamlit app
input_text=st.text_input("Ask your question!")  # Create a text input field in the Streamlit app

# Initialize the Ollama model
llm=Ollama(model="llama2")

# Define a prompt template for the chatbot
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the questions"),
        ("user","Question:{question}")
    ]
)

# Create a chain that combines the prompt and the Ollama model
chain=prompt|llm

# Invoke the chain with the input text and display the output
if input_text:
    st.write(chain.invoke({"question":input_text}))
