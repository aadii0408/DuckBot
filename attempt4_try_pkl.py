import pickle
import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader

# import google.generativeai as genai

# from streamlit import cli as stcli
import streamlit as st

from attempt4 import fetch_and_store_data

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)

# if not GEMINI_API_KEY:
#     raise ValueError("Missing Gemini API Key. Please set it in the .env file.")


# Load the stored data from the pickle file
def load_stored_data(pickle_filename):
    with open(pickle_filename, "rb") as f:
        stored_data = pickle.load(f)
    return stored_data


# Function to get response based on relevant chunks
def get_response(question, stored_data):
    # Search through the stored data to find relevant chunks
    relevant_chunks = []

    for url, chunks in stored_data.items():
        for chunk in chunks:
            if any(
                keyword.lower() in chunk.page_content.lower()
                for keyword in question.split()
            ):
                relevant_chunks.append(chunk)

    if not relevant_chunks:
        return "No relevant information found."

    # Combine relevant chunks into a single context
    knowledge = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    # Initialize OpenAI LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

    # Construct prompt
    prompt = f"""
    You are an assistant that answers questions based only on the provided knowledge.
    Do not use any external information.

    Knowledge: {knowledge}

    Question: {question}
    """

    # Get the response from the LLM
    response = llm.predict(prompt)
    return response


# Load the stored data from the pickle file
stored_data = load_stored_data("data_stored.pkl")

# Ask a question
question = "Who is Sandeep bhatt?"
response = get_response(question, stored_data)

print("Final Answer:", response)


# Streamlit User Interface (UI)
def run_streamlit_ui():
    st.set_page_config(page_title="Knowledge Assistant", page_icon="💡")

    # Set the custom theme colors
    st.markdown(
        """
        <style>
            body {
                background-color: #F4F4F4;
                color: #333;
            }
            .stButton>button {
                background-color: #9B1B30;
                color: white;
                border-radius: 8px;
                font-size: 16px;
            }
            .stTextInput>div>div>input {
                border-radius: 10px;
                background-color: #f2f2f2;
                border: 2px solid #d3d3d3;
            }
            .stTextInput>label {
                font-size: 18px;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header and Intro text
    st.title("DuckOps 🤖")
    st.markdown(
        "Ask me anything related to Stevens Institute of Technology or any provided topic!"
    )

    # Input for user question
    question = st.text_input("Ask your question:", "")

    if question:
        # Load URLs from the pickle file
        with open("data_stored.pkl", "rb") as file:
            urls = pickle.load(file)

        # Get the relevant response from fetched data
        response = fetch_and_store_data(question, urls)

        # Display the answer
        st.subheader("Answer:")
        st.write(response)


# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_ui()
