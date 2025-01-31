import pickle
import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


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
stored_data = load_stored_data("stored_data.pkl")

# Ask a question
question = "Is shuttle service is free for Stevens students?"
response = get_response(question, stored_data)

print("Final Answer:", response)
