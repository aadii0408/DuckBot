from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import openai
from openai import OpenAIError  # Import the base error class
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()


# Function to fetch and process web data
def fetch_web_data(url):
    print(f"Fetching data from: {url}")

    # Load the webpage
    loader = WebBaseLoader(url)
    docs = loader.load()

    if not docs:
        print("No data extracted. Check if the URL is correct and accessible.")
        return None

    # Print extracted content for debugging
    print("Extracted Content Preview:", docs[0].page_content[:500])

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    return chunks


# Function to get response based on fetched data
def get_response(question, chunks):
    if not chunks:
        return "No data available to answer the question."

    # Combine chunks into a single context
    knowledge = "\n\n".join(chunk.page_content for chunk in chunks)

    # Initialize OpenAI LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)  # Correct parameter

    # Construct prompt
    prompt = f"""
    You are an assistant that answers questions based only on the provided knowledge.
    Do not use any external information.

    Knowledge: {knowledge}

    Question: {question}
    """

    # Get the response from the LLM
    response = llm.predict(prompt)  # <-- FIXED: Use `predict()` instead of `invoke()`

    return response


from langchain_community.document_loaders import WebBaseLoader

# Correct way to initialize WebBaseLoader
# loader = WebBaseLoader(
#     "https://stevens.smartcatalogiq.com/en/2021-2022/academic-catalog/college-of-arts-and-letters/faculty/",
# )

# # Fetch and print data
# docs = loader.load()
# for doc in docs:
#     print(doc.page_content)  # Print extracted text


# Example usage of the functions
url = (
    "https://stevens.smartcatalogiq.com/en/2021-2022/academic-catalog/college-of-arts-and-letters/faculty/",
)

chunks = fetch_web_data(url)

if chunks:
    response = get_response("Who is Bradley Fidler", chunks)

    print("Answer:", response)
