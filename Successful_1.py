import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
# Load environment variables
load_dotenv()

# Set User-Agent manually (to avoid request blocking)
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


# Function to fetch only the necessary data from web pages
def fetch_relevant_data(question, urls):
    all_responses = []  # Store responses for all URLs

    # Loop through all provided URLs
    for url in urls:
        print(f"Fetching data from: {url}")

        try:
            # Load webpage content
            loader = UnstructuredURLLoader(
                urls=[url], headers={"User-Agent": os.environ["USER_AGENT"]}
            )
            docs = loader.load()

            if not docs:
                print(f"No relevant data found at {url}.")
                continue  # Move to the next URL if no data is found

            # Split the text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            chunks = text_splitter.split_documents(docs)

            # Pass the relevant chunks for answering
            response = get_response(question, chunks)
            if response.strip():  # If the response is meaningful, add it to results
                all_responses.append(f"Answer from {url}:\n{response}\n")

        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
            continue  # Skip to the next URL if there's an error

    if all_responses:
        return "\n\n".join(all_responses)  # Combine all responses
    else:
        return "No relevant information found."


# Function to get response based on fetched data
def get_response(question, chunks):
    if not chunks:
        return "No data available to answer the question."

    # Combine chunks into a single context
    knowledge = "\n\n".join(chunk.page_content for chunk in chunks)

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


# List of URLs to search from

urls = [
    "https://www.stevens.edu/center-for-complex-systems-and-enterprises",
    "https://www.stevens.edu/center-for-quantum-science-and-engineering",
    "https://www.stevens.edu/information-for-high-school-counselors",
    "https://www.stevens.edu/page-basic/public-transportation",
    "https://www.stevens.edu/transportation-and-parking/stevens-shuttle",
    "https://www.stevens.edu/transportation-and-parking",
    "https://www.stevens.edu/transportation-and-parking/nj-transit",
    "https://www.stevens.edu/academics/academics-at-stevens",
    "https://www.stevens.edu/admission-aid/undergraduate-admissions/new-students/housing-and-dining-new-students",
    "https://www.stevens.edu/admissions",
    "https://www.stevens.edu/admission-aid/graduate-admissions",
    "https://www.stevens.edu/apply",
]


# Ask only for the required information
question = "Tell me about Public Transportation at Stevens Institute of Technology."

# Fetch answers for all URLs
response = fetch_relevant_data(question, urls)

print("Final Answer:\n", response)
