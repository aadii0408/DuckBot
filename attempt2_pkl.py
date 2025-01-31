import os
import pickle
import openai
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
openai.api_key = os.getenv("OPENAI_API_KEY")
# Set User-Agent manually (to avoid request blocking)
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


# Function to fetch and process web data
def fetch_and_store_data(urls, pickle_filename):
    all_chunks = {}  # Store data for each URL

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

            # Store the chunks for this URL
            all_chunks[url] = chunks

        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
            continue  # Skip to the next URL if there's an error

    # Save the fetched data to a pickle file
    with open(pickle_filename, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Data has been stored in {pickle_filename}")


# Example URLs to fetch data from
urls = [
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
# Store the fetched data
fetch_and_store_data(urls, "stored_data.pkl")
