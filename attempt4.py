import os
import pickle
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import nltk

# Example URLs to fetch data from
urls = [
    "https://www.stevens.edu/school-engineering-science/faculty",
    # "https://www.stevens.edu/school-business/faculty",
    # "https://www.stevens.edu/directory",
    # "https://www.stevens.edu/page-basic/deferring-your-enrollment",
    # "https://www.stevens.edu/academics/graduate-study/academic-policies-and-procedures#policies-procedures-a-i",
    # "https://www.stevens.edu/academics/graduate-study/academic-policies-and-procedures#policies-procedures-j-r",
    # "https://www.stevens.edu/academics/graduate-study/academic-policies-and-procedures#policies-procedures-s-z",
    # "https://www.stevens.edu/academics/graduate-study/graduate-student-handbook",
    # "https://www.stevens.edu/admission-aid/undergraduate-admissions/new-students/health-and-immunization-requirements",
    # "https://www.stevens.edu/counseling-psychological-services",
    # "https://www.stevens.edu/apply-i20",
    # "https://www.stevens.edu/financial-documentation-requirements-newly-admitted"
    "https://www.stevens.edu/cpt-frequently-asked-questions",
    "https://www.stevens.edu/off-campus-employment/isss-f1-students/cpt-work-authorization",
    # "https://www.stevens.edu/information-for-parents-and-families",
    # "https://www.stevens.edu/policies-library",
    # "https://www.stevens.edu/academics/stevensonline/stevens-online",
    # "https://www.stevens.edu/graduate-corporate-education",
    # "https://library.stevens.edu/home",
    # "https://www.stevens.edu/student-health-services/campus/health-resources-and-information",
    # "https://www.stevens.edu/student-life/student-affairs",
    # "https://www.stevens.edu/student-life/student-affairs/student-awards",
    # "https://www.stevens.edu/center-for-complex-systems-and-enterprises",
    # "https://www.stevens.edu/center-for-quantum-science-and-engineering",
    # "https://www.stevens.edu/research/research-centers-and-labs/center-for-decision-technologies/center-for-decision-technologies",
    # "https://www.stevens.edu/center-for-environmental-systems",
    # "https://www.stevens.edu/center-for-healthcare-innovation",
    # "https://www.stevens.edu/center-for-neuromechanics",
    # "https://www.stevens.edu/craft",
    # "https://www.stevens.edu/davidson-laboratory",
    # "https://www.stevens.edu/school-business/hanlon-financial-systems-center",
    # "https://www.stevens.edu/stevens-center-for-sustainability",
    # "https://www.stevens.edu/stevens-institute-for-artificial-intelligence",
    # "https://www.stevens.edu/discover-stevens/leadership-and-vision",
    # "https://www.stevens.edu/discover-stevens/leadership-and-vision/the-presidents-leadership-council",
    # "https://www.stevens.edu/discover-stevens/leadership-and-vision/office-president",
    # "https://www.stevens.edu/discover-stevens/leadership-and-vision/board-of-trustees",
    # "https://www.stevens.edu/office-of-the-provost",
    # "https://www.stevens.edu/admission-aid/graduate-admissions/chat-with-a-student",
    # "https://www.stevens.edu/academics/graduate-study/graduate-funding/assistantships-and-fellowships",
    # "https://www.stevens.edu/admission-aid/graduate-admissions/graduate-programs/graduate-degrees-and-programs",
    # "https://www.stevens.edu/admission-aid/tuition-financial-aid/graduate-costs-and-funding",
    # "https://www.stevens.edu/admissions-aid/graduate-admissions/acceptance-categories",
    # "https://www.stevens.edu/information-for-high-school-counselors",
    # "https://www.stevens.edu/page-basic/public-transportation",
    "https://www.stevens.edu/transportation-and-parking/stevens-shuttle",
    # "https://www.stevens.edu/transportation-and-parking",
    "https://www.stevens.edu/transportation-and-parking/nj-transit",
    # "https://www.stevens.edu/academics/academics-at-stevens",
    # "https://www.stevens.edu/admission-aid/undergraduate-admissions/new-students/housing-and-dining-new-students",
    # "https://www.stevens.edu/admissions",
    # "https://www.stevens.edu/admission-aid/graduate-admissions",
    # "https://www.stevens.edu/apply",
]


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
                chunk_size=1000, chunk_overlap=200
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


# Store the fetched data
fetch_and_store_data(urls, "data_stored.pkl")
