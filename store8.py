import os
import requests
from pymongo import MongoClient
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.readers.file import PDFReader, EpubReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging
import sys
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Chroma client
vdb = chromadb.PersistentClient(path="./ebooks_vector")

# MongoDB connection string
mongoURI = ""

# Connect to MongoDB
client = MongoClient(mongoURI)

# Select the database and collection
db = client.yeeplatformDatabase
ebooks_collection = db.ebooks

# Base directory for storing ebooks
base_dir = "./data"
os.makedirs(base_dir, exist_ok=True)

# Load the HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Function to retrieve an ebook URL by ID and create Chroma collection
def get_ebook_data_and_collection(ebook_id):
    try:
        # Convert ebook_id to ObjectId
        ebook_id_obj = ObjectId(ebook_id)

        # Get or create Chroma collection
        chroma_collection = vdb.get_or_create_collection(str(ebook_id))
        
        # Find the ebook by ID
        ebook = ebooks_collection.find_one({"_id": ebook_id_obj})
        if ebook:
            logger.info(f"Ebook found: {ebook}")
            # Check for possible URL fields
            url = ebook.get('ebookUrl') or ebook.get('ebookepubUrl') or ebook.get('ebook_url')
            if url:
                return url, chroma_collection
            else:
                logger.error("Ebook URL not found in the document.")
                return None, None
        else:
            logger.error(f"Ebook with ID {ebook_id} not found.")
            return None, None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None, None

# Function to download a file from a URL
def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded file to {local_filename}")
        return local_filename
    except Exception as e:
        logger.error(f"An error occurred while downloading the file: {e}")
        return None

# Main function to process ebook by ID
def process_ebook(ebook_id):
    try:
        # Retrieve ebook data and collection
        url, chroma_collection = get_ebook_data_and_collection(ebook_id)
        if not url or not chroma_collection:
            return

        # Create directory for the ebook
        ebook_dir = os.path.join(base_dir, str(ebook_id))
        os.makedirs(ebook_dir, exist_ok=True)

        # Determine the file extension
        file_extension = url.split('.')[-1].lower()
        if file_extension not in ['pdf', 'epub']:
            logger.error("Unsupported file type.")
            return

        # Create a clear filename
        file_name = f"ebook_{ebook_id}.{file_extension}"
        local_filepath = os.path.join(ebook_dir, file_name)

        # Download the ebook
        download_file(url, local_filepath)

        # Set up the appropriate parser
        if file_extension == 'pdf':
            parser = PDFReader()
        elif file_extension == 'epub':
            parser = EpubReader()
        else:
            logger.error("Unsupported file type.")
            return

        file_extractor = {f".{file_extension}": parser}

        # Load documents
        logger.info(f"Loading documents from {local_filepath}")
        documents = SimpleDirectoryReader(ebook_dir, file_extractor=file_extractor).load_data()

        # Initialize storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, show_progress=True)

        # Create a query engine and perform a query
        query_engine = index.as_query_engine()
        response = query_engine.query("Who is the author?")
        logger.info(f"Query Response: {response}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Please provide an ebook ID.")
        sys.exit(1)
    ebook_id = sys.argv[1]
    process_ebook(ebook_id)
