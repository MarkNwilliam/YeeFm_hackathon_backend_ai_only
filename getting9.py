import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.openai import OpenAI
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Chroma client
vdb = chromadb.PersistentClient(path="./ebooks_vector")

# Base directory for storing ebooks
base_dir = "./data"

# Load the HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Function to check if the ebook file exists in the local directory
def check_ebook_file(ebook_id):
    ebook_dir = os.path.join(base_dir, str(ebook_id))
    if os.path.exists(ebook_dir):
        for file in os.listdir(ebook_dir):
            if file.startswith("ebook_") and (file.endswith(".epub") or file.endswith(".pdf")):
                return os.path.join(ebook_dir, file)
    return None

# Main function to query ebook by ID and query
def query_ebook(ebook_id, query, user_id, title, Description):
    try:
        # Check if the ebook file exists locally
        ebook_file_path = check_ebook_file(ebook_id)
        if not ebook_file_path:
            logger.error(f"Ebook file not found locally for ID: {ebook_id}")
            return "Ebook file not found."

        logger.info(f"Ebook file found: {ebook_file_path}")

        # Extract file name without extension and generate collection name
        file_name = os.path.splitext(os.path.basename(ebook_file_path))[0]
        collection_name = f"{ebook_id}"
        logger.info(f"Collection name: {collection_name}")

        # Retrieve indexed data using the collection name
        chroma_collection = vdb.get_collection(collection_name)
        if not chroma_collection:
            logger.error(f"Collection '{collection_name}' not found.")
            return "Collection not found."

        # Assign Chroma collection as vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load index from stored vectors
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

        # Load or create chat store for user
        chat_store_path = f"./chat_stores/{user_id}_{ebook_id}_chat_store.json"
        if os.path.exists(chat_store_path):
            try:
                chat_store = SimpleChatStore.from_persist_path(chat_store_path)
            except Exception as e:
                logger.error(f"Error loading chat store: {e}")
                chat_store = SimpleChatStore()
        else:
            chat_store = SimpleChatStore()

        # Configure OpenAI language model
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

        
        # Configure chat engine
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096, chat_store=chat_store, chat_store_key=user_id)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are a chatbot, able to have normal interactions, as well as talk"
                " about the contents of the ebook."
                f"Here is the title {title}."
            ),
       
        )

        response = chat_engine.chat(query)

        logger.info(f"Chat Response: {response}")

        # Save chat store
        try:
            chat_store.persist(chat_store_path)
        except Exception as e:
            logger.error(f"Error saving chat store: {e}")

        return response

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 6:
        logger.error("Please provide an ebook ID, query, and user ID.")
        sys.exit(1)
    ebook_id = sys.argv[1]
    query = sys.argv[2]
    user_id = sys.argv[3]
    title = sys.argv[4]
    Description = sys.argv[5]
    response = query_ebook(ebook_id, query, user_id, title, Description)
    print(response)
