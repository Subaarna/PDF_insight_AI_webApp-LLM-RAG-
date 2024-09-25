import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import datetime
import uuid  # Added for generating document IDs
from chroma_setup import initialize_client  # Adjust the import to your module name

# Load environment variables from the .env file
load_dotenv()

# HuggingFace embedding model setup
def get_embedding_model():
    return embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

def embed_text_chunks(pages_and_chunks: list[dict]) -> pd.DataFrame:
    embedding_model = get_embedding_model()

    # Embed each chunk
    for item in pages_and_chunks:
        # Ensure the embedding is in 1D (flatten if necessary)
        embedding = embedding_model(input=item["sentence_chunk"])

        # If embedding is returned as a batch, take the first element
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            embedding = embedding[0]

        # Store the flattened embedding back in the dictionary
        item["embedding"] = embedding

    # Convert to DataFrame
    df = pd.DataFrame(pages_and_chunks)
    return df

def save_to_chroma_db(embeddings_df: pd.DataFrame, user_id: str, document_id: str):
    # Use a shared collection for all documents
    collection_name = "text_embeddings"

    client = initialize_client()
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"{user_id}_{document_id}_{i}" for i in range(len(embeddings_df))]
    documents = embeddings_df["sentence_chunk"].tolist()
    embeddings = embeddings_df["embedding"].tolist()
    
    # Add metadata for filtering (user_id and document_id)
    metadatas = [{"user_id": user_id, "document_id": document_id} for _ in range(len(embeddings_df))]

    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

def query_chroma_db(user_id: str, document_id: str, query: str):
    collection_name = "text_embeddings"
    client = initialize_client()
    collection = client.get_collection(name=collection_name)

    # Query the collection using metadata with an $and operator for multiple conditions
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"$and": [{"user_id": user_id}, {"document_id": document_id}]}  # Use $and operator to filter by both user and document
    )
    
    # Extract the document snippets
    documents = results["documents"]
    relevant_docs = [doc for sublist in documents for doc in sublist]  # Flatten the list

    # Combine relevant document excerpts into a single string
    context = "\n\n".join(relevant_docs)

    return context


# Function to generate a unique document ID for each new PDF upload
def generate_document_id() -> str:
    return str(uuid.uuid4())  # Unique document ID
