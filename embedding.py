import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import datetime
import uuid  # Added for generating document IDs
from chroma_setup import initialize_client  # Adjust the import to your module name
import numpy as np

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

    for item in pages_and_chunks:
        # Generate the embedding for the sentence chunk
        embedding = embedding_model(input=item["sentence_chunk"])

        # If embedding is a nested list, flatten it
        if isinstance(embedding, list):
            # Flatten the nested list
            embedding = [float(val) for sublist in embedding for val in sublist]
        else:
            raise ValueError(f"Unexpected embedding format: {type(embedding)}")

        # Store the embedding back in the dictionary
        item["embedding"] = embedding

    # Convert the processed data to a DataFrame
    df = pd.DataFrame(pages_and_chunks)
    return df


def save_to_chroma_db(embeddings_df: pd.DataFrame, user_id: str, document_id: str):
    client = initialize_client()
    collection = client.get_or_create_collection(name=f"text_embeddings_{user_id}")

    combined_key = f"{user_id}_{document_id}"

    ids = [f"{combined_key}_{i}" for i in range(len(embeddings_df))]
    documents = embeddings_df["sentence_chunk"].tolist()

    # Convert all embeddings to a flat list of floats
    embeddings = []
    for embedding in embeddings_df["embedding"]:
        # If embedding is a NumPy array, convert it to a flat list
        if isinstance(embedding, np.ndarray):
            embeddings.append(embedding.flatten().tolist())
        else:
            embeddings.append(embedding)  # In case it is already a list of floats

    # Log the combined key and metadata being stored
    metadatas = [{"combined_key": combined_key} for _ in range(len(embeddings_df))]
    print(f"Storing documents with combined_key: {combined_key}")

    collection.add(
        documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas
    )


def query_chroma_db(user_id: str, document_id: str, query: str):
    client = initialize_client()
    collection = client.get_collection(name=f"text_embeddings_{user_id}")

    combined_key = f"{user_id}_{document_id}"

    # Log combined key
    print(f"Querying with combined_key: {combined_key}")

    # Query the collection based on the combined key
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "combined_key": combined_key
        },  # Filter based on the combined metadata key
    )

    # Log the raw results
    print(f"Query Results: {results}")

    # Extract the document snippets
    documents = results.get("documents", [])
    if documents:
        relevant_docs = [
            doc for sublist in documents for doc in sublist
        ]  # Flatten the list
        context = "\n\n".join(relevant_docs)
    else:
        context = "No documents found"

    return context


# Function to generate a unique document ID for each new PDF upload
def generate_document_id() -> str:
    return str(uuid.uuid4())  # Unique document ID
