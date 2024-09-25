import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from chroma_setup import initialize_client  # Adjust the import to your module name

# Load environment variables from the .env file
load_dotenv()

# Initialize Groq client using API key from .env
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChromaDB client using your new function
chroma_client = initialize_client()

# Check if the collection exists, if not, create it
collection_name = "text_embeddings"
try:
    collection = chroma_client.get_collection(name=collection_name)
except chromadb.api.segment.InvalidCollectionException:
    # If collection doesn't exist, create it
    collection = chroma_client.create_collection(name=collection_name)

def query_llm(query: str):
    # Query for relevant documents
    results = collection.query(query_texts=[query], n_results=5)

    # Flatten and structure the list of documents for better context
    if results["documents"]:
        relevant_docs = [doc for sublist in results["documents"] for doc in sublist]  # Flatten list
    else:
        relevant_docs = []

    # Prepare structured context for the LLM
    context = "\n\n---\n\n".join(f"Document Excerpt:\n{doc}" for doc in relevant_docs)

    # Create a prompt with explicit instructions
    prompt = f"""You are an intelligent assistant. Based on the following context, provide a summarized and well-interpreted answer to the question.
    Context: {context}
    
    Question: {query}
    
    Please provide a detailed and clear answer based on the document."""

    # Generate an answer using Groq
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=500,  # Increase if more detailed responses are needed
    )

    # Output the generated text
    return chat_completion.choices[0].message.content
