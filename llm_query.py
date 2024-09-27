import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from chroma_setup import initialize_client  # Adjust the import to your module name
from embedding import query_chroma_db

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

# def query_llm(query: str, user_id: str = None, document_id: str = None):
#     # Prepare the where clause based on available metadata
#     where = {}
#     if user_id:
#         where['user_id'] = user_id
#     if document_id:
#         where['document_id'] = document_id

#     # Query for relevant documents
#     results = collection.query(query_texts=[query], n_results=5, where=where)

#     # Flatten and structure the list of documents for better context
#     if results["documents"]:
#         relevant_docs = [doc for sublist in results["documents"] for doc in sublist]  # Flatten list
#     else:
#         relevant_docs = []

#     # Prepare structured context for the LLM
#     context = "\n\n---\n\n".join(f"Document Excerpt:\n{doc}" for doc in relevant_docs)

#     # Create a prompt with explicit instructions
#     prompt = f"""You are an intelligent assistant. Based on the following context, provide a summarized and well-interpreted answer to the question.
#     Context: {context}

#     Question: {query}

#     Please provide a detailed and clear answer based on the document."""

#     # Generate an answer using Groq
#     chat_completion = groq_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         model="llama3-8b-8192",
#         max_tokens=500,  # Increase if more detailed responses are needed
#     )


#     # Output the generated text
#     return chat_completion.choices[0].message.content
def query_llm(query: str, user_id: str, document_id: str):
    if not user_id or not document_id:
        raise ValueError("Both user_id and document_id are required for querying.")

    context = query_chroma_db(user_id, document_id, query)

    # Log context before sending to LLM
    print(f"LLM Context: {context}")

    # Check if context is empty
    if not context.strip():
        return "No context found for this query."

    prompt = f"""You are an intelligent assistant. Based on the following context, provide a summarized and well-interpreted answer to the question.
    Context: {context}
    
    Question: {query}
    
    Please provide a detailed and clear answer based on the document."""

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=500,
    )

    return chat_completion.choices[0].message.content
