import streamlit as st
import fitz  # PyMuPDF
from pdf_processor import read_pdf, process_chunks
from embedding import (
    embed_text_chunks,
    save_to_chroma_db,
    query_chroma_db,
    generate_document_id,
)
from llm_query import query_llm
from chroma_setup import initialize_client
import random
import uuid
from streamlit_pdf_viewer import pdf_viewer  # Import the PDF viewer

# Initialize the ChromaDB client
client = initialize_client()

# Sarcastic lines for PDF processing
sarcastic_lines = [
    "Cooking PDF... 🍳",
    "This may take a while; I'm busy convincing the PDF to cooperate... 🤔",
    "Magically transforming paper into data... ✨",
    "Waving my magic wand... 🪄",
    "Summoning the PDF spirits... 👻",
    "Just a moment while I teach this PDF some manners... 📚",
    "The PDF is contemplating its existence... 🧘‍♂️",
    "The PDF is resisting but I’ll tame it... any minute now!",
]

# Ensure unique user identifier is set
if "user_id" not in st.session_state:
    st.session_state.user_id = str(
        uuid.uuid4()
    )  # Generate a unique user ID for each session

user_id = st.session_state.user_id  # Access the user ID from session state

# Chat history state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Set the page title and icon
st.set_page_config(page_title="PDF Assistant", page_icon="📄")

# Main app layout
st.title("📄 PDF Assistant: Let's Process Your Document")
st.markdown("Upload your Text-Based PDF and ask questions about its content!")

# Step 1: File upload section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    document_id = generate_document_id()
    st.success("File uploaded successfully! Let's get started.")

    # Display a random sarcastic line during upload
    st.markdown(random.choice(sarcastic_lines))

    # Read the PDF content once
    pdf_bytes = uploaded_file.read()  # Read bytes from uploaded file

    # Display the PDF file using streamlit-pdf-viewer
    st.subheader("Uploaded PDF")
    pdf_viewer(pdf_bytes)  # Use the pdf_viewer function from the package

    # Step 2: Process the PDF
    with st.spinner("Processing..."):
        # Display another random sarcastic line during processing
        st.markdown(random.choice(sarcastic_lines))
        pages_and_text = read_pdf(pdf_bytes)  # Pass the pdf_bytes directly
    st.success("Done processing the PDF!")

    # Step 3: Chunk the text
    with st.spinner("Chunking text..."):
        # Display a sarcastic line during chunk processing
        st.markdown(random.choice(sarcastic_lines))
        processed_chunks = process_chunks(pages_and_text)
    st.success("Text chunks are ready!")

    # Step 4: Embed text chunks
    with st.spinner("Embedding text..."):
        # Display a sarcastic line during embedding
        st.markdown(random.choice(sarcastic_lines))
        embeddings_df = embed_text_chunks(processed_chunks)
    st.success("Embedding complete!")

    # Step 5: Save embeddings to ChromaDB
    with st.spinner("Saving embeddings..."):
        # Display a sarcastic line during saving
        st.markdown(random.choice(sarcastic_lines))
        save_to_chroma_db(embeddings_df, user_id, document_id)
    st.success(f"Embeddings saved for document ID: {document_id}")

    # Step 6: Chat interaction
    query = st.chat_input("Ask a question about the document")

    if query:
        with st.spinner("Searching for answers..."):
            # Ensure you provide user_id and document_id when querying
            context = query_chroma_db(user_id, document_id, query)
            response = query_llm(
                query, user_id, document_id
            )  # Pass user_id and document_id

            # Save the query and response to history
            st.session_state.qa_history.append({"question": query, "answer": response})

# Display chat history (if any), including new messages
if st.session_state.qa_history:
    st.subheader("Chat History")
    for interaction in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(interaction["question"])
        with st.chat_message("assistant"):
            st.write(interaction["answer"])

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.qa_history = []
    st.success("Chat history cleared!")
