# PDF Assistant

📄 PDF Assistant is a powerful tool for processing and querying PDF documents. With a user-friendly interface and advanced features, you can easily upload your PDF files and ask questions about their content.


## Live Demo

Check out the live demo of the project here: [PDF Insight AI](https://subarna00-pdf-insight-ai.hf.space).

## Features

- **PDF Upload**: Easily upload text-based PDF files for processing.
- **Text Extraction**: Automatically extract text from PDF files for analysis.
- **Chunking**: The extracted text is divided into manageable chunks for efficient querying.
- **Embedding**: Text chunks are converted into embeddings for better understanding and context retrieval.
- **ChromaDB Integration**: Save and query embeddings using ChromaDB for efficient information retrieval.
- **Interactive Chat**: Ask questions about the document and receive instant answers.


## Technologies Used

- **Python**: The primary programming language for the application.
- **Streamlit**: For creating the web interface.
- **PyMuPDF**: For PDF file manipulation and text extraction.
- **ChromaDB**: For embedding storage and querying.
- **Streamlit PDF Viewer**: For displaying PDF files in the web interface.

## Installation

To get started with PDF Assistant, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Subaarna/PDF_insight_AI_webApp-LLM-RAG-.git
   cd PDF_insight_AI_webApp-LLM-RAG-
   ```
2. **Create .env file**:
   ```bash
   HUGGINGFACE_API_KEY = "Your_API_KEY"
   GROQ_API_KEY = "Your_Groq's_Key"
   ```
3. **Build docker***:
   ```bash
   docker compose up --build
   ```
## Usage

1. **Upload a PDF:** Click on the "Upload a PDF" button to select and upload your document.
2. **Interact with the Document:** Once the PDF is processed, ask questions in the chat input to retrieve information.
3. **View Chat History:** Check your previous interactions in the chat history section.


![Screenshot 2024-09-25 195145](https://github.com/user-attachments/assets/5c2898fc-3d91-4828-bbe1-bad8949b7ace)
![Screenshot 2024-09-25 195229](https://github.com/user-attachments/assets/4b4e6a47-5c15-4a47-a6fd-21a412111cba)
![Screenshot 2024-09-25 234536](https://github.com/user-attachments/assets/fd45afb7-537f-4931-b87e-14ec005532c9)
