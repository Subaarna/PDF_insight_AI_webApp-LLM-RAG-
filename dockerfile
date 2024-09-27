# Base image with Python 3.10
FROM python:3.10.14

# Set working directory
WORKDIR /app

# Install system dependencies (including Poppler)
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code to the container
COPY . .

# Expose Streamlit's default port (8501)
EXPOSE 8501

# Set environment variables (You can modify this to load variables from Hugging Face Secrets)
ENV STREAMLIT_SERVER_PORT=8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
