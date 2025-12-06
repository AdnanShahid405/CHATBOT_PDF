# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create temp directory for OCR
RUN mkdir -p temp_ocr

# Set environment variable for Gemini API (CHANGE THIS!)
ENV GEMINI_API=your_gemini_api_key_here

# Expose port
EXPOSE 8000

# Set other environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]